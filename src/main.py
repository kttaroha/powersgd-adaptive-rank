"""Entry point for distributed PowerSGD training experiments."""
import os
from datetime import datetime
import argparse
import yaml
import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: F401
from torch.utils.data import DataLoader
import torch.distributed as dist
from typing import Dict, Any, Optional, Tuple

from utils.distributed import setup_distributed, cleanup_distributed
from utils.logger import log_message
from utils.powerSGD_benchmark import PowerSGDBenchmark, measure_network_performance
from utils.network_limiter import NetworkLimiter
from utils.config_adapter import load_and_adapt_config
from utils.seed import set_global_seed
from datasets.mnist import get_mnist_dataloader
from datasets.cifar import get_cifar_dataloader
from datasets.cifar100 import get_cifar100_dataloader
from datasets.imagenet import get_imagenet_dataloader
from models.resnet import get_resnet_model

from train.train import train


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Distributed Training Framework (PowerSGD)")
    parser.add_argument('--config', type=str, required=True, help="Path to the config file")
    parser.add_argument('--rank', type=int, required=True, help="Global rank of the process")
    parser.add_argument('--local_rank', type=int, required=True, help="Local rank on the node")
    parser.add_argument('--world_size', type=int, required=True, help="Total number of processes")
    parser.add_argument('--master_addr', type=str, required=True, help="Master node address")
    parser.add_argument('--master_port', type=int, default=12355, help="Master node port")
    parser.add_argument('--benchmark', action='store_true', help="Enable PowerSGD benchmarking")
    parser.add_argument('--network_test', action='store_true', help="Run network performance test")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to a YAML configuration file.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def get_dataloader(
    dataset_name: str,
    batch_size: int,
    num_replicas: int,
    rank: int,
    train: bool = True
) -> DataLoader:
    """Build a dataset-specific dataloader.

    Args:
        dataset_name: Dataset name.
        batch_size: Mini-batch size.
        num_replicas: Number of distributed replicas.
        rank: Global rank.
        train: Whether to build a training dataloader.

    Returns:
        DataLoader: Configured dataloader.

    Raises:
        ValueError: If the dataset name is unsupported.
    """
    dataset_name = dataset_name.lower()
    if dataset_name == "mnist":
        return get_mnist_dataloader(batch_size, num_replicas, rank, train=train)
    elif dataset_name == "cifar":
        return get_cifar_dataloader(batch_size, num_replicas, rank, train=train)
    elif dataset_name == "cifar100":
        return get_cifar100_dataloader(batch_size, num_replicas, rank, train=train)
    elif dataset_name == "imagenet":
        return get_imagenet_dataloader(batch_size, num_replicas, rank, train=train)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_model(config: Dict[str, Any]) -> torch.nn.Module:
    """Build a model instance from configuration.

    Args:
        config: Experiment configuration.

    Returns:
        torch.nn.Module: Model instance.

    Raises:
        ValueError: If the model name is unsupported.
    """
    dataset_name = config["dataset"]["name"].lower()
    use_cifar_stem = dataset_name in ("cifar", "cifar100")
    model_name = config["model"]["name"]

    if model_name in ("resnet18", "resnet34", "resnet50"):
        return get_resnet_model(
            config["model"]["num_classes"],
            resnet_type=model_name,
            pretrained=config["model"].get("pretrained", False),
            cifar_stem=use_cifar_stem,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def build_scheduler(
    optimizer: optim.Optimizer,
    training_cfg: Dict[str, Any],
    total_epochs: int,
    rank: int,
) -> Optional[Any]:
    """Build a learning-rate scheduler with optional warmup.

    Args:
        optimizer: Optimizer instance.
        training_cfg: Training configuration section.
        total_epochs: Total number of training epochs.
        rank: Global rank.

    Returns:
        Optional[Any]: Scheduler object, or ``None`` if disabled.
    """
    scheduler: Optional[Any] = None
    scheduler_type = training_cfg.get("scheduler", "none")
    warmup_epochs = int(training_cfg.get("warmup_epochs", 0))
    warmup_start_lr = float(training_cfg.get("warmup_start_lr", 0.0))
    warmup_min_factor = float(training_cfg.get("warmup_min_factor", 1e-8))

    if scheduler_type == "step":
        step_size = training_cfg.get("step_size", 30)
        gamma = training_cfg.get("gamma", 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        if rank == 0:
            print(f"[Scheduler] Using StepLR(step_size={step_size}, gamma={gamma})")
    elif scheduler_type == "multistep":
        milestones = training_cfg.get("milestones", [150, 250])
        gamma = training_cfg.get("gamma", 0.1)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        if rank == 0:
            print(f"[Scheduler] Using MultiStepLR(milestones={milestones}, gamma={gamma})")
    elif scheduler_type == "cosine":
        t_max = training_cfg.get("T_max", total_epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        if rank == 0:
            print(f"[Scheduler] Using CosineAnnealingLR(T_max={t_max})")
    else:
        if rank == 0:
            print("[Scheduler] No scheduler is used (scheduler='none' or not set)")

    # Warmup: use LinearLR only, or chain LinearLR -> main scheduler.
    if warmup_epochs > 0:
        base_lr = max(optimizer.param_groups[0]["lr"], 1e-12)
        start_factor = max(warmup_min_factor, warmup_start_lr / base_lr)

        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=start_factor,
                total_iters=warmup_epochs,
            )
            if rank == 0:
                print(f"[Warmup] Using LinearLR for {warmup_epochs} epochs (no main scheduler)")
        else:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=start_factor,
                total_iters=warmup_epochs,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[warmup_epochs],
            )
            if rank == 0:
                print(f"[Warmup] Using LinearLR for {warmup_epochs} epochs before {scheduler_type}")

    return scheduler


def measure_bandwidth_allreduce(
    rank: int,
    world_size: int,
    message_size: int = 64 * 1024 * 1024,
    warmup: int = 5,
    iters: int = 20
):
    """Estimate allreduce bandwidth and latency from repeated all-reduce runs.

    Args:
        rank: Global rank.
        world_size: Number of distributed processes.
        message_size: Message size in bytes for each all-reduce.
        warmup: Number of warmup iterations.
        iters: Number of timed iterations.

    Returns:
        Optional[Tuple[float, float]]: On rank 0, returns
            ``(bandwidth_MBps, latency_ms)``. Returns ``None`` on non-zero ranks.
    """
    import time
    import torch
    import torch.distributed as dist

    if world_size < 2:
        return (0.0, 0.0) if rank == 0 else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.ones(message_size // 4, dtype=torch.float32, device=device)

    dist.barrier()

    for _ in range(warmup):
        dist.all_reduce(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    times = []
    for _ in range(iters):
        dist.barrier()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        dist.all_reduce(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times)

    # Approximate per-rank traffic volume in ring allreduce.
    bytes_transferred = (2.0 * (world_size - 1) / world_size) * message_size

    bandwidth_MBps = (bytes_transferred / (1024.0 * 1024.0)) / avg_time
    bandwidth_mbps = bandwidth_MBps * 8.0

    # Latency here is an approximation from average allreduce duration.
    latency_ms = avg_time * 1000.0

    return (bandwidth_MBps, latency_ms) if rank == 0 else None


def save_env_metrics_txt(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Persist measured environment metrics to a text file.

    Args:
        config: Experiment configuration with measured environment metrics.
        args: Parsed command-line arguments.
    """
    save_dir = config.get("save_dir", "./results")
    os.makedirs(save_dir, exist_ok=True)

    env = config.get("env_metrics", {})
    bw_MBps = env.get("bandwidth_MBps", None)
    bw_mbps = env.get("bandwidth_mbps", None)
    lat_ms = env.get("latency_ms", None)

    net = config.get("network_limits", {})
    net_enable = bool(net.get("enable", False))
    egress = net.get("egress_mbps", None)
    ingress = net.get("ingress_mbps", None)
    iface = net.get("interface", None)
    target_ip = net.get("target_ip", None)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(save_dir, f"env_metrics_rank0_{ts}.txt")

    lines = [
        f"timestamp: {ts}",
        f"world_size: {args.world_size}",
        f"rank: {args.rank}",
        f"master_addr: {args.master_addr}",
        f"master_port: {args.master_port}",
        "",
        "[measured_allreduce]",
        f"bandwidth_MBps: {bw_MBps}",
        f"bandwidth_mbps: {bw_mbps}",
        f"latency_ms: {lat_ms}",
        "",
        "[network_limits]",
        f"enable: {net_enable}",
        f"interface: {iface}",
        f"egress_mbps: {egress}",
        f"ingress_mbps: {ingress}",
        f"target_ip: {target_ip}",
        "",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines))

    print(f"[ENV] saved env metrics to: {path}")


def measure_and_store_env_metrics(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Measure, broadcast, and store environment metrics in config.

    Args:
        config: Experiment configuration dictionary.
        args: Parsed command-line arguments.
    """
    env_bandwidth = 0.0
    env_latency = 0.0

    bw_lat = measure_bandwidth_allreduce(args.rank, args.world_size)
    if bw_lat is not None and args.rank == 0:
        env_bandwidth, env_latency = bw_lat
        print(f"[ENV] measured on rank0: bandwidth={env_bandwidth:.2f} MB/s, latency={env_latency:.3f} ms")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_tensor = torch.tensor([env_bandwidth, env_latency], dtype=torch.float32, device=dev)
    dist.broadcast(env_tensor, src=0)

    env_bandwidth_mbps = float(env_tensor[0].item()) * 8.0
    env_latency_ms = float(env_tensor[1].item())

    config.setdefault("env_metrics", {})
    config["env_metrics"]["bandwidth_MBps"] = float(env_tensor[0].item())
    config["env_metrics"]["bandwidth_mbps"] = env_bandwidth_mbps
    config["env_metrics"]["latency_ms"] = env_latency_ms

    if args.rank == 0:
        print(
            f"[ENV] broadcast bandwidth={config['env_metrics']['bandwidth_MBps']:.2f} MB/s "
            f"({env_bandwidth_mbps:.2f} Mbps), latency={env_latency_ms:.3f} ms"
        )
        save_env_metrics_txt(config, args)


def build_data_loaders(
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, calibration, and evaluation dataloaders.

    Args:
        config: Experiment configuration dictionary.
        args: Parsed command-line arguments.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Training, calibration,
        and evaluation dataloaders.
    """
    dataset_name = config["dataset"]["name"]
    batch_size = config["dataset"]["batch_size"]

    train_dataloader = get_dataloader(
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_replicas=args.world_size,
        rank=args.rank,
        train=True,
    )

    calib_dataloader = get_dataloader(
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_replicas=args.world_size,
        rank=args.rank,
        train=True,
    )

    if args.rank == 0:
        eval_dataloader_temp = get_dataloader(
            dataset_name=dataset_name,
            batch_size=batch_size,
            num_replicas=args.world_size,
            rank=args.rank,
            train=False,
        )
        eval_dataset = eval_dataloader_temp.dataset
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=eval_dataloader_temp.num_workers,
            pin_memory=eval_dataloader_temp.pin_memory,
        )
    else:
        eval_dataloader = get_dataloader(
            dataset_name=dataset_name,
            batch_size=batch_size,
            num_replicas=args.world_size,
            rank=args.rank,
            train=False,
        )

    return train_dataloader, calib_dataloader, eval_dataloader


def build_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """Build an SGD optimizer from configuration.

    Args:
        model: Model whose parameters are optimized.
        config: Experiment configuration dictionary.

    Returns:
        optim.Optimizer: Configured SGD optimizer.
    """
    return torch.optim.SGD(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        momentum=config["training"].get("momentum", 0.9),
        weight_decay=config["training"].get("weight_decay", 0.0001),
        nesterov=bool(config["training"].get("nesterov", False)),
    )


def init_network_limiter(config: Dict[str, Any], rank: int) -> Optional[NetworkLimiter]:
    """Initialize and optionally apply network limits.

    Args:
        config: Experiment configuration dictionary.
        rank: Global rank.

    Returns:
        Optional[NetworkLimiter]: Initialized limiter if enabled, otherwise ``None``.
    """
    if not config.get("network_limits", {}).get("enable", False):
        return None
    limiter = NetworkLimiter(config)
    if rank == 0:
        limiter.apply_limits()
    return limiter


def run_network_test_if_enabled(args: argparse.Namespace) -> None:
    """Run optional network benchmark and print summary on rank 0.

    Args:
        args: Parsed command-line arguments.
    """
    if not args.network_test:
        return
    if args.rank == 0:
        print("Running network performance test...")
    network_results = measure_network_performance(args.world_size)
    if args.rank == 0 and network_results:
        print("Network Performance Results:")
        for key, value in network_results.items():
            if "bandwidth" in key:
                print(f"  {key}: {value:.2f} MB/s")
            elif "latency" in key:
                print(f"  {key}: {value:.2f} ms")
    dist.barrier()


def run_training_pipeline(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Build training objects and launch the training loop.

    Args:
        config: Experiment configuration dictionary.
        args: Parsed command-line arguments.
    """
    train_dataloader, calib_dataloader, eval_dataloader = build_data_loaders(config, args)

    print(f"the size of train data {len(train_dataloader.dataset)}")
    print(f"the size of eval data {len(eval_dataloader.dataset)}")

    model = build_model(config)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)

    training_cfg = config.get("training", {})
    scheduler = build_scheduler(
        optimizer=optimizer,
        training_cfg=training_cfg,
        total_epochs=int(config["training"]["epochs"]),
        rank=args.rank,
    )

    if args.benchmark:
        config["benchmark_enabled"] = True

    train(
        rank=args.rank,
        local_rank=args.local_rank,
        world_size=args.world_size,
        model=model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=config["training"]["epochs"],
        eval_dataloader=eval_dataloader,
        config=config,
        scheduler=scheduler,
        calib_dataloader=calib_dataloader,
    )


def main() -> None:
    """Run the end-to-end distributed training workflow."""
    args = parse_args()
    config = load_and_adapt_config(args.config, args.master_addr, args.world_size)

    print("start")

    # Seed all frameworks for reproducibility.
    seed = int(config.get("seed", 42))
    set_global_seed(seed)

    # Initialize optional network limiter.
    network_limiter = init_network_limiter(config, args.rank)

    # Initialize distributed process group.
    setup_distributed(args.rank, args.world_size, args.master_addr, args.master_port)

    try:
        measure_and_store_env_metrics(config, args)
    except Exception as e:
        if args.rank == 0:
            print(f"[ENV] measurement failed: {e}")
    try:
        log_message(args.rank, f"Using configuration: {config}")

        # Initial synchronization across all ranks.
        dist.barrier()

        run_network_test_if_enabled(args)
        run_training_pipeline(config, args)

    finally:
        if network_limiter:
            network_limiter.cleanup()
        cleanup_distributed()


if __name__ == "__main__":
    main()
