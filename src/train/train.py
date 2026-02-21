"""Training loop and adaptive rank-control logic for PowerSGD experiments."""
import csv
import os
from typing import Optional, Dict, Any, List, Tuple
import math
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import torch.distributed as dist
from contextlib import nullcontext
from datetime import datetime

from torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook import (
    powerSGD_hook,
    PowerSGDState,
)
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook

from src.evaluate.evaluate import evaluate
from src.utils.powerSGD_benchmark import PowerSGDBenchmark

# ACCORDION decision defaults (kept behavior-compatible with previous code)
FORCE_MAX_EPOCHS = 20
RANK_COOLDOWN_EPOCHS = 10
LR_COOLDOWN_EPOCHS = 20
GRAD_COMPARE_STRIDE_EPOCHS = 10
CRITICAL_CHANGE_THRESHOLD = 0.5


def save_results_to_csv(results: List[Dict[str, Any]], save_dir: str, csv_name: str) -> None:
    """Write epoch-level results to CSV, replacing an existing file.

    Args:
        results: List of epoch-level result dictionaries.
        save_dir: Output directory.
        csv_name: Output CSV filename.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, csv_name)
    if os.path.exists(save_path):
        os.remove(save_path)
    with open(save_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


# ---- Timing helpers ----
def _cuda_sync(device: torch.device):
    """Synchronize CUDA device if needed.

    Args:
        device: Target torch device.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _avg_across_ranks(val: float, device: torch.device) -> float:
    """Compute mean scalar value across distributed ranks.

    Args:
        val: Local scalar value.
        device: Target torch device.

    Returns:
        float: Mean value across ranks.
    """
    t = torch.tensor([val], dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return float(t.item())


def _total_param_bytes(module: nn.Module) -> int:
    """Compute total parameter size in bytes.

    Args:
        module: Target module.

    Returns:
        int: Total parameter bytes.
    """
    total = 0
    for p in module.parameters():
        total += p.numel() * p.element_size()
    return int(total)


def select_best_rank_from_calibration_stats(
    stats: List[Dict[str, Any]],
    ranks: List[int],
    threshold: float,
) -> int:
    """Select a calibrated rank from communication-ratio statistics.

    Selection rule:
        1. Choose the maximum rank whose communication ratio is not larger
           than ``threshold``.
        2. If no rank is feasible, choose the minimum candidate rank.
        3. If selected rank is 1 and rank 2 exists, bump to 2.

    Args:
        stats: Calibration rows including ``rank`` and ``comm_ratio``.
        ranks: Candidate ranks evaluated during calibration.
        threshold: Maximum acceptable communication ratio.

    Returns:
        int: Selected rank.
    """
    feasible = [s for s in stats if float(s["comm_ratio"]) <= float(threshold)]
    if len(feasible) == 0:
        best = min(ranks)
    else:
        best = max([int(s["rank"]) for s in feasible])
    if best == 1 and any(int(r) > 1 for r in ranks):
        best = 2
    return int(best)


def apply_calibration_best_as_max_rank(
    min_rank: int,
    max_rank: int,
    best_rank: int,
) -> Tuple[int, int]:
    """Apply calibration output to effective rank bounds.

    Args:
        min_rank: Minimum allowed rank.
        max_rank: Maximum allowed rank.
        best_rank: Rank selected by calibration.

    Returns:
        Tuple[int, int]: Updated ``(min_rank, effective_max_rank)``.
    """
    min_rank = int(min_rank)
    max_rank = int(max_rank)
    best_rank = int(best_rank)
    clamped_best = max(min_rank, min(max_rank, best_rank))
    eff_max = max(min_rank, min(max_rank, clamped_best))
    if min_rank > eff_max:
        min_rank = eff_max
    return int(min_rank), int(eff_max)


def decide_accordion_target_rank(
    *,
    epoch: int,
    current_rank: int,
    min_rank: int,
    max_rank: int,
    end_norm: float,
    past_value: Optional[float],
    lr_cooldown: int,
    rank_cooldown: int,
    force_max_epochs: int = FORCE_MAX_EPOCHS,
    critical_threshold: float = CRITICAL_CHANGE_THRESHOLD,
) -> Tuple[int, Optional[float], str]:
    """Decide target rank for ACCORDION at epoch end.

    Args:
        epoch: Current epoch.
        current_rank: Current matrix rank.
        min_rank: Minimum allowed rank.
        max_rank: Maximum allowed rank.
        end_norm: Current accumulated gradient norm.
        past_value: Accumulated gradient norm from comparison stride.
        lr_cooldown: Remaining cooldown epochs after LR changes.
        rank_cooldown: Remaining cooldown epochs after rank changes.
        force_max_epochs: Epoch span that forces max rank at early training.
        critical_threshold: Threshold for critical change-rate detection.

    Returns:
        Tuple[int, Optional[float], str]: Target rank, gradient change rate,
        and regime label.
    """
    if (int(epoch) < int(force_max_epochs)) or (int(lr_cooldown) > 0):
        return int(max_rank), None, "force_max"
    if int(rank_cooldown) > 0:
        return int(current_rank), None, "rank_cooldown"

    change_rate = 0.0
    if past_value is not None and float(past_value) > 0.0:
        change_rate = abs(float(end_norm) - float(past_value)) / float(past_value)
        critical = change_rate >= float(critical_threshold)
        target_rank = int(max_rank if critical else min_rank)
        regime = "critical" if critical else "not_critical"
        return target_rank, float(change_rate), regime

    return int(current_rank), float(change_rate), "insufficient_history"


def accordion_candidates_from_best(
    base_rank: int,
    min_rank: int,
    max_rank: int,
) -> List[int]:
    """Build ACCORDION candidate ranks from a base rank.

    Args:
        base_rank: Base rank from calibration.
        min_rank: Minimum allowed rank.
        max_rank: Maximum allowed rank.

    Returns:
        List[int]: Candidate ranks in ``[low, mid, high]`` order.
    """
    base_rank = int(base_rank)
    min_rank = int(min_rank)
    max_rank = int(max_rank)

    base_rank = max(min_rank, min(max_rank, base_rank))

    low = min_rank
    mid = max(min_rank, min(max_rank, base_rank // 2))
    high = max(min_rank, min(max_rank, base_rank))

    if base_rank == min_rank:
        high = min(max_rank, min_rank + 1)
        mid = min_rank

    return [low, mid, high]


def rank_from_schedule(epoch: int, schedule: List[Dict[str, Any]], default_rank: int) -> int:
    """Return rank for an epoch according to rank-schedule ranges.

    Args:
        epoch: Current epoch.
        schedule: List of range dictionaries.
        default_rank: Rank used when no range matches.

    Returns:
        int: Rank for the given epoch.
    """
    for item in schedule:
        start = int(item.get("start_epoch", 0))
        end = int(item.get("end_epoch", -1))
        rank = int(item.get("rank", default_rank))
        if start <= epoch <= end:
            return rank
    return int(default_rank)


# ----------------------------
# Dynamic PowerSGD Hook
# - supports "start_powerSGD_iter" gating (pre-start uses allreduce)
# - supports dynamic rank updates
# ----------------------------
class DynamicPowerSGDHook:
    """DDP communication hook with dynamic rank updates and start gating.

    Args:
        initial_rank: Initial matrix approximation rank.
        use_error_feedback: Whether PowerSGD error feedback is enabled.
        start_powerSGD_iter_cfg: Iteration index where PowerSGD becomes active.
        config: PowerSGD hook configuration dictionary.
        benchmark: Optional benchmark collector.
    """
    def __init__(
        self,
        initial_rank: int,
        use_error_feedback: bool,
        start_powerSGD_iter_cfg: int,
        config: Optional[Dict[str, Any]] = None,
        benchmark: Optional[PowerSGDBenchmark] = None,
    ):
        """Initialize dynamic PowerSGD hook state."""
        self.current_rank = int(initial_rank)
        self.use_error_feedback = bool(use_error_feedback)
        self.config = config or {}
        self.benchmark = benchmark
        self.state: Optional[PowerSGDState] = None

        # gating
        self._start_powerSGD_iter_cfg = int(start_powerSGD_iter_cfg)
        self._current_iter = 0

        # process group for allreduce fallback
        self.process_group = dist.group.WORLD

        # PyTorch comm hook requirements
        self.__name__ = "DynamicPowerSGDHook"
        self.__qualname__ = "DynamicPowerSGDHook"
        self.__module__ = "__main__"

        self._create_state()

    def _create_state(self):
        """Recreate rank-dependent ``PowerSGDState``."""
        try:
            # PowerSGDState's own start_powerSGD_iter is internal; we handle gating ourselves.
            start_iter_internal = 2 if self.use_error_feedback else 1
            cfg = self.config

            self.state = PowerSGDState(
                process_group=self.process_group,
                matrix_approximation_rank=self.current_rank,
                start_powerSGD_iter=start_iter_internal,
                min_compression_rate=float(cfg.get("min_compression_rate", 2)),
                use_error_feedback=self.use_error_feedback,
                warm_start=bool(cfg.get("warm_start", True)),
                orthogonalization_epsilon=float(cfg.get("orthogonalization_epsilon", 1e-8)),
                random_seed=int(cfg.get("random_seed", 42)),
                compression_stats_logging_frequency=int(cfg.get("compression_stats_logging_frequency", 1000)),
                batch_tensors_with_same_shape=bool(cfg.get("batch_tensors_with_same_shape", False)),
            )
        except Exception as e:
            print(f"[PowerSGD] failed to create PowerSGDState: {e}")
            self.state = None

    def set_iter(self, it: int):
        """Update current global iteration for start-gating.

        Args:
            it: Global iteration index.
        """
        self._current_iter = int(it)

    def set_start_iter(self, start_it: int):
        """Set start iteration threshold for PowerSGD activation.

        Args:
            start_it: Activation iteration.
        """
        self._start_powerSGD_iter_cfg = int(start_it)

    def get_start_iter(self) -> int:
        """Return start iteration threshold for PowerSGD activation.

        Returns:
            int: Activation iteration.
        """
        return int(self._start_powerSGD_iter_cfg)

    def update_rank(self, new_rank: int):
        """Update rank and recreate internal PowerSGD state if needed.

        Args:
            new_rank: New matrix approximation rank.
        """
        new_rank = int(new_rank)
        if new_rank != self.current_rank:
            print(f"[PowerSGD] update rank {self.current_rank} -> {new_rank}")
            self.current_rank = new_rank
            self._create_state()

    def __call__(self, state, bucket):
        """Run communication hook with all-reduce fallback and benchmarking.

        Args:
            state: DDP hook state (unused in this wrapper).
            bucket: DDP gradient bucket.

        Returns:
            torch.futures.Future: Future resolving to synchronized bucket tensor.
        """
        # Pre-start: use standard allreduce to keep training correct
        if self._current_iter < self._start_powerSGD_iter_cfg:
            fut = allreduce_hook(self.process_group, bucket)
            if self.benchmark is not None:
                start_t = time.perf_counter()

                def _mark_comm_time(fut_obj):
                    """Record communication timing for pre-start all-reduce."""
                    self.benchmark.last_communication_time += time.perf_counter() - start_t
                    try:
                        return fut_obj.value()
                    except Exception:
                        return fut_obj

                return fut.then(_mark_comm_time)
            return fut

        if self.state is None:
            # fallback: allreduce (not "return bucket.buffer()" which would skip sync)
            fut = allreduce_hook(self.process_group, bucket)
            if self.benchmark is not None:
                start_t = time.perf_counter()

                def _mark_comm_time(fut_obj):
                    """Record communication timing for fallback all-reduce."""
                    self.benchmark.last_communication_time += time.perf_counter() - start_t
                    try:
                        return fut_obj.value()
                    except Exception:
                        return fut_obj

                return fut.then(_mark_comm_time)
            return fut

        if self.benchmark is not None:
            self.benchmark.start_compression()

        fut = powerSGD_hook(self.state, bucket)

        if self.benchmark is not None:
            _ = self.benchmark.end_compression()
            self.benchmark.start_decompression()
            start_t = time.perf_counter()

            def _mark_comm_time(fut_obj):
                """Record communication timing for PowerSGD path."""
                self.benchmark.last_communication_time += time.perf_counter() - start_t
                try:
                    return fut_obj.value()
                except Exception:
                    return fut_obj

            return fut.then(_mark_comm_time)

        return fut

    def get_name(self):
        """Return hook display name.

        Returns:
            str: Hook name.
        """
        return "DynamicPowerSGDHook"


# ---- Calibration (rank sweep by timing) ----
@torch.no_grad()
def _set_bn_eval(module: nn.Module):
    """Set batch-normalization layers to eval mode.

    Args:
        module: Target module.
    """
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
            m.eval()


def calibrate_best_rank(
    *,
    rank: int,
    device: torch.device,
    model_ddp: DDP,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    train_dataloader: DataLoader,
    dynamic_hook: DynamicPowerSGDHook,
    benchmark: PowerSGDBenchmark,
    calib_cfg: Dict[str, Any],
) -> Tuple[int, List[Dict[str, Any]]]:
    """Run calibration timing sweep and select best rank.

    Args:
        rank: Global rank.
        device: Target torch device.
        model_ddp: DDP model.
        optimizer: Optimizer instance.
        criterion: Loss function.
        train_dataloader: Dataloader used for calibration iterations.
        dynamic_hook: Dynamic PowerSGD communication hook.
        benchmark: Benchmark collector.
        calib_cfg: Calibration configuration dictionary.

    Returns:
        Tuple[int, List[Dict[str, Any]]]: Selected rank and calibration rows.

    Raises:
        RuntimeError: If the dataloader does not provide enough batches.
    """
    enable = bool(calib_cfg.get("enable", False))
    if not enable:
        return dynamic_hook.current_rank, []

    # ranks to test
    ranks = calib_cfg.get("ranks", list(range(1, 11)))
    ranks = [int(r) for r in ranks]
    ranks = sorted(list(set(ranks)))

    warmup_iters = int(calib_cfg.get("warmup_iters", 5))
    measure_iters = int(calib_cfg.get("measure_iters", 20))
    max_batches = int(calib_cfg.get("max_batches", warmup_iters + measure_iters))
    threshold = float(calib_cfg.get("comm_ratio_threshold", 0.30))
    force_power_start_iter = int(calib_cfg.get("force_start_iter", 0))

    if rank == 0:
        print(f"[CALIB] enable=True ranks={ranks}")
        print(f"[CALIB] warmup_iters={warmup_iters} measure_iters={measure_iters} threshold={threshold}")
        print(f"[CALIB] force_start_iter={force_power_start_iter}")

    # Save initial states (so each rank test is fair)
    base_model_state = copy.deepcopy(model_ddp.module.state_dict())
    base_optim_state = copy.deepcopy(optimizer.state_dict())

    # Temporarily force PowerSGD to start immediately during calibration
    saved_start_iter = dynamic_hook.get_start_iter()
    dynamic_hook.set_start_iter(force_power_start_iter)

    # Use a fixed batch stream for all ranks: take first max_batches from loader
    # NOTE: keep it simple; for true determinism you might also fix sampler epoch.
    batches = []
    it = iter(train_dataloader)
    for _ in range(max_batches):
        try:
            batches.append(next(it))
        except StopIteration:
            break
    if len(batches) < (warmup_iters + measure_iters):
        raise RuntimeError(f"[CALIB] not enough batches: got={len(batches)} needed={warmup_iters + measure_iters}")

    # measure helper (with/without sync)
    def run_iters(with_sync: bool, cur_iter_base: int) -> float:
        """Run a fixed number of iterations for calibration timing.

        Args:
            with_sync: Whether to enable DDP synchronization.
            cur_iter_base: Base global iteration index for hook gating.

        Returns:
            float: Average iteration time over measured iterations.
        """
        model_ddp.train()
        total = 0.0

        # NOTE: we DO backward+step to trigger DDP comm hook
        # For no_sync mode, we wrap DDP.no_sync to skip comm.
        for i in range(warmup_iters + measure_iters):
            data, target = batches[i]
            data, target = data.to(device), target.to(device)

            # set comm hook iteration counter (affects gating)
            dynamic_hook.set_iter(cur_iter_base + i)

            ctx = nullcontext()
            if not with_sync:
                ctx = model_ddp.no_sync()

            _cuda_sync(device)
            t0 = time.perf_counter()

            with ctx:
                out = model_ddp(data)
                loss = criterion(out, target)
                loss.backward()

            optimizer.zero_grad(set_to_none=True)

            _cuda_sync(device)
            t1 = time.perf_counter()

            # measure only after warmup
            if i >= warmup_iters:
                total += (t1 - t0)

        return total / float(measure_iters)

    stats: List[Dict[str, Any]] = []

    # A monotonically increasing "virtual iter" base so hook gating behaves consistently
    virtual_iter_base = 0

    for r in ranks:
        # restore states
        model_ddp.module.load_state_dict(base_model_state)
        optimizer.load_state_dict(base_optim_state)

        # update rank
        dynamic_hook.update_rank(r)

        # barriers to align all ranks timing
        dist.barrier()

        # measure no-sync (no comm)
        t_nosync_local = run_iters(with_sync=False, cur_iter_base=virtual_iter_base)
        dist.barrier()

        # measure sync (with comm)
        t_sync_local = run_iters(with_sync=True, cur_iter_base=virtual_iter_base + 10_000)
        dist.barrier()

        # Average across ranks to reduce noise
        t_nosync = _avg_across_ranks(t_nosync_local, device)
        t_sync = _avg_across_ranks(t_sync_local, device)

        comm_overhead = max(0.0, t_sync - t_nosync)
        comm_ratio = (comm_overhead / t_sync) if t_sync > 0 else 1.0

        # approximate bytes (for logging only)
        full_bytes = _total_param_bytes(model_ddp.module)
        # PowerSGD communication roughly scales with r, but we keep it as "info" not used for selection
        approx_comp_bytes = None
        try:
            r_max = max(ranks)
            approx_comp_bytes = int(full_bytes * (float(r) / float(r_max)))
        except Exception:
            approx_comp_bytes = None

        row = {
            "rank": r,
            "t_nosync_s": t_nosync,
            "t_sync_s": t_sync,
            "comm_overhead_s": comm_overhead,
            "comm_ratio": comm_ratio,
            "full_param_bytes": full_bytes,
            "approx_comp_bytes": approx_comp_bytes,
        }
        stats.append(row)

        if rank == 0:
            print(
                f"[CALIB] r={r:2d} "
                f"t_nosync={t_nosync:.6f}s t_sync={t_sync:.6f}s "
                f"comm_over={comm_overhead:.6f}s ratio={comm_ratio:.3f}"
            )

        # advance virtual base so later ranks don't reuse same iter ids
        virtual_iter_base += 50_000

    # restore gating start_iter
    dynamic_hook.set_start_iter(saved_start_iter)

    # choose best rank from measured stats
    best = select_best_rank_from_calibration_stats(stats=stats, ranks=ranks, threshold=threshold)

    if rank == 0:
        print(f"[CALIB] best_rank={best} (threshold={threshold})")

    # optionally save calibration stats
    if rank == 0:
        save_dir = calib_cfg.get("save_dir", None)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            csv_path = os.path.join(save_dir, calib_cfg.get("csv_name", "calibration_rank_stats.csv"))
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
                writer.writeheader()
                writer.writerows(stats)
            print(f"[CALIB] wrote stats: {csv_path}")

    # broadcast chosen rank to all ranks
    best_t = torch.tensor([best], dtype=torch.int32, device=device)
    dist.broadcast(best_t, src=0)
    best = int(best_t.item())

    # ----------------------------
    # Restore baseline state after calibration before starting real training.
    # 1) Restore baseline model/optimizer state on all ranks.
    model_ddp.module.load_state_dict(base_model_state)
    optimizer.load_state_dict(base_optim_state)
    dist.barrier()

    # 2) Broadcast parameters/buffers from rank0 to enforce full synchronization.
    with torch.no_grad():
        for p in model_ddp.module.parameters():
            dist.broadcast(p.data, src=0)
        for b in model_ddp.module.buffers():
            dist.broadcast(b.data, src=0)
    dist.barrier()

    return best, stats


def train(
    rank: int,
    local_rank: int,
    world_size: int,
    model: nn.Module,
    train_dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epochs: int,
    eval_dataloader: Optional[DataLoader] = None,
    config: Optional[Dict[str, Any]] = None,
    scheduler: Optional[Any] = None,
    calib_dataloader: Optional[DataLoader] = None
) -> None:
    """Run distributed training with calibration and adaptive rank control.

    Args:
        rank: Global rank.
        local_rank: Local GPU rank.
        world_size: Number of distributed processes.
        model: Training model.
        train_dataloader: Training dataloader.
        optimizer: Optimizer instance.
        criterion: Loss function.
        epochs: Number of training epochs.
        eval_dataloader: Optional evaluation dataloader.
        config: Optional experiment configuration.
        scheduler: Optional LR scheduler.
        calib_dataloader: Optional dataloader for calibration.
    """
    device = torch.device(f"cuda:{local_rank}")
    # SyncBatchNorm and DDP
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model.to(device), device_ids=[local_rank])

    # Benchmark
    benchmark_save_dir = config.get("benchmark_save_dir", "./benchmark_results") if config else "./benchmark_results"
    benchmark = PowerSGDBenchmark(rank, world_size, benchmark_save_dir)

    # configs
    power_cfg = config.get("powerSGD", {}) if config else {}
    comp_cfg = config.get("compression", {}) if config else {}
    training_cfg = config.get("training", {}) if config else {}
    warmup_epochs = int(training_cfg.get("warmup_epochs", 0))
    scheduler_name = str(training_cfg.get("scheduler", "")).lower()
    step_size = int(training_cfg.get("step_size", 0))
    milestones = training_cfg.get("milestones", [])
    use_planned_lr_change = (
        (scheduler_name == "step" and step_size > 0)
        or (scheduler_name == "multistep" and len(milestones) > 0)
    )
    rank_schedule_cfg = comp_cfg.get("rank_schedule", {}) if comp_cfg else {}
    rank_schedule_enabled = bool(rank_schedule_cfg)
    rank_schedule = rank_schedule_cfg.get("ranges", []) if rank_schedule_enabled else []
    rank_schedule_default = int(rank_schedule_cfg.get("default_rank", 1))

    # base rank
    base_rank = int(comp_cfg.get("base_rank", power_cfg.get("matrix_approximation_rank", 1)))
    use_error_feedback = bool(power_cfg.get("use_error_feedback", True))

    # start iteration (delay enabling compression)
    start_power_from_comp = comp_cfg.get("start_power_iterations", None)
    if start_power_from_comp is not None:
        start_powerSGD_iter_cfg = int(start_power_from_comp)
    else:
        start_powerSGD_iter_cfg = int(power_cfg.get("start_power_iterations", 100))

    # switches
    enable_powerSGD = bool(comp_cfg.get("enable_powerSGD", True))
    use_fixed_rank = bool(comp_cfg.get("use_fixed_rank", False))
    enable_stage_adjust = bool(comp_cfg.get("enable_stage_adjust", False))
    enable_adaptive_sync = bool(comp_cfg.get("enable_adaptive_sync", False))

    fixed_rank_value = int(comp_cfg.get("fixed_rank_value", base_rank))
    min_rank = int(comp_cfg.get("min_rank", 1))
    max_rank = int(comp_cfg.get("max_rank", max(8, base_rank)))

    # ACCORDION controls (disabled by default unless configured).
    accordion_cfg = config.get("accordion", {}) if config else {}
    accordion_enabled = bool(accordion_cfg.get("enable", False))

    print(
        f"[COMPRESSION] enable_powerSGD={enable_powerSGD}, use_fixed_rank={use_fixed_rank}, "
        f"accordion={accordion_enabled}, base_rank={base_rank}, fixed_rank={fixed_rank_value}, "
        f"rank_range=[{min_rank}, {max_rank}], start_power_iterations={start_powerSGD_iter_cfg}"
    )
    if rank_schedule_enabled:
        print(
            f"[RANK-SCHEDULE] enabled default_rank={rank_schedule_default} "
            f"ranges={rank_schedule}"
        )

    # Hook registration
    dynamic_hook = None
    current_matrix_rank = base_rank
    accordion_current_rank = base_rank
    cands = None

    if enable_powerSGD:
        dynamic_hook = DynamicPowerSGDHook(
            initial_rank=base_rank,
            use_error_feedback=use_error_feedback,
            start_powerSGD_iter_cfg=start_powerSGD_iter_cfg,
            config=power_cfg,
            benchmark=benchmark,
        )
        try:
            model.register_comm_hook(None, dynamic_hook)
            print("[PowerSGD] registered DynamicPowerSGDHook")
        except Exception as e:
            print(f"[PowerSGD] failed to register hook: {e}")
            dynamic_hook = None
            enable_powerSGD = False
    else:
        print("[PowerSGD] disabled (pure DDP allreduce)")

    # Calibration (rank sweep) before training.
    calib_cfg = comp_cfg.get("calibration", {}) if comp_cfg else {}
    if enable_powerSGD and dynamic_hook is not None and bool(calib_cfg.get("enable", False)):
        calib_dataloader = calib_dataloader if calib_dataloader is not None else train_dataloader
        best_rank, _stats = calibrate_best_rank(
            rank=rank,
            device=device,
            model_ddp=model,
            optimizer=optimizer,
            criterion=criterion,
            train_dataloader=calib_dataloader,
            dynamic_hook=dynamic_hook,
            benchmark=benchmark,
            calib_cfg=calib_cfg,
        )
        # apply best rank
        best_rank = max(min_rank, min(max_rank, int(best_rank)))
        dynamic_hook.update_rank(best_rank)
        current_matrix_rank = best_rank
        accordion_cfg = config.get("accordion", {}) if config else {}
        accordion_enabled = bool(accordion_cfg.get("enable", False))
        compression_cfg = config.get("compression", {}) if config else {}
        min_rank = int(compression_cfg.get("min_rank", 1))
        max_rank = int(compression_cfg.get("max_rank", 10))
        # Treat calibration best_rank as the effective max_rank.
        min_rank, max_rank = apply_calibration_best_as_max_rank(
            min_rank=min_rank,
            max_rank=max_rank,
            best_rank=best_rank,
        )
        if rank == 0:
            print(f"[CALIB] using best_rank as max_rank={max_rank}")

        if accordion_enabled:
            cands = accordion_candidates_from_best(best_rank, min_rank, max_rank)
            if not use_fixed_rank:
                accordion_current_rank = cands[2]
                current_matrix_rank = accordion_current_rank
                dynamic_hook.update_rank(accordion_current_rank)
            print(f"[ACCORDION] candidate ranks: {cands}")
            if not use_fixed_rank:
                print(f"[ACCORDION] start rank={accordion_current_rank}")
        else:
            print("[ACCORDION] disabled")

        # If you want to lock rank after calibration, set use_fixed_rank_after_calibration: true
        if bool(calib_cfg.get("use_fixed_rank_after_calibration", True)):
            use_fixed_rank = True
            fixed_rank_value = best_rank
            print(f"[CALIB] locking rank to {best_rank} for training")

    # fixed rank mode (if enabled)
    if enable_powerSGD and dynamic_hook is not None and use_fixed_rank:
        fixed_r = max(min_rank, min(max_rank, int(fixed_rank_value)))
        dynamic_hook.update_rank(fixed_r)
        current_matrix_rank = fixed_r
        accordion_current_rank = fixed_r
        print(f"[FIXED-RANK] using fixed rank={fixed_r}")
    if accordion_enabled and cands is None:
        cands = accordion_candidates_from_best(current_matrix_rank, min_rank, max_rank)
        if not use_fixed_rank:
            accordion_current_rank = cands[2]
            current_matrix_rank = accordion_current_rank
            if dynamic_hook is not None:
                dynamic_hook.update_rank(accordion_current_rank)
        print(f"[ACCORDION] candidate ranks: {cands}")
        if not use_fixed_rank:
            print(f"[ACCORDION] start rank={accordion_current_rank}")

    # Stage-based adaptive config.
    adaptive_config = config.get("adaptive_training", {}) if config else {}
    stages = adaptive_config.get("stages", []) if adaptive_config.get("enable", False) else []
    current_stage_index = 0

    if stages:
        print("[STAGES] enabled")
        for s in stages:
            print(
                f"  Stage {s['name']}: epoch {s['start_epoch']}–{s['end_epoch']}, "
                f"base_matrix_rank={s['base_matrix_rank']}, "
                f"sync_freq={s['base_sync_frequency']}, lr={s['learning_rate']}"
            )
    else:
        print("[STAGES] disabled")

    # ACCORDION state (epoch-end gradient norm based).
    grad_norm_history: Dict[int, float] = {}
    accordion_log_interval = 5
    rank_cooldown = 0
    lr_cooldown = 0
    grad_norm_log_interval = 10

    start_time = datetime.now()
    epoch_results: List[Dict[str, Any]] = []
    iteration_count = 0

    # ---- Epoch loop ----
    for epoch in range(epochs):

        model.train()
        if hasattr(train_dataloader, "sampler") and hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)

        if use_planned_lr_change and epoch >= warmup_epochs:
            if scheduler_name == "step" and step_size > 0:
                if (epoch + 1) % step_size == 0:
                    lr_cooldown = LR_COOLDOWN_EPOCHS
            elif scheduler_name == "multistep":
                try:
                    ms = [int(m) for m in milestones]
                except Exception:
                    ms = []
                if (epoch + 1) in ms:
                    lr_cooldown = LR_COOLDOWN_EPOCHS

        compute_accum_epoch = (epoch % 10 == 0)
        epoch_loss = 0.0
        num_data_processed = 0
        epoch_grad_norm_log_sum = 0.0
        epoch_grad_norm_log_count = 0
        grad_accumulators = None
        optimizer.zero_grad(set_to_none=True)
        if (
            compute_accum_epoch
            and accordion_enabled
            and (not rank_schedule_enabled)
            and dynamic_hook is not None
        ):
            grad_accumulators = []
            for p in model.parameters():
                if p.requires_grad:
                    grad_accumulators.append(p.detach().new_zeros(p.shape))
                else:
                    grad_accumulators.append(None)

        # ---- Batch loop ----
        for batch_idx, (data, target) in enumerate(train_dataloader):
            benchmark.start_iteration()

            data, target = data.to(device), target.to(device)
            num_data_processed += len(data)

            # stage sync freq
            current_sync_frequency = 1
            if stages and enable_adaptive_sync:
                for stage in stages:
                    if stage["start_epoch"] <= epoch <= stage["end_epoch"]:
                        current_sync_frequency = stage.get("base_sync_frequency", 1)
                        break

            accum_steps = max(1, int(current_sync_frequency))
            sync_now = ((batch_idx + 1) % accum_steps == 0)

            # Let hook know the current iteration (for start_powerSGD_iter gating)
            if enable_powerSGD and dynamic_hook is not None:
                dynamic_hook.set_iter(iteration_count)

            # no_sync for grad accumulation
            ctx = model.no_sync() if not sync_now else nullcontext()
            with ctx:
                output = model(data)
                loss = criterion(output, target) / accum_steps
                loss.backward()

            # PowerSGD active?
            is_powerSGD_active = enable_powerSGD and (iteration_count >= start_powerSGD_iter_cfg)
            current_rank_val = current_matrix_rank if is_powerSGD_active else 0

            # ACCORDION (accumulated gradient over epoch)
            if (
                compute_accum_epoch
                and accordion_enabled
                and (not rank_schedule_enabled)
                and is_powerSGD_active
                and dynamic_hook is not None
                and grad_accumulators is not None
            ):
                for idx, p in enumerate(model.parameters()):
                    if p.grad is None:
                        continue
                    acc = grad_accumulators[idx]
                    if acc is None:
                        continue
                    acc.add_(p.grad.detach())

            # Grad norm logging (sampled every N batches)
            if (batch_idx % grad_norm_log_interval) == 0:
                total_norm_sq = 0.0
                for p in model.parameters():
                    if p.grad is None:
                        continue
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm_sq += float(param_norm.item() ** 2)
                local_grad_norm = math.sqrt(total_norm_sq)
                epoch_grad_norm_log_sum += local_grad_norm
                epoch_grad_norm_log_count += 1

            # optimizer step
            communication_time = 0.0
            if sync_now:
                # NOTE: actual comm happens during backward, so this is not "true comm time"
                # We keep it for compatibility with your benchmark tool.
                benchmark.start_communication()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                fallback_comm_time = benchmark.end_communication()
                communication_time = (
                    benchmark.last_communication_time
                    if benchmark.last_communication_time > 0.0
                    else fallback_comm_time
                )
                benchmark.last_communication_time = 0.0

            epoch_loss += loss.item() * accum_steps

            decompression_time = benchmark.end_decompression()

            # Logging sizes: use total parameter bytes for original_size (more meaningful than data tensor)
            original_size = _total_param_bytes(model.module)
            compressed_size = original_size  # keep as placeholder; calibration uses real timing

            benchmark.record_result(
                epoch=epoch,
                iteration=iteration_count,
                communication_time=communication_time,
                compression_time=benchmark.last_compression_time,
                decompression_time=decompression_time,
                original_size=original_size,
                compressed_size=compressed_size,
                loss=loss.item() * accum_steps,
                powerSGD_active=is_powerSGD_active,
                current_rank=current_rank_val,
            )

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch}, Batch {batch_idx}: "
                    f"Sync every {accum_steps} steps, (bench)Comm time={communication_time:.6f}s, rank={current_rank_val}"
                )

            if enable_powerSGD and iteration_count == start_powerSGD_iter_cfg:
                print(f"=== PowerSGD ACTIVATED at iteration {iteration_count} ===")
                print(f"matrix rank: {current_matrix_rank}, error_feedback={use_error_feedback}")

            iteration_count += 1

        # epoch end
        print(f"Epoch {epoch}, Loss: {epoch_loss / len(train_dataloader)}")

        if rank == 0 and eval_dataloader:
            results = evaluate(model.module, eval_dataloader, start_time, device)
            accuracy = results["accuracy"]
            elapsed_time = results["elapsed_time"]
            print(f"Validation Accuracy: {accuracy:.2f}%, Time: {elapsed_time:.2f}s")

            try:
                benchmark.record_val_accuracy(epoch=epoch, accuracy_pct=accuracy, iteration=iteration_count - 1)
            except AttributeError:
                pass

            epoch_results.append(
                {
                    "epoch": epoch,
                    "epoch_loss": epoch_loss,
                    "accuracy": accuracy,
                    "elapsed_time": elapsed_time,
                    "num_data_processed": num_data_processed,
                    "grad_norm_avg": None,
                    "grad_accum_norm": None,
                    "grad_change_rate": None,
                }
            )

        # Grad norm logging (epoch average)
        if epoch_grad_norm_log_count > 0:
            stats = torch.tensor(
                [epoch_grad_norm_log_sum, float(epoch_grad_norm_log_count)],
                dtype=torch.float32,
                device=device,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            epoch_grad_norm_log_avg = float(stats[0].item()) / max(1.0, float(stats[1].item()))
        else:
            epoch_grad_norm_log_avg = None

        if rank == 0 and epoch_results:
            epoch_results[-1]["grad_norm_avg"] = epoch_grad_norm_log_avg

        grad_accum_norm = None
        grad_change_rate = None

        # Rank schedule (epoch-based fixed ranks)
        if rank_schedule_enabled and enable_powerSGD and dynamic_hook is not None:
            target_rank = rank_from_schedule(epoch, rank_schedule, rank_schedule_default)
            if target_rank != current_matrix_rank:
                current_matrix_rank = target_rank
                accordion_current_rank = target_rank
                try:
                    dynamic_hook.update_rank(target_rank)
                    print(f"[RANK-SCHEDULE] epoch={epoch} rank -> {target_rank}")
                except Exception as e:
                    print(f"[RANK-SCHEDULE] rank update failed: {e}")

        # ACCORDION rank decision at epoch end (accumulated grad norm)
        if (
            compute_accum_epoch
            and accordion_enabled
            and (not rank_schedule_enabled)
            and dynamic_hook is not None
            and grad_accumulators is not None
        ):
            total_norm_sq = None
            for acc in grad_accumulators:
                if acc is None:
                    continue
                val = acc.detach().pow(2).sum()
                total_norm_sq = val if total_norm_sq is None else total_norm_sq + val
            if total_norm_sq is None:
                end_norm = None
            else:
                end_norm = float(torch.sqrt(total_norm_sq).item())
            if end_norm is None:
                continue
            # average accumulated grad norm across ranks
            end_norm = _avg_across_ranks(end_norm, device)
            grad_norm_history[epoch] = end_norm
            grad_accum_norm = end_norm

            past_value = grad_norm_history.get(epoch - GRAD_COMPARE_STRIDE_EPOCHS)
            target_rank, grad_change_rate, regime = decide_accordion_target_rank(
                epoch=epoch,
                current_rank=current_matrix_rank,
                min_rank=min_rank,
                max_rank=max_rank,
                end_norm=end_norm,
                past_value=past_value,
                lr_cooldown=lr_cooldown,
                rank_cooldown=rank_cooldown,
            )

            if (epoch % accordion_log_interval) == 0 and regime in (
                "critical",
                "not_critical",
                "insufficient_history",
            ):
                if past_value is not None and past_value > 0.0 and grad_change_rate is not None:
                    print(
                        f"[ACCORDION] check epoch={epoch} "
                        f"end_norm={end_norm:.4e} "
                        f"past10_value={past_value:.4e} "
                        f"change_rate={grad_change_rate:.4e} "
                        f"regime={regime}"
                    )
                else:
                    print(
                        f"[ACCORDION] check epoch={epoch} "
                        f"end_norm={end_norm:.4e} "
                        f"past10_value=NA "
                        f"change_rate=NA "
                        f"regime={regime}"
                    )

            if target_rank != current_matrix_rank and not use_fixed_rank:
                old_rank = current_matrix_rank
                current_matrix_rank = target_rank
                accordion_current_rank = target_rank
                try:
                    dynamic_hook.update_rank(target_rank)
                    print(
                        f"[ACCORDION] epoch={epoch} "
                        f"rank {old_rank} -> {target_rank}"
                    )
                    rank_cooldown = RANK_COOLDOWN_EPOCHS
                except Exception as e:
                    print(f"[ACCORDION] rank update failed: {e}")

        if lr_cooldown > 0:
            lr_cooldown -= 1
        if rank_cooldown > 0:
            rank_cooldown -= 1

        if rank == 0 and epoch_results:
            epoch_results[-1]["grad_accum_norm"] = grad_accum_norm
            epoch_results[-1]["grad_change_rate"] = grad_change_rate

        if rank == 0 and config:
            save_dir = config.get("save_dir", "./results")
            results_csv = config.get("results_csv", "accuracy_results.csv")
            if epoch_results:
                save_results_to_csv(epoch_results, save_dir, results_csv)

        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.optimizer.param_groups[0]["lr"]
            print(f"[Scheduler] Epoch {epoch} finished. New lr = {current_lr:.6f}")

    benchmark.save_results()
    benchmark.print_summary()
    benchmark.create_visualizations()
    print("Training completed!")
