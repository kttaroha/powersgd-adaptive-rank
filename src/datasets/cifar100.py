import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.seed import seed_worker, make_generator

def get_cifar100_dataloader(batch_size, num_replicas, rank, root='./data/cifar100', train=True, seed=42):
    """
    CIFAR-100データセット用の分散対応データローダを取得する

    Args:
        batch_size (int): 各プロセスが処理するバッチサイズ
        num_replicas (int): 分散学習におけるプロセスの総数（全ワーカー数）
        rank (int): 現在のプロセスのランク
        root (str): CIFAR-100データセットの保存先ディレクトリ
        train (bool): トレーニング用データローダか評価用データローダかを指定

    Returns:
        torch.utils.data.DataLoader: 分散対応のCIFAR-100データローダ
    """
    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # ランダムクロップ
            transforms.RandomHorizontalFlip(),  # ランダム水平反転
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])  # CIFAR-100の平均と標準偏差
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        ])

    # データセットをロード
    dataset = datasets.CIFAR100(root=root, train=train, download=True, transform=transform)

    # 分散用サンプラー
    sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_replicas, rank=rank, shuffle=train, seed=seed, drop_last=True)
    g = make_generator(seed)

    # データローダを作成
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,  # 並列データローディングのためのワーカー数
        pin_memory=True,  # CUDAの高速化のため
        worker_init_fn=seed_worker,  # ★追加
        generator=g,                 # ★追加
        persistent_workers=True 
    )

    return dataloader
