import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP

def get_mnist_dataloader(batch_size, num_replicas, rank, root='./data', train=True):
    """
    MNISTデータセット用の分散対応データローダを取得する。

    Args:
        batch_size (int): 各プロセスが処理するバッチサイズ。
        num_replicas (int): 分散学習におけるプロセスの総数（全ワーカー数）。
        rank (int): 現在のプロセスのランク。
        root (str): データセットの保存先ディレクトリ。
        train (bool): トレーニング用データローダか評価用データローダかを指定。

    Returns:
        torch.utils.data.DataLoader: 分散対応のMNISTデータローダ。
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)

    # 訓練用と評価用にデータセットを分割
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    if train:
        target_dataset = train_dataset
    else:
        target_dataset = val_dataset

    # 分散用サンプラー
    sampler = torch.utils.data.DistributedSampler(target_dataset, num_replicas=num_replicas, rank=rank)

    dataloader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    return dataloader

