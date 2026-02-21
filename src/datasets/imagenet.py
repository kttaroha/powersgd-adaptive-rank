import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_imagenet_dataloader(batch_size, num_replicas, rank, root='./data/imagenet/ILSVRC/Data/CLS-LOC', train=True):
    """
    ImageNetデータセット用の分散対応データローダを取得する

    Args:
        batch_size (int): 各プロセスが処理するバッチサイズ
        num_replicas (int): 分散学習におけるプロセスの総数（全ワーカー数）
        rank (int): 現在のプロセスのランク
        root (str): ImageNetデータセットの保存先ディレクトリ
        train (bool): トレーニング用データローダか評価用データローダかを指定

    Returns:
        torch.utils.data.DataLoader: 分散対応のImageNetデータローダ
    """
    # データ変換 (トレーニング用と評価用で異なる設定)

    import os


    import torchvision.transforms as transforms
    import torchvision.datasets as datasets

    # トレーニング用のデータセットをテスト読み込み

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        dataset = datasets.ImageFolder(root=f"{root}/train", transform=transform)
        print(f"Loaded dataset with {len(dataset)} images.")
    except Exception as e:
        print(f"Error loading dataset: {e}")


    if train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),  # ランダムクロップ
            transforms.RandomHorizontalFlip(),  # ランダム水平反転
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正規化
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),  # 画像の短い辺を256ピクセルにリサイズ
            transforms.CenterCrop(224),  # 中心クロップ
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # データセットをロード
    dataset = datasets.ImageFolder(root=f"{root}/train" if train else f"{root}/val", transform=transform)

    # 分散用サンプラー
    sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_replicas, rank=rank, shuffle=train)

    # データローダを作成
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,  # 並列データローディングのためのワーカー数
        pin_memory=True  # CUDAの高速化のため
    )

    return dataloader

