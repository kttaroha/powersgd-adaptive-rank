import torchvision.models as models
import torch.nn as nn

def get_resnet_model(num_classes, resnet_type="resnet18", pretrained=False, cifar_stem=False):
    """
    ResNetモデルを取得する。

    Args:
        num_classes (int): 出力クラス数。
        resnet_type (str): 使用するResNetモデルの種類（例: "resnet18", "resnet34", "resnet50" など）。
        pretrained (bool): ImageNetで事前学習済みの重みを使用する場合はTrue。
        cifar_stem (bool): CIFAR向けの小入力用stemに置き換える場合はTrue。

    Returns:
        torch.nn.Module: 指定された種類のResNetモデル。

    Raises:
        ValueError: サポートされていないResNetの種類が指定された場合。

    Example:
        # ResNet18を取得
        model = get_resnet_model(num_classes=10, resnet_type="resnet18")

        # ResNet50を事前学習済みで取得
        model = get_resnet_model(num_classes=10, resnet_type="resnet50", pretrained=True)
    """
    # サポートされるResNetの種類
    resnet_map = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "resnet152": models.resnet152
    }

    # 指定されたResNetの種類がサポートされているかチェック
    if resnet_type not in resnet_map:
        raise ValueError(f"Unsupported resnet_type '{resnet_type}'. Supported types are: {list(resnet_map.keys())}")

    # 指定された種類のResNetモデルを取得
    model = resnet_map[resnet_type](pretrained=pretrained)

    if cifar_stem:
        # CIFAR(32x32)向け: 3x3 conv, stride=1, no maxpool
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    
    # 出力層を調整
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
