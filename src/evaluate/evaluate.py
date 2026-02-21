import torch
from datetime import datetime

def evaluate(model, dataloader, start_time, device):
    """
    モデルを評価し、精度と評価にかかった時間を記録する

    Args:
        model (torch.nn.Module): 評価対象のモデル
        dataloader (torch.utils.data.DataLoader): 評価データローダ
        start_time (datetime.datetime): 訓練開始時間
        device (torch.device): デバイス（例: "cuda" または "cpu"）

    Returns:
        dict: 評価結果を含む辞書
            - "accuracy": モデルの精度（正解率）
            - "elapsed_time": 評価処理にかかった時間（秒単位）
    """
    model.eval()
    correct = 0
    total = 0

    elapsed_time = (datetime.now() - start_time).total_seconds()  # 経過時間

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(correct, total)

    accuracy = 100 * correct / total if total > 0 else 0.0

    return {"accuracy": accuracy, "elapsed_time": elapsed_time}
