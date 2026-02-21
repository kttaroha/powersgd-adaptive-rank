"""
設定ファイルの動的調整ユーティリティ
ネットワーク制限のtarget_ipを実行環境に応じて自動調整
"""

import os
import yaml
from typing import Dict, Any, Optional

def adapt_network_config(config: Dict[str, Any], master_addr: str, world_size: int) -> Dict[str, Any]:
    """
    ネットワーク設定を実行環境に応じて調整
    
    Args:
        config: 設定辞書
        master_addr: マスターノードアドレス
        world_size: ワールドサイズ
    
    Returns:
        調整された設定辞書
    """
    if not config.get("network_limits", {}).get("enable", False):
        return config
    
    network_config = config["network_limits"]
    
    # 単一ノード実行の場合
    if world_size == 1:
        network_config["target_ip"] = "127.0.0.1"
        network_config["interface"] = "eth0"
        print("単一ノード実行: ネットワーク制限をlocalhostに設定")
    
    # 分散学習の場合
    else:
        network_config["target_ip"] = master_addr
        # ネットワークインターフェースを自動検出
        interface = detect_network_interface()
        if interface:
            network_config["interface"] = interface
        print(f"分散学習: ネットワーク制限を{master_addr}に設定")
    
    return config

def detect_network_interface() -> Optional[str]:
    """
    利用可能なネットワークインターフェースを検出
    
    Returns:
        推奨インターフェース名、検出できない場合はNone
    """
    import subprocess
    
    try:
        # ipコマンドでインターフェース一覧を取得
        result = subprocess.run(
            ["ip", "link", "show"],
            capture_output=True, text=True, check=True
        )
        
        lines = result.stdout.split('\n')
        interfaces = []
        
        for line in lines:
            if ': ' in line and 'state UP' in line:
                # インターフェース名を抽出
                interface = line.split(': ')[1].split('@')[0]
                if interface not in ['lo', 'docker0']:  # ループバックとDockerブリッジを除外
                    interfaces.append(interface)
        
        # 優先順位で選択
        priority_interfaces = ['enp0s31f6', 'eth0', 'ens33', 'enp0s3']
        
        for preferred in priority_interfaces:
            if preferred in interfaces:
                return preferred
        
        # 優先順位にない場合は最初のものを返す
        if interfaces:
            return interfaces[0]
        
        return None
        
    except Exception as e:
        print(f"ネットワークインターフェースの検出に失敗: {e}")
        return None

def load_and_adapt_config(config_path: str, master_addr: str, world_size: int) -> Dict[str, Any]:
    """
    設定ファイルを読み込んで環境に応じて調整
    
    Args:
        config_path: 設定ファイルのパス
        master_addr: マスターノードアドレス
        world_size: ワールドサイズ
    
    Returns:
        調整された設定辞書
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return adapt_network_config(config, master_addr, world_size)
