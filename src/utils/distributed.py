from datetime import timedelta
import os
import torch.distributed as dist


def setup_distributed(rank, world_size, master_addr, master_port):
    """
    分散環境を初期化する。

    Args:
        rank (int): 現在のプロセスのランク（グローバルランク）。
        world_size (int): 分散環境でのプロセスの総数（ワーカー数）。
        master_addr (str): マスター（ランク0）ノードのIPアドレス。
        master_port (int): マスター（ランク0）ノードのポート番号。

    Returns:
        None

    Example:
        setup_distributed(rank=0, world_size=2, master_addr="127.0.0.1", master_port=12355)
    """
    print("setting up init process group")
    timeout = timedelta(seconds=60)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size, timeout=timeout)
    print("compeleted setting up init process group")

def cleanup_distributed():
    """
    分散環境を終了する。

    Returns:
        None

    Example:
        cleanup_distributed()
    """
    dist.destroy_process_group()
