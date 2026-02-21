# utils/seed.py
import os
import random
import numpy as np
import torch

def set_global_seed(seed: int, deterministic: bool = True):
    # Python / NumPy
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    # Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id: int):
    """
    For DataLoader workers:
    ensures RandomCrop / Flip / etc are reproducible
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g