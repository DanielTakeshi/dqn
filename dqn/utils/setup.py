import random
import numpy as np
import torch
import os


def set_all_seeds(seed, gpu=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if gpu:
        torch.cuda.manual_seed_all(seed)


def cuda_config(gpu=False, gpu_id=0):
    if gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        torch.cuda.set_device(0)
        # torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        return True
    return False
