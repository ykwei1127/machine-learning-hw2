import torch
from matplotlib import pyplot as plt
import pandas as pd

# WRMSE
def WRMSE(preds, target, device):
    # torch.cuda.set_device(GPU)
    weight = torch.tensor([
        0.05223, 0.0506, 0.05231, 0.05063, 0.05073,
        0.05227, 0.05177, 0.05186, 0.05076, 0.05063,
        0.0173, 0.05233, 0.05227, 0.05257, 0.05259,
        0.05222, 0.05204, 0.05185, 0.05229, 0.05074
    ]).to(device)
    wrmse = torch.pow(preds-target, 2)
    wrmse = torch.sum(wrmse * weight)
    return wrmse.item()
    