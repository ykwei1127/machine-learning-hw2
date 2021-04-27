import torch.nn as nn
import torch

class MyModel(nn.Module):
    """Some Information about MyModel"""
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=223, out_features=512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=512, out_features=512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=512, out_features=1024),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=1024, out_features=20)
        )

    def forward(self, x):
        x = self.model(x)
        return x