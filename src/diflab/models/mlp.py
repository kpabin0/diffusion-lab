import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_features=2, out_features=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 20),
            nn.ReLU(),
            nn.Linear(20, out_features),
        )

    def forward(self, X):
        return self.net(X)


