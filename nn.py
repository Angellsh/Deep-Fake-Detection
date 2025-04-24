import torch.nn as nn
import torchvision
import torch.utils
class nnWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten =nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(28*28, 512), 
            nn.ReLU(), 
            nn.Linear(512, 512),
            nn.ReLU(), 
            nn.Linear(512, 1)
        )
    def forward(self, x):
        x= self.flatten()
        logits = self.linear(x)
        return logits
