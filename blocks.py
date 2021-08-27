import torch
import torch.nn as nn


"""
Primary model blocks to create the base blocks


"""

def Conv(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)


class PCBA(nn.Module):
    def __init__(self,
                 in_channels = 150,
                 out_channels = 32,
                 kernel_size = (3, 3),
                 stride = 1,
                 padding = 1,
                 momentum = 0.9):
        super().__init__()
        self.pcba = nn.Sequential(
            nn.MaxPool2d(kernel_size = (2, 2)),
            Conv(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.BatchNorm2d(num_features = out_channels, momentum = momentum),
        )

    def forward(self, x):
        return self.pcba(x)


class CBA(nn.Module):
    def __init__(self,
                 in_channels=150,
                 out_channels=32,
                 kernel_size=(3, 3),
                 stride=1,
                 padding=1,
                 momentum=0.9):
        super().__init__()
        self.cba = nn.Sequential(
            Conv(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels, momentum=momentum),
        )

    def forward(self, x):
        return self.cba(x)


class UCBA(nn.Module):
    def __init__(self,
                 in_channels=150,
                 out_channels=32,
                 kernel_size=(3, 3),
                 stride=1,
                 padding=1,
                 momentum=0.9):
        super().__init__()
        self.ucba = nn.Sequential(
            nn.Upsample(scale_factor = 2),
            Conv(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels, momentum=momentum),
        )

    def forward(self, x):
        return self.ucba(x)


class BAD(nn.Module):
    def __init__(self,
                 in_channels,
                 momentum = 0.9,
                 dropout = 0.1):
        super().__init__()
        self.bad = nn.Sequential(
            nn.BatchNorm2d(num_features = in_channels, momentum = momentum),
            nn.ReLU(),
            nn.Dropout(p = dropout),
        )

    def forward(self, x):
        return self.bad(x)
