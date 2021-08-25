import torch
import torch.nn as nn

def conv(cin, out, kernel_size = (3, 3), padding = (1, 1), stride = 1):
    return nn.Conv2d(in_channels = cin, out_channels = out, kernel_size = kernel_size, padding = padding, stride = stride)


def first_block(data, kernel_size = (3,3), padding = (1, 1), stride = 1, pool = "avg"):

    mode1 = nn.Sequential(
            conv(cin = data.shape[1], out = 32, padding = padding, stride = stride),
            nn.BatchNorm2d(num_features = 32, momentum = 0.9),
            nn.ReLU(),
        )
    if pool == 'avg':
        mode2 = nn.Sequential(
            nn.AvgPool2d(kernel_size = kernel_size),
            conv(cin = data.shape[1], out = 32, padding = padding, stride = stride),
            nn.BatchNorm2d(num_features = 32, momentum=0.9),
            nn.ReLU(),
        )
    else:
        mode2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=kernel_size),
            conv(cin = data.shape[1], out = 32, padding = padding, stride = stride),
            nn.BatchNorm2d(num_features = 32, momentum=0.9),
            nn.ReLU(),
        )
    return (mode1(data), mode2(data))        # normal , lowered


def second_block(data, kernel_size = (3, 3), padding = (1, 1), stride = 1, pool = "avg", downsample = True):

    if downsample == True:
        return first_block(data = data, kernel_size = kernel_size, padding = padding, stride = stride, pool = pool)

    else:
        mode1 = nn.Sequential(
            conv(cin=data.shape[1], out=32, padding=padding, stride=stride),
            nn.BatchNorm2d(num_features=32, momentum=0.9),
            nn.ReLU(),
        )
        mode2 = nn.Sequential(
            nn.Upsample(scale_factor = kernel_size[0]),
            conv(cin=data.shape[1], out=32, padding=padding, stride=stride),
            nn.BatchNorm2d(num_features=32, momentum=0.9),
            nn.ReLU(),
        )

    return (mode1(data), mode2(data))   # normal , lowered / upsampled

def third_block(data, kernel_size = (3, 3), padding = (1, 1), stride = 1, upsample = True):

    if upsample == True:
        return nn.Sequential(
            nn.Upsample(scale_factor = kernel_size[0]),
            conv(cin=data.shape[1], out=32, padding=padding, stride=stride),
            nn.BatchNorm2d(num_features=32, momentum=0.9),
            nn.ReLU(),
        )(data)  # lowered
    else:
        return nn.Sequential(
            conv(cin=data.shape[1], out=32, padding=padding, stride=stride),
            nn.BatchNorm2d(num_features=32, momentum=0.9),
            nn.ReLU(),
        )(data)   # upsampled




