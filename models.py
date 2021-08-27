from blocks import *
import numpy as np
import random
import torch
import torch.nn as nn
from torchsummary import summary

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)


class MFCF(nn.Module):

    def __init__(self, in_channels=50,
                 out_channels = 32,
                 kernel_size = (3, 3),
                 stride=1,
                 padding = 1):

        super(MFCF, self).__init__()
        self.pcba1h = PCBA(in_channels, out_channels, kernel_size, stride, padding)
        self.pcba1l = PCBA(1, out_channels, kernel_size, stride, padding)
        self.cba1h = CBA(in_channels, out_channels, kernel_size, stride, padding)
        self.cba1l = CBA(1, out_channels, kernel_size, stride, padding)
        self.pcba2h = PCBA(out_channels*2, out_channels*2, kernel_size, stride, padding)
        self.ucba2l = UCBA(out_channels*2, out_channels*2, kernel_size, stride, padding)
        self.cba2h = CBA(out_channels*2, out_channels*2, kernel_size, stride, padding)
        self.cba2l = CBA(out_channels*2, out_channels*2, kernel_size, stride, padding)
        self.ucba3l = UCBA(out_channels*2, out_channels, kernel_size, stride, padding)
        self.cba3h = CBA(out_channels*2, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        h_data, l_data = x

        h1h = self.cba1h(h_data)
        h1l = self.pcba1h(h_data)
        l1h = self.cba1l(l_data)
        l1l = self.pcba1l(l_data)

        h1 = torch.cat((h1h, l1h), dim = 1)
        l1 = torch.cat((h1l,l1l), dim = 1)

        h2h = self.cba2h(h1)
        h2l = self.pcba2h(h1)
        l2h = self.ucba2l(l1)
        l2l = self.cba2l(l1)

        h2 = torch.cat((h2h, l2h), dim = 1)
        l2 = torch.cat((h2l, l2l), dim = 1)

        h3 = self.cba3h(h2)
        l3 = self.ucba3l(l2)

        return torch.cat((h3, l3), dim = 1)





class FG_conv(nn.Module):

    def __init__(self,
                 in_channels = 64,
                 out_channels = 16,
                 kernel_size = (3, 3),
                 stride=1,
                 padding = 1,
                 momentum=0.9,
                 dropout = 0.1):

        super().__init__()
        self.p1 = [Conv(in_channels, out_channels, kernel_size, stride, padding) for _ in range(4)]
        self.p2 = [Conv(out_channels, out_channels, kernel_size, stride, padding) for _ in range(4)]
        self.p3 = [Conv(out_channels, out_channels, kernel_size, stride, padding) for _ in range(4)]
        self.bad = BAD(in_channels = out_channels, momentum = momentum, dropout = dropout)

    def forward(self, x):

        x1 = self.p1[0](x)
        x2 = self.p1[1](x)
        x3 = self.p1[2](x)
        x4 = self.p1[3](x)


        c1 = torch.cat((x1, x2, x3, x4), dim = 1)

        x1 = self.p2[0](self.bad(x1))
        x2 = self.p2[1](self.bad(x2))
        x3 = self.p2[2](self.bad(x3))
        x4 = self.p2[3](self.bad(x4))

        c2 = torch.cat((x1, x2, x3, x4), dim = 1)

        x1 = self.p3[0](self.bad(x1))
        x2 = self.p3[1](self.bad(x2))
        x3 = self.p3[2](self.bad(x3))
        x4 = self.p3[3](self.bad(x4))

        c3 = torch.cat((x1, x2, x3, x4),dim = 1)

        return torch.cat((c1, c2, c3), dim = 1)


class SPBr(nn.Module):

    def __init__(self,
                 in_channels = 192,
                 kernel_size = (3, 3),
                 num_layers = 64,
                 padding = 1,
                 stride = 1,
                 dropout = 0.1,
                 momentum = 0.9):

        super(SPBr, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.padding = padding
        self.stride = stride
        self.dropout = dropout
        self.momentum = momentum
        self.c0 = conv(cin = self.in_channels,
                       out = self.num_layers,
                       kernel_size = self.kernel_size,
                       padding = self.padding,
                       stride = self.stride)
        self.c1 = conv(cin = self.num_layers,
                       out = self.num_layers,
                       kernel_size = self.kernel_size,
                       padding = self.padding,
                       stride = self.stride)
        self.c2 = conv(cin=self.num_layers,
                       out=self.num_layers,
                       kernel_size=self.kernel_size,
                       padding=self.padding,
                       stride=self.stride)
        self.bn_ac_dr_1 = nn.Sequential(
            nn.BatchNorm2d(num_features = self.num_layers, momentum = self.momentum),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
        )
        self.bn_ac_dr_2 = nn.Sequential(
            nn.BatchNorm2d(num_features=self.num_layers, momentum=self.momentum),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
        )
        self.bn_ac_dr_3 = nn.Sequential(
            nn.BatchNorm2d(num_features=self.num_layers, momentum=self.momentum),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
        )

    def forward(self, x):

        x1 = self.c0(x)
        x1 = self.bn_ac_dr_1(x1)

        x2 = self.c1(x1)
        x2 = self.bn_ac_dr_2(x2)

        x3 = self.c1(x2)
        x3 = self.bn_ac_dr_3(x3)

        final_op = torch.cat((x1, x2, x3), dim = 1)

        return final_op




class FGCN(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels = 50,
                 kernel_size = (3, 3),
                 gab_layers = 64,
                 padding = 1,
                 stride = 1,
                 dropout = 0.1,
                 momentum = 0.9,
                 w0 = 1.0,
                 w1 = 1.0):

        super(FGCN, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.gab_layers = gab_layers
        self.padding = padding
        self.stride = stride
        self.dropout = dropout
        self.momentum = momentum
        self.num_classes = num_classes
        self.w0 = w0
        self.w1 = w1
        self.c0 = conv(cin = 192,
                       out = 64,
                       kernel_size = (1, 1),
                       padding = self.padding,
                       stride = self.stride)
        self.c1 = conv(cin = 64,
                       out = self.num_classes,
                       kernel_size = (1, 1),
                       padding = self.padding,
                       stride = self.stride)
        self.fc = nn.Linear(in_features = 104*104*self.num_classes, out_features = self.num_classes)
        self.mfcf_Block = MFCF()
        self.fg_conv_Block = FG_conv(in_channels = 64)
        self.spbr_Block = SPBr(in_channels = 50)


    def forward(self, h_data, l_data):


        mfcf_op = self.mfcf_Block((h_data, l_data))
        fg_conv_op = self.fg_conv_Block(mfcf_op)
        spbr_op = self.spbr_Block(h_data)

        w_ad_op = self.w0 * fg_conv_op + self.w1 * spbr_op

        conv1 = self.c0(w_ad_op)
        conv2 = self.c1(conv1)
        fl_op = nn.Flatten()(conv2)
        output = self.fc(fl_op)

        return output
