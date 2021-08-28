"""

FGCN Model Implementation in Pytorch
© Sagnik Roy, 2021

"""

from blocks import *
import random
import torch
import torch.nn as nn
from torchsummary import summary


torch.manual_seed(42)
random.seed(42)



class MFCF(nn.Module):

    def __init__(self, in_channels = 50,
                 out_channels = 32,
                 kernel_size = (1, 1),
                 stride = 1,
                 padding = 0):

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

    def forward(self, h_data, l_data):

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

        h2 = h2h + l2h
        l2 = h2l + l2l

        h3 = self.cba3h(h2)
        l3 = self.ucba3l(l2)

        return h3 + l3


class FG_Conv(nn.Module):

    def __init__(self,
                 in_channels = 64,
                 out_channels = 16,
                 kernel_size = (3, 3),
                 stride = 1,
                 padding = 1,
                 momentum = 0.9,
                 dropout = 0.1):

        super().__init__()
        self.p1 = [Conv(in_channels, out_channels, kernel_size, stride, padding) for _ in range(4)]
        self.p2 = [Conv(out_channels, out_channels, kernel_size, stride, padding) for _ in range(4)]
        self.p3 = [Conv(out_channels, out_channels, kernel_size, stride, padding) for _ in range(4)]
        self.bad = BAD(out_channels, momentum, dropout)

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
                 in_channels = 64,
                 out_channels = 64,
                 kernel_size = (1, 1),
                 stride = 1,
                 padding = 0,
                 momentum = 0.9,
                 dropout = 0.1):
        super().__init__()

        self.c1 = Conv(in_channels, out_channels, kernel_size, stride, padding)
        self.c2 = Conv(out_channels, out_channels, kernel_size, stride, padding)
        self.c3 = Conv(out_channels, out_channels, kernel_size, stride, padding)
        self.bad = BAD(out_channels, momentum, dropout)

    def forward(self, x):

        x1 = self.c1(x)
        x2 = self.c2(self.bad(x1))
        x3 = self.c3(self.bad(x2))

        return torch.cat((x1, x2, x3), dim = 1)


class FGCN(nn.Module):
    def __init__(self,
                 num_classes,
                 channels = 64,
                 w0 = 1.0,
                 w1 = 1.0,
                 H = 100,
                 W = 100):
        super().__init__()
        self.w0 = w0
        self.w1 = w1
        self.mfcf_block = MFCF(channels)
        self.fg_conv = FG_Conv(in_channels = 32)
        self.spbr = SPBr(channels)
        self.c0 = Conv(192, channels, (1, 1), 1, 0)
        self.c1 = Conv(channels, num_classes, (1, 1), 1, 0)
        self.fc = nn.Linear(W*H*num_classes, num_classes)

    def forward(self, h_data, l_data):
        mfcf_op = self.mfcf_block(h_data, l_data)
        fg_conv_op = self.fg_conv(mfcf_op)
        spbr_op = self.spbr(h_data)

        wad_op = self.w0 * spbr_op + self.w1 * fg_conv_op

        conv_op = self.c1(self.c0(wad_op))
        output = self.fc(nn.Flatten()(conv_op))

        return output



if __name__ == "__main__":
    model = FGCN(num_classes = 10, H = 200, W = 150)
    summary(model, [(64,200,150), (1,200,150)], device = "cpu")
