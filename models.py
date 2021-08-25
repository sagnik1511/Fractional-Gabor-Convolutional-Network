from blocks import *
import numpy as np
import random
import torch
import torch.nn as nn

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)


class MFCF(nn.Module):

    def __init__(self, channels = 32,
                 kernel_size = (2, 2),
                 padding = 1,
                 stride = 1):

        super(MFCF, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

    def forward(self, h_data, l_data):


        h1h, h1l = first_block(h_data,
                               kernel_size = self.kernel_size,
                               padding = self.padding,
                               stride = self.stride,
                               pool = "max")
        l1h, l1l = first_block(l_data,
                               kernel_size = self.kernel_size,
                               padding = self.padding,
                               stride = self.stride,
                               pool = "max")
        h2 = torch.cat((h1h, l1h), dim = 1)
        l2 = torch.cat((h1l, l1l), dim = 1)

        h3h, h3l = second_block(data = h2,
                                kernel_size = self.kernel_size,
                                padding = self.padding,
                                stride = self.stride,
                                pool = "max")
        l3l, l3h = second_block(data = l2,
                                kernel_size = self.kernel_size,
                                padding = self.padding,
                                stride = self.stride,
                                pool = "max",
                                downsample = False)

        h4 = torch.cat((h3h,l3h), dim = 1)
        l4 = torch.cat((h3l, l3l), dim = 1)


        h5 = third_block(data = h4,
                         kernel_size = self.kernel_size,
                         padding = self.padding,
                         stride = self.stride,
                         upsample = False)
        l5 = third_block(data = l4,
                         kernel_size = self.kernel_size,
                         padding = self.padding,
                         stride = self.stride)

        mfcf_op= torch.cat((h5, l5), axis = 1)

        return mfcf_op

class FG_conv(nn.Module):

    def __init__(self,
                 in_channels = 64,
                 kernel_size = (3, 3),
                 num_layers = 16,
                 padding = 1,
                 stride = 1,
                 dropout = 0.1,
                 momentum = 0.9):

        super(FG_conv, self).__init__()

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
            nn.Dropout(p = self.dropout),
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


       # Level - 1

        p1 = self.c0(x)
        p2 = self.c0(x)
        p3 = self.c0(x)
        p4 = self.c0(x)
        p1 = self.bn_ac_dr_1(p1)
        p2 = self.bn_ac_dr_1(p2)
        p3 = self.bn_ac_dr_1(p3)
        p4 = self.bn_ac_dr_1(p4)

        c1 = torch.cat((p1, p2, p3, p4), dim=1)

       # Level - 2

        p1 = self.c1(p1)
        p2 = self.c1(p2)
        p3 = self.c1(p3)
        p4 = self.c1(p4)
        p1 = self.bn_ac_dr_2(p1)
        p2 = self.bn_ac_dr_2(p2)
        p3 = self.bn_ac_dr_2(p3)
        p4 = self.bn_ac_dr_2(p4)

        c2 = torch.cat((p1, p2, p3, p4), dim=1)

       # Level - 3

        p1 = self.c2(p1)
        p2 = self.c2(p2)
        p3 = self.c2(p3)
        p4 = self.c2(p4)
        p1 = self.bn_ac_dr_3(p1)
        p2 = self.bn_ac_dr_3(p2)
        p3 = self.bn_ac_dr_3(p3)
        p4 = self.bn_ac_dr_3(p4)

        c3 = torch.cat((p1, p2, p3, p4), dim = 1)

        final_op = torch.cat((c1, c2, c3), dim = 1)

        return final_op

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
                 momentum = 0.9):

        super(FGCN, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.gab_layers = gab_layers
        self.padding = padding
        self.stride = stride
        self.dropout = dropout
        self.momentum = momentum
        self.num_classes = num_classes
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


    def forward(self, h_data, l_data, w0 = 1.0, w1 = 1.0):

        mfcf_Block = MFCF()
        fg_conv_Block = FG_conv(in_channels = 64)
        spbr_Block = SPBr(in_channels = 50)

        mfcf_op = mfcf_Block(h_data = h_data, l_data = l_data)
        fg_conv_op = fg_conv_Block(mfcf_op)
        spbr_op = spbr_Block(h_data)

        w_ad_op = w0 * fg_conv_op + w1 * spbr_op

        conv1 = self.c0(w_ad_op)
        conv2 = self.c1(conv1)
        fl_op = nn.Flatten()(conv2)
        output = self.fc(fl_op)

        return output




def test():
    h_data = torch.rand(1, 50, 100,100)
    l_data = torch.rand(1, 1, 100,100)
    model = FGCN(num_classes = 5)
    model_op = model(h_data = h_data, l_data = l_data)

    assert model_op.shape == (1, 5)
test()
