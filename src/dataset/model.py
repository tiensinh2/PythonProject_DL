import math
import matplotlib.pyplot as plt
from torch.onnx.symbolic_opset9 import contiguous, softmax

from src.dataset import *
from src.util.logconf import *
import torch
from torch import nn as nn
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

class LunaModel(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 8):
        super().__init__()
        self.tail_batchnorm = nn.BatchNorm3d(1)
        self.block1 = LunaBlock(in_channels, out_channels)
        self.block2 = LunaBlock(out_channels, out_channels * 2)
        self.block3 = LunaBlock(out_channels * 2, out_channels * 4)
        self.block4 = LunaBlock(out_channels * 4, out_channels * 8)
        self.head = nn.Linear(out_channels * 8, 2)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()
    def init_weights(self):
        list_layer = (nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d)
        for m in self.modules():
            if isinstance(m, list_layer):
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)
    def forward(self, x):
        bn_out = self.tail_batchnorm(x)
        x = self.block1(bn_out)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x_flat = contiguous(x).view(x.size(0), -1)
        linear = self.head(x_flat)
        return linear, self.softmax(linear)


class LunaBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channel, out_channel, kernel_size = 3, padding = 1, bias = True)
        self.relu1 = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv3d(out_channel, out_channel, kernel_size = 3, padding = 1, bias = True)
        self.relu2 = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool3d(kernel_size = 2, stride = 2)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)
        block_out = self.maxpool(block_out)
        return block_out

