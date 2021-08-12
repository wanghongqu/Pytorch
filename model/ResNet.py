import torch
import numpy as np
from torch import nn

from torch.nn import functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.branch2 = nn.Sequential()
        if stride != 1:
            self.branch2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        v1 = self.branch1(x)
        v2 = self.branch2(x)
        return F.relu6(v1 + v2, inplace=True)


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 64

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True)
        )

        self.block2 = self.__make_layers__(64, 3, 1)
        self.block3 = self.__make_layers__(128, 4, 2)
        self.block4 = self.__make_layers__(256, 6, 2)
        self.block5 = self.__make_layers__(512, 3, 2)
        self.block6 = nn.AdaptiveAvgPool2d(1)
        self.block7 = nn.Dropout(0.5)
        self.linear = nn.Linear(512, 100)

        # self.apply(weight_init(self.modules()))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = x.view(x.shape[0], -1)
        x = self.block7(x)
        x = self.linear(x)
        return x;

    def __make_layers__(self, output_channel, repeat, stride):
        layers = []
        for i in range(repeat):
            layers.append(BasicBlock(self.in_channels, output_channel, stride))
            stride = 1;
            self.in_channels = output_channel
        return nn.Sequential(*layers)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
