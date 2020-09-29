import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from trick import Swish
from torch.nn.parameter import Parameter

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

# GhostModule
class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


# 基础残差
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, k_size=3):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            # nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            GhostModule(inchannel, outchannel, kernel_size=3, stride=stride),
            # nn.BatchNorm2d(outchannel),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            GhostModule(outchannel, outchannel, kernel_size=3, stride=1),
            # nn.BatchNorm2d(outchannel),
            # add eca_layer
            eca_layer(outchannel, k_size)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                # nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                GhostModule(inchannel, outchannel, kernel_size=1, stride=stride),
                # nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out

# 基础残差配置SeNet
# class ResidualBlock(nn.Module):
#     def __init__(self, inchannel, outchannel, stride=1):
#         super(ResidualBlock, self).__init__()
#         self.left = nn.Sequential(
#             # nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
#             GhostModule(inchannel, outchannel, kernel_size=3, stride=stride),
#             nn.BatchNorm2d(outchannel),
#             nn.ReLU(inplace=True),
#             # nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
#             GhostModule(outchannel, outchannel, kernel_size=3, stride=1),
#             nn.BatchNorm2d(outchannel)
#         )
#         self.shortcut = nn.Sequential()
#         if stride != 1 or inchannel != outchannel:
#             self.shortcut = nn.Sequential(
#                 # nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
#                 GhostModule(inchannel, outchannel, kernel_size=1, stride=stride),
#                 nn.BatchNorm2d(outchannel)
#             )
#         # Se layers
#         self.fc1 = nn.Conv2d(outchannel, outchannel // 16, kernel_size=1)
#         self.fc2 = nn.Conv2d(outchannel // 16, outchannel, kernel_size=1)
#
#     def forward(self, x):
#         out = self.left(x)
#
#         # se
#         w = F.avg_pool2d(out, out.size(2))
#         w = F.relu(self.fc1(w))
#         w = F.sigmoid(self.fc2(w))
#
#         # excitation
#         out = out * w
#
#         out = out + self.shortcut(x)
#         out = F.relu(out)
#
#         return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet18():
    return ResNet(ResidualBlock)
