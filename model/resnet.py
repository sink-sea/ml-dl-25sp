'''
ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.transformer import ViTBlock

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, attn=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        
        if attn:
            self.attn = ViTBlock(planes, num_heads=4, mlp_dim=planes*4)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        if hasattr(self, 'attn'):
            out = self.attn(out)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, attn=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        if attn:
            self.attn = ViTBlock(self.expansion*planes, num_heads=4, mlp_dim=self.expansion*planes*4)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        if hasattr(self, 'attn'):
            out = self.attn(out)
        out = F.relu(out)
        return out


# ResNet model with optional attention mechanism (Late Fusion)
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, attn=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.attn = attn

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.attn))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


# ResNet with attention mechanism (Late Fusion)
class ResNet_ViT(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet_ViT, self).__init__()
        self.resnet = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, attn=True)
        self.vit = ViTBlock(512, num_heads=4, mlp_dim=2048)

    def forward(self, x):
        out_resnet = self.resnet(x)
        out_vit = self.vit(out_resnet.unsqueeze(1))  # Add a
        # channel dimension for ViTBlock
        out_vit = out_vit.mean(dim=1)  # Global average pooling
        out = out_resnet + out_vit
        return out


def ResNet18(num_classes=10, attn=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, attn=attn)


def ResNet34(num_classes=10, attn=False):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, attn=attn)


def ResNet50(num_classes=10, attn=False):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, attn=attn)


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])