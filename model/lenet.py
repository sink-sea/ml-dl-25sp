'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
from model.transformer import ViTBlock, ViT

# LeNet-5 with optional attention mechanism (Early Fusion)
class LeNet(nn.Module):
    def __init__(self, num_classes=10, attn=False):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)
        if attn:
            self.attn = ViTBlock(16, num_heads=4, mlp_dim=64)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        if hasattr(self, 'attn'):
            x = self.attn(x.flatten(2).transpose(1, 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ViT with optional attention mechanism (Late Fusion)
class LeNet_ViT(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet_ViT, self).__init__()
        self.lenet = LeNet(num_classes=num_classes, attn=False)
        self.vit = ViT(num_classes=num_classes)

    def forward(self, x):
        out_lenet = self.lenet(x)
        out_vit = self.vit(x)
        out = out_lenet + out_vit
        return out