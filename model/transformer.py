import torch
import torch.nn as nn
import torch.nn.functional as F

# Vision Transformer (ViT) block
class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1):
        super(ViTBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = self.norm1(x)
        mlp_out = self.mlp(x)
        x = x + mlp_out
        x = self.norm2(x)
        return x

# Vision Transformer (ViT) model
class ViT(nn.Module):
    def __init__(self, num_classes=10, img_size=32, patch_size=4, dim=128, depth=6, num_heads=4, mlp_dim=256):
        super(ViT, self).__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        self.dim = dim

        self.patch_embedding = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        self.blocks = nn.ModuleList([ViTBlock(dim, num_heads, mlp_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x).flatten(2).transpose(1, 2)  # (B, N, D)
        x += self.position_embedding
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return x
    