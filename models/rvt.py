"""
Residual Vision Transformer (RVT)

A custom vision transformer architecture with residual connections
across transformer layers and a depthwise residual projection head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalTokenEmbedding(nn.Module):
    """Converts image patches into token embeddings using a convolutional layer."""

    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with residual connection."""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.WQ = nn.Linear(embed_dim, embed_dim, bias=False)
        self.WK = nn.Linear(embed_dim, embed_dim, bias=False)
        self.WV = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        Q = self.WQ(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.WK(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.WV(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)
        return x + self.out_proj(attn_output)  # Residual connection


class FeedForward(nn.Module):
    """Feed-forward network with residual connection."""

    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        return x + self.fc2(F.relu(self.fc1(x)))  # Residual FFN


class CVTLayer(nn.Module):
    """Single transformer layer: attention + feed-forward."""

    def __init__(self, embed_dim, num_heads, hidden_dim):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = FeedForward(embed_dim, hidden_dim)

    def forward(self, x):
        x = self.attn(x)
        x = self.ffn(x)
        return x


class DepthwiseResidualProjection(nn.Module):
    """Depthwise convolution with residual connection for final projection."""

    def __init__(self, embed_dim):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)

    def forward(self, x):
        return x + self.depthwise_conv(x)


class RVT(nn.Module):
    """
    Residual Vision Transformer.

    Args:
        in_channels: Number of input image channels (e.g., 3 for RGB).
        embed_dim: Embedding dimension for tokens.
        patch_size: Size of each image patch.
        num_heads: Number of attention heads.
        hidden_dim: Hidden dimension in the feed-forward network.
        num_classes: Number of output classes.
    """

    def __init__(self, in_channels, embed_dim, patch_size, num_heads, hidden_dim, num_classes):
        super().__init__()
        self.embedding = ConvolutionalTokenEmbedding(in_channels, embed_dim, patch_size)

        self.cvt1 = CVTLayer(embed_dim, num_heads, hidden_dim)
        self.cvt2 = CVTLayer(embed_dim, num_heads, hidden_dim)
        self.cvt3 = CVTLayer(embed_dim, num_heads, hidden_dim)

        self.projection = DepthwiseResidualProjection(embed_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x)

        x1 = self.cvt1(x)
        x2 = self.cvt2(x1 + x)   # Residual connection
        x3 = self.cvt3(x2 + x1)  # Residual connection

        x3 = x3.mean(dim=1)  # Global feature aggregation
        x3 = self.projection(x3.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)

        return self.classifier(x3)
