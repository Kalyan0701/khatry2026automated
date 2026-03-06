"""
Vision Transformer (ViT)

Wrapper around the timm ViT-Base model (vit_base_patch16_224) pretrained
on ImageNet, fine-tuned for binary classification.
"""

import torch.nn as nn
import timm


class ViTModel(nn.Module):
    """
    ViT-Base/16 via timm with pretrained ImageNet weights.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to load pretrained weights.
    """

    def __init__(self, num_classes=2, pretrained=True):
        super(ViTModel, self).__init__()
        self.model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)
