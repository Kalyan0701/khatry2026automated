"""Model registry for all architectures used in the comparative analysis."""

from models.rvt import RVT
from models.cnn import ConvNet
from models.resnet import ResNet50
from models.vit import ViTModel
from models.cvt import CvTModel


# Maps model names (used in config YAML) to constructor functions
MODEL_REGISTRY = {
    "rvt": lambda cfg: RVT(
        in_channels=cfg.get("in_channels", 3),
        embed_dim=cfg.get("embed_dim", 64),
        patch_size=cfg.get("patch_size", 4),
        num_heads=cfg.get("num_heads", 8),
        hidden_dim=cfg.get("hidden_dim", 128),
        num_classes=cfg.get("num_classes", 2),
    ),
    "cnn": lambda cfg: ConvNet(num_classes=cfg.get("num_classes", 2)),
    "resnet50": lambda cfg: ResNet50(
        image_channels=cfg.get("in_channels", 3),
        num_classes=cfg.get("num_classes", 2),
    ),
    "vit": lambda cfg: ViTModel(
        num_classes=cfg.get("num_classes", 2),
        pretrained=cfg.get("pretrained", True),
    ),
    "cvt": lambda cfg: CvTModel(num_classes=cfg.get("num_classes", 2)),
}

# Models that return HuggingFace-style outputs (use .logits)
HUGGINGFACE_MODELS = {"cvt"}


def build_model(model_name, cfg):
    """
    Instantiate a model by name.

    Args:
        model_name: Key from MODEL_REGISTRY (e.g., 'rvt', 'cnn').
        cfg: Dict of model hyperparameters from the config YAML.

    Returns:
        nn.Module instance.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](cfg)
