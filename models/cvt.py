"""
Convolutional Vision Transformer (CvT)

Wrapper around the HuggingFace CvT-13 model (microsoft/cvt-13) pretrained
on ImageNet, fine-tuned for binary classification.

Note: CvT outputs use `.logits` attribute (HuggingFace convention),
which is handled by the `use_logits=True` flag in the training pipeline.
"""

from transformers import CvtForImageClassification


def CvTModel(num_classes=2):
    """
    Loads the CvT-13 model from HuggingFace with the specified number of output classes.

    Returns:
        CvtForImageClassification model instance.
    """
    model = CvtForImageClassification.from_pretrained(
        "microsoft/cvt-13",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    return model
