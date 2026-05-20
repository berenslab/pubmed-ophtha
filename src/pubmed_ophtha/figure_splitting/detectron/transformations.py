"""Module for Detectron2 data transformations."""

from detectron2.config import CfgNode
from detectron2.data import DatasetMapper
from detectron2.data.transforms import PILColorTransform
from torchvision.transforms import v2


def build_train_transforms() -> v2.Compose:
    """
    Build the training transforms for RetinaNet.

    Returns:
        v2.Compose: Training transforms.

    """
    # No normalization here, as RetinaNet learns it
    train_transforms = v2.Compose(
        [
            v2.RandomOrder(
                [
                    v2.RandomAdjustSharpness(sharpness_factor=2.0, p=0.2),
                    v2.RandomAutocontrast(p=0.2),
                    v2.RandomApply(
                        [v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0))],
                        p=0.2,
                    ),
                    v2.RandomApply([v2.JPEG(quality=(5, 50))], p=0.2),
                    v2.RandomPosterize(bits=4, p=0.2),
                    v2.RandomChoice(
                        [
                            v2.RandomApply(
                                [
                                    v2.ColorJitter(
                                        brightness=0.2,
                                        contrast=0.2,
                                        saturation=0.2,
                                        hue=0.1,
                                    )
                                ],
                                p=0.2,
                            ),
                            v2.RandomGrayscale(p=0.2),
                        ]
                    ),
                ]
            )
        ]
    )

    return train_transforms


class TransformDatasetMapper(DatasetMapper):
    """Dataset Mapper that applies color transformations."""

    @classmethod
    def from_config(cls, cfg: CfgNode, is_train: bool = True) -> dict:
        """
        Load the dataset mapper from the config.

        Args:
            cfg (CfgNode): Model configuration.
            is_train (bool, optional): If True, load a training dataset mapper.
                Defaults to True.

        Returns:
            dict: Dictionary for instantiating the dataset mapper.

        """
        ret = super().from_config(cfg, is_train)

        # Color augmentations and normalization
        if is_train:
            ret["augmentations"].append(PILColorTransform(build_train_transforms()))

        return ret
