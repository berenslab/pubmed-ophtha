"""Train a annotation type predictor model."""

import json
import logging
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm.auto import tqdm

from pubmed_ophtha.figure_splitting.labeling.label_studio_annotations import (
    AnnotationTypeEnum,
    parse_label_studio_annotations,
)

logger = logging.getLogger(__name__)


class PDFPageCrop(torch.nn.Module):
    """Transform to crop an image to a bbox with optional random perturbations."""

    def __init__(
        self,
        random_perturbations: bool = False,
        perturbation_amount_decrease: float = 0.2,
        perturbation_amount_increase: float = 0.3,
    ):
        """
        Initialize the PDFPageCrop transform.

        Args:
            random_perturbations (bool, optional): If True, randomly perturbs the crop.
                Defaults to False.
            perturbation_amount_decrease (float, optional): Maximum perturbation amount
                when decreasing the bbox size. This value is multiplied with the
                width/height of the bbox. Defaults to 0.2.
            perturbation_amount_increase (float, optional): Maximum perturbation amount
                when increasing the bbox size. This value is multiplied with the
                width/height of the bbox. Defaults to 0.3.

        """
        super().__init__()

        self.random_perturbations = random_perturbations
        self.perturbation_amount_decrease = perturbation_amount_decrease
        self.perturbation_amount_increase = perturbation_amount_increase

    def forward(
        self, img: Image.Image, center_box: tuple[int, int, int, int]
    ) -> Image.Image:
        """
        Crop an image to the given bounding box with optional random perturbations.

        Args:
            img (Image.Image): Image to crop.
            center_box (tuple[int, int, int, int]): Bounding box to crop to \
                (x0, y0, x1, y1) in absolute coordinates.

        Returns:
            Image.Image: Image crop.

        """
        if self.random_perturbations:
            bbox_width = center_box[2] - center_box[0]
            bbox_height = center_box[3] - center_box[1]

            image_width, image_height = img.size

            x0_perturbation_amount = torch.rand(1).item() * (
                self.perturbation_amount_decrease
                if torch.rand(1).item() < 0.5
                else -1.0 * self.perturbation_amount_increase
            )
            y0_perturbation_amount = torch.rand(1).item() * (
                self.perturbation_amount_decrease
                if torch.rand(1).item() < 0.5
                else -1.0 * self.perturbation_amount_increase
            )
            x1_perturbation_amount = torch.rand(1).item() * (
                -1.0 * self.perturbation_amount_decrease
                if torch.rand(1).item() < 0.5
                else self.perturbation_amount_increase
            )
            y1_perturbation_amount = torch.rand(1).item() * (
                -1.0 * self.perturbation_amount_decrease
                if torch.rand(1).item() < 0.5
                else self.perturbation_amount_increase
            )

            center_box = (
                max(0, center_box[0] - int(x0_perturbation_amount * bbox_width)),
                max(0, center_box[1] - int(y0_perturbation_amount * bbox_height)),
                min(
                    image_width,
                    center_box[2] + int(x1_perturbation_amount * bbox_width),
                ),
                min(
                    image_height,
                    center_box[3] + int(y1_perturbation_amount * bbox_height),
                ),
            )

        return img.crop(center_box)


class ImageDataset(Dataset):
    """Dataset for loading images and their labels from a dataframe."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        split: str,
        transform: v2.Compose | v2.Transform | None = None,
        random_crop: bool = False,
    ):
        """
        Create the ImageDataset.

        Args:
            dataframe (pd.DataFrame): Dataframe containing the image paths and labels.
            split (str): Split of the dataset to use ('train' or 'test').
            transform (v2.Compose | v2.Transform | None, optional): Transformations to
                apply to the images. Defaults to None.
            random_crop (bool, optional): If True, applies random crops.
                Defaults to False.

        """
        assert split in dataframe["split"].unique().tolist()
        self.dataframe = dataframe[dataframe["split"] == split].reset_index(drop=True)
        self.transform = transform
        self.random_crop = random_crop

        self.pdf_page_crop = PDFPageCrop(random_perturbations=random_crop)

    def __len__(self) -> int:
        """
        Get the kength of the dataset.

        Returns:
            int: Length of the dataset.

        """
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the item at the given index.

        Args:
            idx (int): Index of the item to get.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing the image tensor and
                the label tensor.

        """
        img_path = self.dataframe.loc[idx, "file_name"]
        label = self.dataframe.loc[idx, "label"]
        bounding_box_x0 = self.dataframe.loc[idx, "bbox_x0"]
        bounding_box_y0 = self.dataframe.loc[idx, "bbox_y0"]
        bounding_box_x1 = self.dataframe.loc[idx, "bbox_x1"]
        bounding_box_y1 = self.dataframe.loc[idx, "bbox_y1"]

        if (
            img_path is None
            or not isinstance(img_path, str)
            or not os.path.exists(img_path)
        ):
            raise ValueError(f"Invalid image path at index {idx}: {img_path}")

        # Load full image
        image = Image.open(img_path).convert("RGB")

        # Crop to bounding box with optional random perturbations
        image = self.pdf_page_crop(
            image,
            (
                bounding_box_x0,
                bounding_box_y0,
                bounding_box_x1,
                bounding_box_y1,
            ),
        )

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_dataset_as_df(
    annotation_file_path: str,
    image_base_path: str,
    dataset_csv_file_path: str,
    dataset_seed: int | None = None,
    train_split_ratio: float = 0.8,
) -> pd.DataFrame:
    """
    Load the dataset as a dataframe. If the CSV file exists, load it from there.

    Args:
        annotation_file_path (str): Raw label studio annotation file.
        image_base_path (str): Folder path containing the images.
        dataset_csv_file_path (str): Location to save/load the dataset CSV file.
        dataset_seed (int | None, optional): Seed for the train test split. Defaults to
            None.
        train_split_ratio (float, optional): Percentage of images in the train split.
            Defaults to 0.8.

    Returns:
        pd.DataFrame: Dataframe containing the dataset.

    """
    if os.path.exists(dataset_csv_file_path):
        return pd.read_csv(dataset_csv_file_path)

    annotations = parse_label_studio_annotations(annotation_file_path)

    data = []
    for ann in annotations:
        if not ann.has_annotations:
            continue
        if ann.was_cancelled:
            continue

        image_path = os.path.join(image_base_path, ann.base_image_path)

        selected_annotation = ann.finished_annotations[0]

        if selected_annotation.bounding_boxes is None:
            continue

        for box in selected_annotation.bounding_boxes:
            annotation_labels = [
                label for label in box.labels if isinstance(label, AnnotationTypeEnum)
            ]
            if len(annotation_labels) == 0:
                continue

            if len(annotation_labels) > 1:
                logger.warning(
                    f"Multiple annotation types for box in {image_path}. "
                    "Using the first one."
                )

            class_id = 0 if annotation_labels[0] == AnnotationTypeEnum.PLAIN else 1

            data.append(
                {
                    "file_name": image_path,
                    "label": class_id,
                    "bbox_x0": int(box.x0),
                    "bbox_y0": int(box.y0),
                    "bbox_x1": int(box.x1),
                    "bbox_y1": int(box.y1),
                }
            )
    df = pd.DataFrame(data)

    if dataset_seed is not None:
        random.seed(dataset_seed)

    total_file_names = df["file_name"].unique().tolist()

    random.shuffle(total_file_names)
    train_size = int(len(total_file_names) * train_split_ratio)
    train_file_names = total_file_names[:train_size]

    df["split"] = df["file_name"].apply(
        lambda x: "train" if x in train_file_names else "test"
    )

    df.to_csv(dataset_csv_file_path, index=False)

    return df


def create_dataloader(
    annotation_file_path: str,
    image_data_csv: str,
    batch_size: int,
    image_base_path: str,
    dataset_seed: int | None = None,
    train_split_ratio: float = 0.8,
    shuffle: bool = True,
    num_workers: int = 4,
    use_page_crop: bool = True,
    use_oversampling: bool = False,
) -> tuple[DataLoader, DataLoader, list[float]]:
    """
    Create the train and test dataloaders.

    Args:
        annotation_file_path (str): Raw label studio annotation file.
        image_data_csv (str): Dataset CSV file path.
        batch_size (int): Batch size.
        image_base_path (str): Folder path containing the images.
        dataset_seed (int | None, optional): Seed for train/test split creation.
            Defaults to None.
        train_split_ratio (float, optional): Train split ratio. Defaults to 0.8.
        shuffle (bool, optional): If True, shuffle the training data. Defaults to True.
        num_workers (int, optional): Number of workers to load the dataset. Defaults to
            4.
        use_page_crop (bool, optional): If True, use PDFPageCrop. Defaults to True.
        use_oversampling (bool, optional): If True, oversample the training dataset.
            Defaults to False.

    Returns:
        tuple[DataLoader, DataLoader, list[float]]: Tuple containing the train
            dataloader, test dataloader and class weights for the training set.

    """
    # CSV has the columns: image_path, split and label
    # split is either 'train' or 'test'
    # label is an integer representing the class label

    # First define transforms
    train_transforms = v2.Compose(
        [
            v2.Resize((224, 224)),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomOrder(
                [
                    v2.RandomAdjustSharpness(sharpness_factor=2.0, p=0.2),
                    v2.RandomAutocontrast(p=0.2),
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
            ),
            v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # pyright: ignore[reportArgumentType]
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],  # Imagenet stats
            ),
        ]
    )

    test_transforms = v2.Compose(
        [
            v2.Resize((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Read the CSV file
    df = get_dataset_as_df(
        annotation_file_path,
        image_base_path,
        image_data_csv,
        dataset_seed=dataset_seed,
        train_split_ratio=train_split_ratio,
    )

    train_set_size_per_class = []

    for class_id in sorted(df[df["split"] == "train"]["label"].unique().tolist()):  # pyright: ignore[reportArgumentType]
        num_samples_in_class = len(
            df[(df["split"] == "train") & (df["label"] == class_id)]
        )

        train_set_size_per_class.append(num_samples_in_class)

    train_set_weights = [
        1.0 / size if size > 0 else 1.0 for size in train_set_size_per_class
    ]

    if use_oversampling:
        max_size = max(train_set_size_per_class)
        sampling_factor_per_class = [
            int(math.ceil(max_size / size)) if size > 0 else 1
            for size in train_set_size_per_class
        ]

        oversampled_data = []
        for class_id, factor in enumerate(sampling_factor_per_class):
            if factor <= 1:
                continue
            class_data = df[(df["split"] == "train") & (df["label"] == class_id)]
            for _ in range(factor - 1):
                oversampled_data.append(class_data.copy())

        if len(oversampled_data) > 0:
            oversampled_df = pd.concat(oversampled_data, ignore_index=True)
            df = pd.concat([df, oversampled_df], ignore_index=True)

    # Create train and test datasets
    train_dataset = ImageDataset(
        df, "train", transform=train_transforms, random_crop=use_page_crop
    )
    test_dataset = ImageDataset(df, "test", transform=test_transforms)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader, train_set_weights


def checkpoint(
    model: torch.nn.Module,
    output_path: str,
    epoch: int,
    train_loss: float | None = None,
    test_loss: float | None = None,
    confusion_matrix: np.ndarray | None = None,
    lr: float | None = None,
):
    """
    Save model checkpoint and metrics.

    Args:
        model (torch.nn.Module): Model to save.
        output_path (str): Path to save the checkpoint and metrics to.
        epoch (int): Current epoch.
        train_loss (float | None, optional): Current train loss. Defaults to None.
        test_loss (float | None, optional): Current test loss. Defaults to None.
        confusion_matrix (np.ndarray | None, optional): Numpy array containing the
            confusion matrix. Defaults to None.
        lr (float | None, optional): Current learning rate. Defaults to None.

    """
    os.makedirs(output_path, exist_ok=True)
    checkpoint_path = os.path.join(output_path, f"model_epoch_{epoch + 1}.pth")

    torch.save(model.state_dict(), checkpoint_path)

    metrics = []
    if os.path.exists(os.path.join(output_path, "metrics.json")):
        with open(os.path.join(output_path, "metrics.json")) as f:
            metrics = json.load(f)

    metric_dict = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "checkpoint_path": checkpoint_path,
    }

    if lr is not None:
        metric_dict["learning_rate"] = lr

    if confusion_matrix is not None:
        assert confusion_matrix.shape[0] == confusion_matrix.shape[1]
        metric_dict["accuracy"] = confusion_matrix.diagonal().sum() / (
            confusion_matrix.sum() + 1e-8
        )

        # Per class precision and recall
        for i in range(confusion_matrix.shape[0]):
            precision = confusion_matrix[i, i] / (confusion_matrix[:, i].sum() + 1e-8)

            recall = confusion_matrix[i, i] / (confusion_matrix[i, :].sum() + 1e-8)

            metric_dict[f"class_{i}_precision"] = precision
            metric_dict[f"class_{i}_recall"] = recall
            metric_dict[f"class_{i}_f1_score"] = (
                2 * precision * recall / (precision + recall + 1e-8)
            )
    metrics.append(metric_dict)

    with open(os.path.join(output_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)


def load_model(num_classes: int) -> torch.nn.Module:
    """
    Load a resnet 50 model with number of classes.

    Args:
        num_classes (int): Number of classes for classification.

    Returns:
        torch.nn.Module: Loaded model.

    """
    # Load resnet50 model with imagenet weights
    model = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.DEFAULT
    )
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    return model


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    output_path: str,
    loss_weight: list[float] | None = None,
):
    """
    Train the model.

    Args:
        model (torch.nn.Module): Model to train.
        train_loader (DataLoader): Train dataloader.
        test_loader (DataLoader): Test dataloader.
        num_epochs (int): Number of epochs to train.
        learning_rate (float): Starting learning rate.
        output_path (str): Path to save the model and metrics.
        loss_weight (list[float] | None, optional): Weights for weighted cross entropy
            loss. Defaults to None.

    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1, weight=loss_weight)  # pyright: ignore[reportArgumentType]

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch in range(num_epochs):
        model.train()

        loss_sum = 0.0
        num_batches = 0
        for images, labels in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"
        ):
            batch_images = images.cuda() if torch.cuda.is_available() else images
            batch_labels = labels.cuda() if torch.cuda.is_available() else labels
            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            num_batches += 1

        lr_scheduler.step()
        current_lr = lr_scheduler.get_last_lr()[0]
        train_loss = loss_sum / num_batches

        model.eval()
        total = 0
        correct = 0
        loss_sum = 0.0
        num_batches = 0

        # Create a confusion matrix
        gt_classes = {}
        with torch.no_grad():
            for images, labels in tqdm(
                test_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Testing"
            ):
                batch_images = images.cuda() if torch.cuda.is_available() else images
                batch_labels = labels.cuda() if torch.cuda.is_available() else labels
                outputs = model(batch_images)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

                loss = criterion(outputs, batch_labels)
                loss_sum += loss.item()
                num_batches += 1

                for i in range(len(batch_labels)):
                    gt = batch_labels[i].item()
                    pred = predicted[i].item()
                    if gt not in gt_classes:
                        gt_classes[gt] = {}
                    if pred not in gt_classes[gt]:
                        gt_classes[gt][pred] = 0
                    gt_classes[gt][pred] += 1

        test_loss = loss_sum / num_batches

        # Create a confusion matrix dataframe
        confusion_matrix_index = [f"class_{i}" for i in range(model.fc.out_features)]  # pyright: ignore[reportArgumentType, reportAttributeAccessIssue]

        confusion_matrix = pd.DataFrame(
            0, index=confusion_matrix_index, columns=confusion_matrix_index
        )
        for gt_class, pred_class_list in gt_classes.items():
            for pred_class in pred_class_list:
                confusion_matrix.loc[f"class_{gt_class}", f"class_{pred_class}"] = (
                    gt_classes[gt_class][pred_class]
                )

        checkpoint(
            model,
            output_path,
            epoch,
            train_loss=train_loss,
            test_loss=test_loss,
            confusion_matrix=confusion_matrix.to_numpy(),
            lr=current_lr,
        )

        # Use seaborn to create a heatmap and save it
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(f"Confusion Matrix - Epoch {epoch + 1}")
        os.makedirs(os.path.join(output_path, "plots"), exist_ok=True)
        plt.savefig(
            os.path.join(
                output_path, "plots", f"confusion_matrix_epoch_{epoch + 1}.png"
            )
        )
        plt.close()

        logger.info(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
            f"Test Loss: {test_loss:.4f}"
        )
