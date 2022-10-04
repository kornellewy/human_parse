from pathlib import Path
from typing import List

import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import albumentations as A

from utils.utils import load_images


class HumanParsingDatasetSyntetic(torch.utils.data.Dataset):
    IMAGES_DIR_NAME = "image_without_background"
    LABELS_DIR_NAME = "image_parse_new"
    AGUMENTATIONS = A.Compose(
        [
            A.ChannelShuffle(p=0.2),
            A.RGBShift(),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0
            ),
            A.PadIfNeeded(
                min_height=224, min_width=224, always_apply=True, border_mode=0
            ),
            A.RandomCrop(height=224, width=224, always_apply=True),
            A.IAAAdditiveGaussianNoise(p=0.2),
            A.IAAPerspective(p=0.5),
            A.OneOf(
                [
                    A.CLAHE(p=1),
                    A.RandomBrightness(p=1),
                    A.RandomGamma(p=1),
                ],
                p=0.9,
            ),
            A.OneOf(
                [
                    A.IAASharpen(p=1),
                    A.Blur(blur_limit=3, p=1),
                    A.MotionBlur(blur_limit=3, p=1),
                ],
                p=0.9,
            ),
            A.OneOf(
                [
                    A.RandomContrast(p=1),
                    A.HueSaturationValue(p=1),
                ],
                p=0.9,
            ),
        ],
        additional_targets={"label": "mask"},
    )
    TRANSFORMS = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    CLASS_MAP = {
        0: "background",
        1: "body",
        2: "upper clothes",
        3: "bottom clothes",
    }
    SYNTETIC_DATASET_MODE = "syntetic"
    REAL_DATASET_MODE = "real"

    def __init__(
        self,
        dataset_paths: List[Path],
        output_size: tuple = (224, 224),
        train_mode: bool = True,
        classification: bool = True,
    ) -> None:
        self.dataset_paths = dataset_paths
        self.output_size = output_size
        self.train_mode = train_mode
        self.classification = classification

        self.images_paths = []
        self.labels_paths = []
        for dataset_path in self.dataset_paths:
            self.images_dir_path = (
                Path(dataset_path).joinpath(self.IMAGES_DIR_NAME).as_posix()
            )
            self.images_paths += load_images(self.images_dir_path)
            self.labels_dir_path = (
                Path(dataset_path).joinpath(self.LABELS_DIR_NAME).as_posix()
            )
            self.labels_paths += load_images(self.labels_dir_path)

    def __len__(self) -> int:
        return len(self.images_paths)

    def __getitem__(self, index: int) -> dict:
        image_path = self.images_paths[index]
        image_name = Path(image_path).stem
        label_path = self.labels_paths[index]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = cv2.imread(label_path, 0)

        dataset_type = (
            self.SYNTETIC_DATASET_MODE
            if "syn" in Path(image_path).parent.parent.stem
            else self.REAL_DATASET_MODE
        )

        if self.train_mode:
            agumented_data = self.AGUMENTATIONS(image=image)
            image = agumented_data["image"].astype(np.uint8)

        label = self.translate_parse(label=label, dataset_type=dataset_type)

        image = cv2.resize(image, self.output_size, cv2.INTER_AREA).astype(np.uint8)
        label = cv2.resize(label, self.output_size, cv2.INTER_AREA).astype(np.uint8)

        if self.classification:
            label_classification, _ = np.histogram(
                label,
                bins=len(self.CLASS_MAP),
                range=(-0.5, len(self.CLASS_MAP) - 0.5),
            )
            label_classification = np.asarray(
                np.asarray(label_classification, dtype=np.bool), dtype=np.uint8
            )
        else:
            label_classification = 0

        label = torch.from_numpy(label).long()
        image = self.TRANSFORMS(image)
        return {
            "image_name": image_name,
            "image": image,
            "label": label,
            "label_classification": label_classification,
        }

    def translate_parse(self, label: np.ndarray, dataset_type: str) -> np.ndarray:
        if dataset_type == self.SYNTETIC_DATASET_MODE:
            label = np.where(label == 3, 1, label)
            label = np.where(label == 7, 2, label)
            label = np.where(label == 11, 3, label)
        if dataset_type == self.REAL_DATASET_MODE:
            label = np.where(label == 1, 1, label)
            label = np.where(label == 2, 1, label)
            label = np.where(label == 3, 2, label)
            label = np.where(label == 4, 3, label)
            label = np.where(label == 5, 1, label)
            label = np.where(label == 6, 1, label)
            label = np.where(label == 7, 1, label)
            label = np.where(label == 8, 1, label)
        if any(np.unique(label) > 3):
            raise ValueError(
                f"parser value outside a map, np.unique(label): {np.unique(label)}"
            )
        return label.astype(np.uint8)


if __name__ == "__main__":
    dataset_paths = [
        Path("J:/deepcloth/datasets/human_body_parsing/kjn_parse_dataset_001"),
        Path("J:/deepcloth/datasets/syn_viton_001"),
    ]
    dataset = HumanParsingDatasetSyntetic(dataset_paths=dataset_paths, train_mode=True)
    for i in range(len(dataset) - 1):
        output = dataset[i]
        output["image"].size
        output["label"].size
        break
