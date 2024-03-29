from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
import albumentations as A
import json

from utils.utils import load_images, load_images_to_image_name_path_map


class HumanParsingDataset(torch.utils.data.Dataset):
    IMAGES_DIR_NAME = "image"
    LABELS_DIR_NAME = "label"
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
            transforms.Normalize([0.5240, 0.4880, 0.4602], [0.3021, 0.2982, 0.3014]),
        ]
    )
    CLASS_MAP = {
        0: "Background",
        1: "Hat+Hair",
        2: "Sunglasses+Face+Neck",
        3: "UpperClothes+Dress+Coat",
        4: "Skirt+Pants",
        5: "Left-leg+Left-shoe",
        6: "Right-leg+Right-shoe",
        7: "Left-arm+Left-hend",
        8: "Right-arm+Right-hend",
    }

    def __init__(
        self,
        dataset_path: str,
        output_size: tuple = (224, 224),
        train_mode: bool = True,
    ) -> None:
        self.dataset_path = dataset_path
        self.output_size = output_size
        self.train_mode = train_mode
        self.images_dir_path = Path(self.dataset_path).joinpath(self.IMAGES_DIR_NAME)
        self.labels_dir_path = Path(self.dataset_path).joinpath(self.LABELS_DIR_NAME)

        self.image_paths = load_images(self.images_dir_path.as_posix())
        self.label_name_to_path_map = load_images_to_image_name_path_map(
            self.labels_dir_path
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict:
        image_path = self.image_paths[index]
        image_name = Path(image_path).stem
        label_path = self.label_name_to_path_map[image_name]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = cv2.imread(label_path, 0)

        if self.train_mode:
            agumented_data = self.AGUMENTATIONS(image=image)
            image = agumented_data["image"].astype(np.uint8)

        image = cv2.resize(image, self.output_size, cv2.INTER_AREA).astype(np.uint8)
        label = cv2.resize(label, self.output_size, cv2.INTER_AREA).astype(np.uint8)

        label = torch.from_numpy(label).long()
        image = self.TRANSFORMS(image)
        return {
            "image_name": image_name,
            "image": image,
            "label": label,
        }


if __name__ == "__main__":
    dataset_path = Path(
        "J:/deepcloth/datasets/human_body_parsing/kjn_parse_dataset_001"
    )
    dataset = HumanParsingDataset(dataset_path=dataset_path, train_mode=True)
    for i in range(len(dataset) - 1):
        output = dataset[i]
        output["image"].size
        output["label"].size
