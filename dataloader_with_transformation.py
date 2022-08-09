from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
import albumentations as A

from virtual_try_on.utils.utils import load_images, visualize_agumented_data


def load_images_to_image_name_path_map(dataset_path: Path) -> dict:
    return {Path(path).stem: path for path in load_images(dataset_path.as_posix())}


class HumanParsingWithTransformation(torch.utils.data.Dataset):
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
    atr_class_map = {
        1: "Hat",
        2: "Hair",
        3: "Sunglasses",
        4: "UpperClothes",
        5: "Skirt",
        6: "Pants",
        7: "Dress",
        8: "Belt",
        9: "Left-shoe",
        10: "Right-shoe",
        11: "Face+Neck",
        12: "Left-leg",
        13: "Right-leg",
        14: "Left-arm",
        15: "Right-arm",
        16: "Bag",
        17: "Scarf",
    }
    atr_to_kjn_map = {
        0: 0,
        1: 1,
        2: 1,
        3: 2,
        4: 3,
        5: 4,
        6: 4,
        7: 3,
        9: 5,
        10: 6,
        11: 2,
        12: 5,
        13: 6,
        14: 7,
        15: 8,
        16: 0,
        17: 0,
    }
    lip_to_kjn_map = {
        0: 0,
        1: 1,
        2: 1,
        3: 0,
        4: 2,
        5: 3,
        6: 3,
        7: 3,
        8: 0,
        9: 4,
        10: 4,
        11: 0,
        12: 4,
        13: 2,
        14: 7,
        15: 8,
        16: 5,
        17: 6,
        18: 5,
        19: 6,
    }

    def __init__(
        self,
        dataset_path: str,
        one_shot_dataset_image_path: str,
        one_shot_dataset_label_path: str,
        atr_dataset_image_path: str,
        atr_dataset_label_path: str,
        train_lip_dataset_image_path: str,
        train_lip_dataset_label_path: str,
        valid_lip_dataset_image_path: str,
        valid_lip_dataset_label_path: str, 
        output_size: tuple = (224, 224),
        train_mode: bool = True,
    ) -> None:
        self.dataset_path = dataset_path
        self.output_size = output_size
        self.train_mode = train_mode
        
        