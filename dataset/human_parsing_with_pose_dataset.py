from pathlib import Path
import json

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
import albumentations as A

from utils.utils import load_images, load_images_to_image_name_path_map, load_poses_to_image_name_path_map


class HumanParsingWithPoseDataset(torch.utils.data.Dataset):
    IMAGES_DIR_NAME = "image"
    LABELS_DIR_NAME = "label"
    POSE_DIR_NAME = "pose"
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
        keypoint_params=A.KeypointParams(format="xy"),
        additional_targets={
            "label": "mask",
            "keypoints": "keypoints",
        },
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

    TRANSFORMS_MASK = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
    )

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
        self.pose_dir_path = Path(self.dataset_path).joinpath(self.POSE_DIR_NAME)

        self.image_paths = load_images(self.images_dir_path.as_posix())
        self.label_name_to_path_map = load_images_to_image_name_path_map(
            self.labels_dir_path
        )
        self.pose_name_to_path_map = load_poses_to_image_name_path_map(
            self.pose_dir_path
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict:
        image_path = self.image_paths[index]
        image_name = Path(image_path).stem
        label_path = self.label_name_to_path_map[image_name]
        pose_path = self.pose_name_to_path_map[image_name.replace("_keypoints", "")]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_path, 0)
        pose = self.read_pose_points(pose_path=pose_path)
        # pose = self._check_and_add_number_of_pose_points(pose=pose)

        if self.train_mode:
            agumented_data = self.AGUMENTATIONS(image=image, keypoints=pose)
            image = agumented_data["image"].astype(np.uint8)
            pose = np.array(agumented_data["keypoints"])

        pose = self._create_pose_map(pose=pose)
        image = cv2.resize(image, self.output_size, cv2.INTER_AREA).astype(np.uint8)
        label = cv2.resize(label, self.output_size, cv2.INTER_AREA).astype(np.uint8)

        label = torch.from_numpy(label).long()
        image = self.TRANSFORMS(image)
        pose = self._transform_stack_of_gray_images(pose)
        return {
            "image_name": image_name,
            "image": image,
            "label": label,
            "pose": pose,
        }

    def read_pose_points(self, pose_path: str) -> np.ndarray:
        with open(pose_path, "r") as file:
            pose_label = json.load(file)
            pose_data = pose_label["people"][0]["pose_keypoints"]
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))
            pose_data = np.delete(pose_data, 2, axis=1)
            pose_data = np.where(pose_data == 192, 192 - 0.01, pose_data)
            pose_data = np.where(pose_data == 256, 256 - 0.01, pose_data)
            pose_data[:, 0] = np.where(
                pose_data[:, 0] > 192, 192 - 0.01, pose_data[:, 0]
            )
            pose_data[:, 1] = np.where(
                pose_data[:, 1] > 256, 256 - 0.01, pose_data[:, 1]
            )
        pose_data = pose_data.astype(np.uint8)
        return pose_data

    def _create_pose_map(self, pose: np.ndarray) -> np.ndarray:
        point_numers = pose.shape[0]
        point_pose_map = np.zeros(shape=(18, self.output_size[1], self.output_size[0]))
        for i_point_num in range(18):
            single_point_map = np.zeros(
                shape=(self.output_size[1], self.output_size[0])
            ).astype(np.uint8)
            if i_point_num <= point_numers - 1:
                point_x = int(pose[i_point_num, 0])
                point_y = int(pose[i_point_num, 1])
                if point_x > 1 and point_y > 1:
                    single_point_map = cv2.circle(
                        single_point_map,
                        (point_x, point_y),
                        radius=5,
                        color=(255),
                        thickness=-1,
                    )
            point_pose_map[i_point_num, :, :] = single_point_map
        return point_pose_map

    def _transform_stack_of_gray_images(
        self, stack_of_gray_images: np.ndarray
    ) -> torch.tensor:
        gray_images_number = stack_of_gray_images.shape[0]
        stack = torch.zeros(
            gray_images_number, self.output_size[1], self.output_size[0]
        )
        for i in range(0, gray_images_number):
            single_gray_image = stack_of_gray_images[i, :, :]
            if np.max(single_gray_image) > 1:
                single_gray_image = np.clip(single_gray_image, a_min=0, a_max=1)
            single_gray_image = self.TRANSFORMS_MASK(single_gray_image)
            stack[i, :, :] = single_gray_image
        return stack


if __name__ == "__main__":
    dataset_path = Path("J:/deepcloth/datasets/human_body_parsing/kjn_parse_dataset")
    dataset = HumanParsingWithPoseDataset(dataset_path=dataset_path, train_mode=True)
    for i in range(len(dataset) - 1):
        output = dataset[i]
        print(output["image"].size())
        print(output["label"].size())
        print(output["pose"].size())
        break
