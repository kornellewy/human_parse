"""
code made by Korneliusz Lewczuk korneliuszlewczuk@gmail.com
sorces: 
https://github.com/cheind/py-thin-plate-spline/blob/master/TPS.ipynb
"""

import os
import numpy as np
import random
import cv2
from pathlib import Path
from typing import List
from collections import defaultdict

import torch
from torchvision.utils import save_image


def load_jsons(dir_path: str) -> List[str]:
    jsons = []
    valid_jsons = [".json"]
    for f in os.listdir(dir_path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_jsons:
            continue
        jsons.append(os.path.join(dir_path, f))
    return jsons


def load_images(path):
    images = []
    valid_images = [".jpeg", ".jpg", ".png"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        images.append(os.path.join(path, f))
    return images


def load_images_to_image_name_path_map(dataset_path: Path) -> dict:
    return {Path(path).stem: path for path in load_images(dataset_path.as_posix())}


def load_poses_to_image_name_path_map(dataset_path: Path) -> dict:
    return {
        Path(path).stem.replace("_keypoints", ""): path
        for path in load_jsons(dataset_path.as_posix())
    }


def yield_files_with_extensions(folder_path, file_extension):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(file_extension):
                yield os.path.join(root, file)


def load_images2(path):
    images_exts = (".jpg", ".jpeg", ".png")
    return yield_files_with_extensions(path, images_exts)


def random_idx_with_exclude(exclude, idx_range):
    randInt = random.randint(idx_range[0], idx_range[1])
    return (
        random_idx_with_exclude(exclude, idx_range) if randInt in exclude else randInt
    )


def visualize_agumented_data(agumented_data):
    test_save_path = "test_files"
    test_save_path = Path(test_save_path)
    test_save_path.mkdir(exist_ok=True, parents=True)
    for key, item in agumented_data.items():
        if key in ["image_name", "pose_map", "image_parse"]:
            continue
        image_name = f"{key}.png"
        item_save_path = (test_save_path / image_name).as_posix()

        if type(item) is np.ndarray:
            if len(item.shape) < 3:
                item = cv2.cvtColor(item, cv2.COLOR_GRAY2BGR)
            else:
                item = cv2.cvtColor(item, cv2.COLOR_BGR2RGB)
            if np.amax(item) < 25:
                item = item * 10
            cv2.imwrite(item_save_path, item)
        else:
            if item.shape[0] < 4 and len(item.shape) == 3:
                save_image(item.float(), item_save_path)
            elif item.shape[0] > 3 or len(item.shape) > 3:
                item = item.squeeze(0)
                item = (torch.argmax(item, dim=0) / 255) * 10
                save_image(item.float(), item_save_path)


def get_structure(dataset_path: Path, dirs_names: dict) -> dict:
    dirs_paths = {}
    for key_name, dir_name in dirs_names.items():
        new_key_name = key_name.replace("name", "path")
        dir_path = dataset_path.joinpath(dir_name)
        dirs_paths[new_key_name] = dir_path.as_posix()
    return dirs_paths


def get_image_name_to_cycle_images_paths_map(images_paths: List[str]) -> dict:
    image_name_to_cycle_images_paths = defaultdict(list)
    for image_path in images_paths:
        image_name = Path(image_path).stem.split("___")[0]
        image_name_to_cycle_images_paths[image_name].append(image_path)
    return image_name_to_cycle_images_paths


# def denormalize_tensor_3_channel(normalize_tensor):
#     inv_normalize = transforms.Normalize(
#         mean=[0.5, 0.5, 0.5],
#         std=[0.5, 0.5, 0.5]
#     )
#     return inv_normalize(normalize_tensor)

# def denormalize_tensor_1_channel(normalize_tensor):
#     inv_normalize = transforms.Normalize(
#         mean=[0.5],
#         std=[0.5]
#     )
#     return inv_normalize(normalize_tensor)
