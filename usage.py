from pathlib import Path
from PIL import Image
import os

import numpy as np
import torch
import cv2
import pytorch_lightning as pl

from human_parsing import HumanParsing
from human_parsing_dataset import HumanParsingDataset
from configs.config import read_conf_file


def save_images(img_tensors, img_names, save_dir):
    for img_tensor, img_name in zip(img_tensors, img_names):
        tensor = (img_tensor.clone() + 1) * 0.5 * 255
        tensor = tensor.cpu().clamp(0, 255)
        array = tensor.numpy().astype("uint8")
        if array.shape[0] == 1:
            array = array.squeeze(0)
        elif array.shape[0] == 3:
            array = array.swapaxes(0, 1).swapaxes(1, 2)
        Image.fromarray(array).save(os.path.join(save_dir, f"{img_name}.png"))


if __name__ == "__main__":
    hparams = read_conf_file(yaml_path="configs/configs/conv_unet_Adam_GCC.yaml")
    weight_path = "best/last.ckpt"
    model = (
        HumanParsing(hparams=hparams)
        .load_from_checkpoint(weight_path, hparams=hparams)
        .eval()
    )
    dataset = HumanParsingDataset(
        dataset_path=hparams["dataset_path"], train_mode=False
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
    )
    for batch in dataloader:
        with torch.no_grad():
            output = model.forward(batch=batch)[1]
        output = output.cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
        smgm_path = Path("test").joinpath(f"{batch['image_name'][0]}.png").as_posix()
        cv2.imwrite(smgm_path, output)
        break
