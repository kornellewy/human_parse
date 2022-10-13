import numpy as np
import torch
import cv2
from pathlib import Path

from configs.config import read_conf_file
from modules.human_parsing_with_classifcation import HumanParsingWtihClassifcation
from data_module.human_parsing_data_module_syntetic import HumanParsingDataModule


def create_evaluation_results(batch, prediction, output_path):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    results = []
    for i in range(batch["image"].shape[0]):
        i_prediction = prediction[i]
        i_prediction = i_prediction.cpu().numpy().transpose(1, 2, 0)
        i_prediction = np.asarray(
            np.argmax(i_prediction, axis=2), dtype=np.uint8
        ).reshape((224, 224, 1))
        i_prediction = i_prediction.squeeze(-1)
        h, w = i_prediction.shape
        i_prediction = i_prediction * 25
        i_prediction = cv2.cvtColor(i_prediction, cv2.COLOR_GRAY2BGR)

        i_image = batch["image"][i]
        i_image = denormalize(i_image.cpu().numpy(), mean, std)
        i_image = i_image.transpose(1, 2, 0).reshape((h, w, 3))

        i_label = batch["label"][i]
        i_label = i_label.cpu().numpy().astype(np.uint8)
        i_label = i_label * 25
        i_label = cv2.cvtColor(i_label, cv2.COLOR_GRAY2BGR).astype(np.uint8)

        i_result = np.vstack([i_image, i_label, i_prediction])
        results.append(i_result)
    results_as_image = np.hstack(results)
    cv2.imwrite(output_path, results_as_image)


def denormalize(img, mean, std):
    c, _, _ = img.shape
    for idx in range(c):
        img[idx, :, :] = img[idx, :, :] * std[idx] + mean[idx]
    return img


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    hparams = read_conf_file(yaml_path="configs/configs/conv_unet_SGD_GCC.yaml")
    weight_path = "best/conv_unet_SGD_GCC-epoch=49-loss_valid=0.86.ckpt"
    module = (
        HumanParsingWtihClassifcation(hparams=hparams)
        .load_from_checkpoint(weight_path, hparams=hparams)
        .eval()
    )
    datamodule = HumanParsingDataModule(hparams=hparams)
    datamodule.setup("")
    test_dataloader = datamodule.test_dataloader()
    output_path = f"{Path(weight_path).stem}.png"
    for batch_idx, batch in enumerate(test_dataloader):
        with torch.no_grad():
            prediction = module.predict_step(batch, batch_idx)
        create_evaluation_results(batch, prediction, output_path)
        break
