from pathlib import Path
from typing import List

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from configs.config import read_conf_file
from human_parsing import HumanParsing

# from human_parsing_with_pose import HumanParsingWithPose
# from human_parsing_wtih_classifcation import HumanParsingWtihClassifcation
# from human_parsing_data_module_with_classifcation import (
#     HumanParsingDataModuleWtihClassifcation,
# )
# from human_parsing_with_pose_with_classifcation import (
#     HumanParsingWtihPoseWtihClassifcation,
# )
from human_parsing_data_module import HumanParsingDataModule


def run_experiments(experiments: List[str]) -> None:
    for experiment in experiments:
        hparams = read_conf_file(yaml_path=experiment)
        print("hparams: ", hparams)
        module = HumanParsing(hparams=hparams)
        datamodule = HumanParsingDataModule(hparams=hparams)
        checkpoint_path = None
        if hparams["checkpoint_path"]:
            if Path(hparams["checkpoint_path"]).exists() and str(
                Path(hparams["checkpoint_path"])
            ).endswith(".ckpt"):
                checkpoint_path = hparams["checkpoint_path"]
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="loss_valid",
            dirpath="best/",
            filename=f"{hparams['tags']}" + "-{epoch:02d}-{loss_valid:.2f}",
            save_top_k=3,
            mode="min",
            save_last=True,
            every_n_epochs=1,
        )
        Path(hparams["base_dir_path"]).mkdir(exist_ok=True, parents=True)
        logger = TensorBoardLogger(
            save_dir=hparams["base_dir_path"], name=hparams["tags"]
        )
        trainer = pl.Trainer(
            gpus=[0],
            precision=16,
            check_val_every_n_epoch=1,
            max_epochs=hparams["epochs_num"],
            default_root_dir=hparams["base_dir_path"],
            detect_anomaly=True,
            resume_from_checkpoint=checkpoint_path,
            logger=logger,
            callbacks=[checkpoint_callback],
        )
        trainer.fit(module, datamodule)
        trainer.test(model=module, datamodule=datamodule)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    experiments = [
        # "configs/configs/conv_unet_Adam_GCC.yaml",
        # "configs/configs/unet_Adam_GCC.yaml",
        # "configs/configs/unet_SGD_GCC.yaml",
        "configs/configs/att_conv_unet_Adam_GCC.yaml"
    ]
    run_experiments(experiments=experiments)
