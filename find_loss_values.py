from pathlib import Path
from typing import List
from itertools import combinations_with_replacement

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from configs.config import read_conf_file
from human_parsing import HumanParsing
from human_parsing_data_module import HumanParsingDataModule


def find_loss_values(experiments: List[str]) -> None:
    results = {}
    for experiment in experiments:
        hparams = read_conf_file(yaml_path=experiment)
        all_combinations = list(combinations_with_replacement(range(0, 11, 5), 3))
        all_combinations = all_combinations[1:]
        print(len(all_combinations))
        print(all_combinations)
        for combination in all_combinations:
            try:
                print(combination)
                hparams["cross_entropy_loss_lambda"] = combination[0]
                hparams["iou_loss_lambda"] = combination[1]
                hparams["canny_loss_lambda"] = combination[2]
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
                    save_top_k=10,
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
                output = trainer.test(model=module, datamodule=datamodule)
                results[combination] = output[0]["loss_test"]
            except Exception as e:
                print(e)
                results[combination] = 0.0
    print(results)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    experiments = ["configs/configs/conv_unet_SGD_GCC.yaml"]
    find_loss_values(experiments=experiments)
