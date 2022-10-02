"""
code made by Korneliusz Lewczuk korneliuszlewczuk@gmail.com
sorces:
https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cyclegan/cyclegan.py
https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py
https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/pix2pix/models.py
https://www.programmersought.com/article/97425131352/
https://gombru.github.io/2018/05/23/cross_entropy_loss/
https://www.youtube.com/watch?v=Hgg8Xy6IRig
https://github.com/python-engineer/pytorch-examples/blob/master/pytorch-lightning/lightning.py
https://colab.research.google.com/drive/1F_RNcHzTfFuQf-LeKvSlud6x7jXYkG31#scrollTo=ArrPXFM371jR
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl

from u_net_models.u_net_models import (
    U_ConvNext,
    GeneratorConvNext001,
    GeneratorResNet,
    U_Net2,
    R2U_Net2,
    AttU_Net2,
    R2AttU_Net2,
    init_weights,
)
from optims.Adam import Adam_GCC2, AdamW_GCC
from optims.SGD import SGD_GCC
from losses.canny_loss import CannyLoss
from torchmetrics import JaccardIndex


class HumanParsing(pl.LightningModule):
    def __init__(
        self,
        hparams: dict,
    ):
        super().__init__()
        self._hparams = hparams
        self.model = self.init_model()
        self.criterionCrossEntropy = nn.CrossEntropyLoss()
        self.criterionCANNY = CannyLoss()
        self.criterionIOU = JaccardIndex(num_classes=9)

        self.model = self.model.to(self.device)

        if not self._hparams["checkpoint_path"]:
            # self.save_hyperparameters()
            init_weights(self.model)

    def init_model(self):
        if self._hparams["model_name"] == "unet":
            model = U_Net2(
                img_ch=self._hparams["model_in_channels"],
                output_ch=self._hparams["model_out_channels"],
            )
        elif self._hparams["model_name"] == "aunet":
            model = AttU_Net2(
                img_ch=self._hparams["model_in_channels"],
                output_ch=self._hparams["model_out_channels"],
            )
        elif self._hparams["model_name"] == "r2unet":
            model = R2U_Net2(
                img_ch=self._hparams["model_in_channels"],
                output_ch=self._hparams["model_out_channels"],
            )
        elif self._hparams["model_name"] == "r2aunet":
            model = R2AttU_Net2(
                img_ch=self._hparams["model_in_channels"],
                output_ch=self._hparams["model_out_channels"],
            )
        elif self._hparams["model_name"] == "conv_unet":
            model = U_ConvNext(
                img_ch=self._hparams["model_in_channels"],
                output_ch=self._hparams["model_out_channels"],
            )
        elif self._hparams["model_name"] == "att_conv_unet":
            model = AttU_ConvNext(
                img_ch=self._hparams["model_in_channels"],
                output_ch=self._hparams["model_out_channels"],
            )

        elif self._hparams["model_name"] == "resnet":
            model = GeneratorResNet(
                img_ch=self._hparams["model_in_channels"],
                output_ch=self._hparams["model_out_channels"],
            )
        elif self._hparams["model_name"] == "convnext_resnet":
            model = GeneratorConvNext001(
                img_ch=self._hparams["model_in_channels"],
                output_ch=self._hparams["model_out_channels"],
            )
        return model

    def forward(self, batch):
        image = batch["image"]
        output_generator = self.model(image)
        output_generator_mask = torch.argmax(output_generator, dim=1)
        return output_generator, output_generator_mask

    def training_step(self, batch, _):
        image = batch["image"]
        label = batch["label"]

        output_generator = self.model(image)
        output_generator_mask = torch.argmax(output_generator, dim=1)

        cross_entropy_loss = self.criterionCrossEntropy(output_generator, label)
        output_generator_mask = torch.unsqueeze(output_generator_mask, dim=1)
        label = torch.unsqueeze(label, dim=1)
        canny_loss = self.criterionCANNY(output_generator_mask.float(), label.float())
        iou_loss = self.criterionIOU(output_generator_mask, label)
        loss = (
            self._hparams["cross_entropy_loss_lambda"] * cross_entropy_loss
            + self._hparams["iou_loss_lambda"] * iou_loss
            + self._hparams["canny_loss_lambda"] * canny_loss
        )

        self.log("cross_entropy_loss/train", cross_entropy_loss)
        self.log("canny_loss/train", canny_loss)
        self.log("iou_loss/train", iou_loss)
        self.log("loss/train", loss)
        return loss

    def validation_step(self, batch, _):
        image = batch["image"]
        label = batch["label"]

        output_generator = self.model(image)
        output_generator_mask = torch.argmax(output_generator, dim=1)

        cross_entropy_loss = self.criterionCrossEntropy(output_generator, label)
        output_generator_mask = torch.unsqueeze(output_generator_mask, dim=1)
        label = torch.unsqueeze(label, dim=1)
        canny_loss = self.criterionCANNY(output_generator_mask.float(), label.float())
        iou_loss = self.criterionIOU(output_generator_mask, label)
        loss = (
            self._hparams["cross_entropy_loss_lambda"] * cross_entropy_loss
            + self._hparams["iou_loss_lambda"] * iou_loss
            + self._hparams["canny_loss_lambda"] * canny_loss
        )

        self.log("cross_entropy_loss/valid", cross_entropy_loss)
        self.log("canny_loss/valid", canny_loss)
        self.log("iou_loss/valid", iou_loss)
        self.log("loss/valid", loss)
        self.log("loss_valid", loss)
        return [
            [
                output_generator_mask,
                label,
            ]
        ]

    def validation_epoch_end(self, validation_step_outputs):
        visuals_test_preprocesed = [[]]
        for step_idx, valid_step_output in enumerate(validation_step_outputs):
            if step_idx > 10:
                break
            for visual_test in valid_step_output[0]:
                visual_test = visual_test.to(torch.float32)
                visuals_test_preprocesed[0].append(visual_test)
        # board_add_images(
        #     board=self.logger.experiment,
        #     tag_name=f"{self._hparams['tags']}_valid",
        #     img_tensors_list=visuals_test_preprocesed,
        #     step_count=self.current_epoch,
        # )

    def test_step(self, batch, _):
        image = batch["image"]
        label = batch["label"]

        output_generator = self.model(image)
        output_generator_mask = torch.argmax(output_generator, dim=1)

        cross_entropy_loss = self.criterionCrossEntropy(output_generator, label)
        output_generator_mask = torch.unsqueeze(output_generator_mask, dim=1)
        label = torch.unsqueeze(label, dim=1)
        canny_loss = self.criterionCANNY(output_generator_mask.float(), label.float())
        iou_loss = self.criterionIOU(output_generator_mask, label)
        loss = 1 * cross_entropy_loss + 1 * iou_loss + 1 * canny_loss

        self.log("cross_entropy_loss/test", cross_entropy_loss)
        self.log("canny_loss/test", canny_loss)
        self.log("iou_loss/test", iou_loss)
        self.log("loss/test", loss)
        self.log("loss_test", loss)

        return [[output_generator_mask, label]]

    def test_epoch_end(self, test_step_outputs):
        visuals_test_preprocesed = [[]]
        for step_idx, valid_step_output in enumerate(test_step_outputs):
            if step_idx > 10:
                break
            for visual_test in valid_step_output[0]:
                visual_test = visual_test.to(torch.float32)
                visuals_test_preprocesed[0].append(visual_test)
        # board_add_images(
        #     board=self.logger.experiment,
        #     tag_name=f"{self._hparams['tags']}_valid",
        #     img_tensors_list=visuals_test_preprocesed,
        #     step_count=self.current_epoch,
        # )

    def configure_optimizers(self):
        lr = self._hparams["lr"]
        if self._hparams["optim_type"] == "Adam_GCC2":
            optimizer = Adam_GCC2(self.model.parameters(), lr=lr)
        elif self._hparams["optim_type"] == "AdamW_GCC":
            optimizer = AdamW_GCC(self.model.parameters(), lr=lr)
        elif self._hparams["optim_type"] == "SGD_GCC":
            optimizer = SGD_GCC(self.model.parameters(), lr=lr)
        return optimizer
