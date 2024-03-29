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

from optims.Adam import Adam_GCC2, AdamW_GCC
from u_net_models.u_net_models import U_ConvNextWithClassification
from torchmetrics import JaccardIndex


class HumanParsingWtihPoseWtihClassifcation(pl.LightningModule):
    def __init__(
        self,
        hparams: dict,
    ):
        super().__init__()
        self._hparams = hparams
        self.model = self.init_model()
        self.criterion_segmentation = nn.NLLLoss(weight=None)
        self.criterion_classification = nn.BCEWithLogitsLoss(weight=None)
        self.criterionIOU = JaccardIndex(num_classes=9)

    def init_model(self):
        if self._hparams["model_name"] == "conv_unet":
            model = U_ConvNextWithClassification(
                img_ch=self._hparams["model_in_channels"],
                output_ch=self._hparams["model_out_channels"],
            )
        return model

    def configure_optimizers(self):
        lr = self._hparams["lr"]
        if self._hparams["optim_type"] == "Adam_GCC2":
            optimizer = Adam_GCC2(self.model.parameters(), lr=lr)
        elif self._hparams["optim_type"] == "AdamW_GCC":
            optimizer = AdamW_GCC(self.model.parameters(), lr=lr)
        return optimizer

    def forward(self, batch):
        image = batch["image"]
        pose = batch["pose"]
        representation = torch.cat([image, pose], 1)
        output_predition, _ = self.model(representation)
        return output_predition

    def training_step(self, batch, _):
        image = batch["image"]
        label = batch["label"]
        pose = batch["pose"]
        label_classification = batch["label_classification"]
        representation = torch.cat([image, pose], 1)
        output_predition, output_classification = self.model(representation)
        segmentation_loss = self.criterion_segmentation(output_predition, label)
        classification_loss = self.criterion_classification(
            output_classification, label_classification
        )
        loss = segmentation_loss + 1.0 * classification_loss
        self.log("segmentation_loss/train", segmentation_loss)
        self.log("classification_loss/train", classification_loss)
        self.log("loss/train", loss)
        return loss

    def validation_step(self, batch, _):
        image = batch["image"]
        label = batch["label"]
        pose = batch["pose"]
        label_classification = batch["label_classification"]
        representation = torch.cat([image, pose], 1)
        output_predition, output_classification = self.model(representation)
        segmentation_loss = self.criterion_segmentation(output_predition, label)
        classification_loss = self.criterion_classification(
            output_classification, label_classification
        )
        loss = segmentation_loss + 1.0 * classification_loss
        self.log("segmentation_loss/valid", segmentation_loss)
        self.log("classification_loss/valid", classification_loss)
        self.log("loss/valid", loss)
        self.log("loss_valid", loss)
        return [
            [
                torch.argmax(output_predition, dim=1).unsqueeze(dim=1),
                label.unsqueeze(dim=1),
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
        pose = batch["pose"]
        representation = torch.cat([image, pose], 1)
        output_predition, _ = self.model(representation)
        loss = self.criterionIOU(output_predition, label)
        output_predition = output_predition.unsqueeze(dim=1)
        label = label.unsqueeze(dim=1)
        self.log("loss/test", loss)
        return [
            [
                torch.argmax(output_predition, dim=1),
                label,
            ]
        ]

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
