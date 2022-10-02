import torch
import pytorch_lightning as pl

from dataset.human_parsing_dataset import HumanParsingDataset


class HumanParsingDataModule(pl.LightningDataModule):
    def __init__(self, hparams: dict):
        super().__init__()
        self._hparams = hparams

    def setup(self, stage=""):
        dataset = HumanParsingDataset(self._hparams["dataset_path"])
        if len(dataset) > 2000:
            train_size = int(0.98 * len(dataset))
            valid_size = int(0.01 * len(dataset))
            test_size = int(0.01 * len(dataset))
        else:
            train_size = int(0.90 * len(dataset))
            valid_size = int(0.05 * len(dataset))
            test_size = int(0.05 * len(dataset))
        rest = len(dataset) - train_size - valid_size - test_size
        train_size = train_size + rest
        self.train_set, self.valid_set, self.test_set = torch.utils.data.random_split(
            dataset, [train_size, valid_size, test_size]
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set, batch_size=self._hparams["batch_size"]
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_set, batch_size=self._hparams["batch_size"]
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_set, batch_size=self._hparams["batch_size"]
        )
