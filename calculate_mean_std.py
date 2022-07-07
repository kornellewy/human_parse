from pip import main
import torch
from tqdm import tqdm

from dataloader import HumanParsingDataset


def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data in tqdm(dataloader):
        data = data["image"]
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


# https://deeplizard.com/learn/video/lu7TCu7HeYc
if __name__ == "__main__":
    dataset = HumanParsingDataset(
        "J:/deepcloth/datasets/human_body_parsing/kjn_parse_dataset_001",
        train_mode=False,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, num_workers=1)
    mean, std = get_mean_and_std(dataloader)
    print(mean, std)
