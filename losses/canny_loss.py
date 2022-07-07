from matplotlib.pyplot import xlabel, ylim
import torch.nn as nn
import kornia


class CannyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, x, y):
        x, y = kornia.filters.canny(x)[1], kornia.filters.canny(y)[1].detach()
        return self.criterion(x, y)
