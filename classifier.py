import torch.nn as nn
import torch.nn.functional as F
from torch import save, load, device

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # nn.Conv2d: num of channels, num of filters, size of filters
        self.conv = nn.Sequential(
            nn.Conv2d(3, 6, 5, padding=2), # 32 *3 -> 32 *6 -> 16 *6
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 12, 5, padding=2), # 16 *6 -> 16 *12 -> 8 *12
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(12, 24, 5, padding=2), # 8 *12 -> 8 *24 -> 4 *24
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.dense = nn.Sequential(
            nn.Linear(4* 4* 24, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return x

if __name__ == '__main__':
    net = Net()
    net.load_state_dict(load('../weights/CNNv2.pt', map_location=device('cpu')))
