import torch
import torch.nn as nn


class QNet(nn.Module):
    def __init__(self):
        # TODO: add more hyper parameters
        super().__init__()
        self.rotate_layer = nn.Sequential(nn.Conv2d(1, 4, 5, 1, 2))
        self.deep_sample = nn.Sequential(
            nn.Conv2d(4, 8, 3, 1, 1),
            nn.LeakyReLU(),
        )
        self.res_block1 = nn.Sequential(
            ResModule(8, 32),
            nn.BatchNorm2d(num_features=32),
            ResModule(32, 16),
            nn.BatchNorm2d(num_features=16),
        )

        self.shrink_conv1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 2, 2),
            nn.LeakyReLU(),
        )
        self.res_block2 = nn.Sequential(
            ResModule(16, 32),
            nn.BatchNorm2d(num_features=32),
            ResModule(32, 16),
            nn.BatchNorm2d(num_features=16),
        )
        self.shrink_conv2 = nn.Sequential(
            nn.Conv2d(16, 8, 3, 2, 2),
            nn.LeakyReLU(),
        )
        self.adaptive_avg = nn.AdaptiveAvgPool2d(1)
        self.output = nn.Sequential(
            nn.Linear(8, 4),
            nn.LeakyReLU(),
        )
        # self.dnn = nn.Sequential(
        #     nn.Linear(16, 32),
        #     nn.ReLU(),
        #     # nn.Linear(32, 128),
        #     # nn.Linear(128,32),
        #     nn.Linear(32, 4),
        #     nn.ReLU(),
        # )

    def normalize(self, x):
        x = x - 2048
        x = x / 4096
        return x

    def invers_normalize(self, x):
        x = x * 4096
        x = x + 2048
        return x

    def avg_rotate(self, state):
        """
        input a state [B,4,4] rotate and average
        """
        r1 = torch.rot90(state, 1, [-2, -1])
        r2 = torch.rot90(state, 2, [-2, -1])
        r3 = torch.rot90(state, 3, [-2, -1])
        x = (
            self.rotate_layer(r1)
            + self.rotate_layer(r2)
            + self.rotate_layer(r3)
            + self.rotate_layer(state)
        ) * 0.25
        return x

    def forward(self, x: torch.Tensor):
        """
        return logits, shape [batch, 4]
        """
        s = self.normalize(x)
        s = self.avg_rotate(s)
        s = self.deep_sample(s)
        s = self.res_block1(s)
        s = self.shrink_conv1(s)
        s = self.res_block2(s)
        s = self.shrink_conv2(s)
        s = self.adaptive_avg(s)
        s = torch.squeeze(s, -1)
        s = torch.squeeze(s, -1)
        out = self.output(s)
        out = self.invers_normalize(out)
        return out

    # def forward(self, x):
    #     x = torch.flatten(x, start_dim=1, end_dim=-1)
    #     x = self.dnn(x)
    #     return x

    def get_action(self, state):
        logits = self(state)
        return torch.max(logits.reshape(-1, 4), 1)[1].data.cpu().numpy()[0]


class ResModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels * 2, out_channels, 3, 1, 1),
            nn.LeakyReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.conv_x(x)
        return x1 + x2
