import torch
import torch.nn as nn


class QNet(nn.Module):
    def __init__(self):
        # TODO: add more hyper parameters
        super().__init__()
        # self.deep_sample = nn.Sequential(
        #     nn.Conv2d(1, 8, 3, 1, 1), nn.Conv2d(8, 64, 3, 1, 1)
        # )
        # self.res_block = nn.Sequential(ResModule(64, 32), ResModule(32, 16))
        # self.shrink_conv = nn.Conv2d(16, 16, 3, 2, 2)
        # self.adaptive_avg = nn.AdaptiveAvgPool2d(1)
        # self.output = nn.Linear(16, 4)
        self.dnn = nn.Sequential(
            nn.Linear(16, 32),
            # nn.Linear(32, 128),
            # nn.Linear(128,32),
            nn.Linear(32, 4),
        )

    # def forward(self, x: torch.Tensor):
    #     """
    #     return logits, shape [batch, 4]
    #     """
    #     s = self.deep_sample(x)
    #     s = self.res_block(s)
    #     s = self.shrink_conv(s)
    #     s = self.adaptive_avg(s)
    #     s = torch.squeeze(s, -1)
    #     s = torch.squeeze(s, -1)
    #     out = self.output(s)
    #     return out
    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.dnn(x)
        return x

    def get_action(self, state):
        logits = self(state)
        return torch.max(logits.reshape(-1, 4), 1)[1].data.cpu().numpy()[0]


class ResModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 3, 1, 1),
            nn.Conv2d(in_channels * 2, out_channels, 3, 1, 1),
        )
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, 1, 0))

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.conv_x(x)
        return x1 + x2
