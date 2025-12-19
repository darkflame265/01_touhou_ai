# vision/models/center_regressor.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterRegressor(nn.Module):
    """
    Tiny CNN that predicts (x, y, conf) from a (C,H,W) input.
    We use C=4 (4-frame grayscale stack) by default.

    Output:
      - x, y: [0,1] (sigmoid)
      - conf: [0,1] (sigmoid)
    """
    def __init__(self, in_ch: int = 4, base: int = 32):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )

        self.b1 = block(base, base)
        self.p1 = nn.MaxPool2d(2)  # /2

        self.b2 = block(base, base * 2)
        self.p2 = nn.MaxPool2d(2)  # /4

        self.b3 = block(base * 2, base * 4)
        self.p3 = nn.MaxPool2d(2)  # /8

        self.b4 = block(base * 4, base * 4)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 4, base * 2),
            nn.ReLU(inplace=True),
            nn.Linear(base * 2, 3),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.b1(x); x = self.p1(x)
        x = self.b2(x); x = self.p2(x)
        x = self.b3(x); x = self.p3(x)
        x = self.b4(x)
        x = self.head(x)
        x = torch.sigmoid(x)  # (x,y,conf) all in [0,1]
        return x


def loss_center_regression(pred, target_xy, target_conf, lambda_conf: float = 1.0):
    """
    pred: (B,3) sigmoid outputs
    target_xy: (B,2) in [0,1]
    target_conf: (B,1) in {0,1}
    Loss:
      coord SmoothL1, weighted by target_conf
      + BCE conf
    """
    pred_xy = pred[:, 0:2]
    pred_conf = pred[:, 2:3]

    coord = F.smooth_l1_loss(pred_xy, target_xy, reduction="none")  # (B,2)
    coord = coord.mean(dim=1, keepdim=True)                         # (B,1)
    coord = (coord * target_conf).sum() / (target_conf.sum().clamp_min(1.0))

    conf = F.binary_cross_entropy(pred_conf, target_conf)
    return coord + lambda_conf * conf, coord.detach(), conf.detach()
