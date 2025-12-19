# vision/models/heatmap_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class HeatmapNet(nn.Module):
    """
    Simple encoder-decoder that outputs a 1-channel heatmap (H,W).
    Input: (B, C, H, W) where C=4 (frame stack), H=120, W=160
    Output: (B, 1, H, W) logits (NOT sigmoid)
    """
    def __init__(self, in_ch=4, base=32):
        super().__init__()

        def conv(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        # Encoder
        self.e1 = nn.Sequential(conv(in_ch, base), conv(base, base))
        self.p1 = nn.MaxPool2d(2)  # /2

        self.e2 = nn.Sequential(conv(base, base * 2), conv(base * 2, base * 2))
        self.p2 = nn.MaxPool2d(2)  # /4

        self.e3 = nn.Sequential(conv(base * 2, base * 4), conv(base * 4, base * 4))
        self.p3 = nn.MaxPool2d(2)  # /8

        self.mid = nn.Sequential(conv(base * 4, base * 4), conv(base * 4, base * 4))

        # Decoder
        self.u3 = nn.ConvTranspose2d(base * 4, base * 4, 2, stride=2)  # x2
        self.d3 = nn.Sequential(conv(base * 8, base * 2), conv(base * 2, base * 2))

        self.u2 = nn.ConvTranspose2d(base * 2, base * 2, 2, stride=2)
        self.d2 = nn.Sequential(conv(base * 4, base), conv(base, base))

        self.u1 = nn.ConvTranspose2d(base, base, 2, stride=2)
        self.d1 = nn.Sequential(conv(base * 2, base), conv(base, base))

        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)          # (B,base,H,W)
        e2 = self.e2(self.p1(e1))# (B,2b,H/2,W/2)
        e3 = self.e3(self.p2(e2))# (B,4b,H/4,W/4)
        m  = self.mid(self.p3(e3))  # (B,4b,H/8,W/8)

        y = self.u3(m)                 # (B,4b,H/4,W/4)
        y = torch.cat([y, e3], dim=1)  # (B,8b,H/4,W/4)
        y = self.d3(y)                 # (B,2b,H/4,W/4)

        y = self.u2(y)                 # (B,2b,H/2,W/2)
        y = torch.cat([y, e2], dim=1)  # (B,4b,H/2,W/2)
        y = self.d2(y)                 # (B,b,H/2,W/2)

        y = self.u1(y)                 # (B,b,H,W)
        y = torch.cat([y, e1], dim=1)  # (B,2b,H,W)
        y = self.d1(y)                 # (B,b,H,W)

        logits = self.head(y)          # (B,1,H,W)
        return logits


def soft_argmax_2d(logits, beta=10.0):
    """
    logits: (B,1,H,W)  (not sigmoid)
    returns:
      xy_norm: (B,2) in [0,1] (x then y)
      conf: (B,1) peak probability (0~1)
    """
    B, _, H, W = logits.shape
    x = logits.view(B, -1) * beta
    prob = F.softmax(x, dim=1)  # (B, H*W)

    # coordinates grid
    ys = torch.linspace(0, 1, H, device=logits.device)
    xs = torch.linspace(0, 1, W, device=logits.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # (H,W)
    xx = xx.reshape(-1)  # (H*W,)
    yy = yy.reshape(-1)

    x_exp = (prob * xx.unsqueeze(0)).sum(dim=1, keepdim=True)  # (B,1)
    y_exp = (prob * yy.unsqueeze(0)).sum(dim=1, keepdim=True)

    conf = prob.max(dim=1, keepdim=True).values  # peak prob
    xy = torch.cat([x_exp, y_exp], dim=1)        # (B,2)
    return xy, conf


def bce_dice_loss(logits, target):
    """
    logits: (B,1,H,W)
    target: (B,1,H,W) in {0..1} (gaussian peak)
    """
    p = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target)

    # dice
    eps = 1e-6
    inter = (p * target).sum(dim=(2,3))
    union = (p + target).sum(dim=(2,3))
    dice = 1.0 - ((2 * inter + eps) / (union + eps))
    dice = dice.mean()

    return bce + dice, bce.detach(), dice.detach()
