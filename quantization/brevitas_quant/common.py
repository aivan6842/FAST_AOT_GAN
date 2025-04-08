import torch
from brevitas.nn import QuantConv2d
from torch import nn

from AOT_GAN.src.model.common import BaseNetwork


class QuantInpaintGenerator(BaseNetwork):
    def __init__(self, args):  # 1046
        super(QuantInpaintGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            QuantConv2d(4, 64, 7),
            nn.ReLU(True),
            QuantConv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(True),
            QuantConv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(True),
        )

        self.middle = nn.Sequential(*[AOTBlock(256, args.rates) for _ in range(args.block_num)])

        self.decoder = nn.Sequential(
            UpConv(256, 128), nn.ReLU(True), UpConv(128, 64), nn.ReLU(True), QuantConv2d(64, 3, 3, stride=1, padding=1), nn.Tanh()
        )

        self.init_weights()

        self.activations = []
        for i in range(args.block_num):
            self.middle.register_forward_hook(self.get_activation(f"middle.{i}"))

    def get_activation(self, name):
        def hook(model, input, output):
            self.activations.append(output.detach())
        return hook

    def forward(self, x, mask):
        x = torch.cat([x, mask], dim=1)
        x = self.encoder(x)
        x_mid = self.middle(x)
        x = self.decoder(x_mid)
        # x = torch.tanh(x)
        acts = torch.stack(self.activations) if self.activations else torch.tensor([])
        self.activations = []
        return x, acts


class UpConv(nn.Module):
    def __init__(self, inc, outc, scale=2):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv = QuantConv2d(inc, outc, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True))


class AOTBlock(nn.Module):
    def __init__(self, dim, rates):
        super(AOTBlock, self).__init__()
        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                "block{}".format(str(i).zfill(2)),
                nn.Sequential(
                    nn.ReflectionPad2d(rate), QuantConv2d(dim, dim // 4, 3, padding=0, dilation=rate), nn.ReLU(True)
                ),
            )
        self.fuse = nn.Sequential(nn.ReflectionPad2d(1), QuantConv2d(dim, dim, 3, padding=0, dilation=1))
        self.gate = nn.Sequential(nn.ReflectionPad2d(1), QuantConv2d(dim, dim, 3, padding=0, dilation=1))

    def forward(self, x):
        out = [self.__getattr__(f"block{str(i).zfill(2)}")(x) for i in range(len(self.rates))]
        out = torch.cat(out, 1)
        out = self.fuse(out)
        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + out * mask


def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat