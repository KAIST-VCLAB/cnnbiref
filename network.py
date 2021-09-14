import torch
import torch.nn as nn


class DeblurModuleDouble(nn.Module):
    def __init__(self):
        super(DeblurModuleDouble, self).__init__()
        self.backbone = FeatExtractorSimple_v4()
        self.head_e = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 3, stride=2, padding=1, output_padding=1, dilation=1),
            nn.PReLU(),
            nn.Conv2d(3, 3, 1, 1, 0, 1, 1, False)
        )
        self.head_o = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 3, stride=2, padding=1, output_padding=1, dilation=1),
            nn.PReLU(),
            nn.Conv2d(3, 3, 1, 1, 0, 1, 1, False)
        )

        def init_weights(m):
            if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                torch.nn.init.xavier_uniform_(m.weight)

        # init weight
        self.backbone.apply(init_weights)
        self.head_e.apply(init_weights)
        self.head_o.apply(init_weights)

    def forward(self, img):
        out = self.backbone(img)
        im_e = -self.head_e(out) + img
        im_o = self.head_o(out) + img
        return im_e, im_o


class FeatExtractorSimple_v4(nn.Module):
    def __init__(self):
        super(FeatExtractorSimple_v4, self).__init__()
        self.conv0 = nn.Sequential(
            SingleConvBlock(3, 32, 1, 1, 1),
            ResBlock(32, 32, 1, 1, 1),
            ResBlock(32, 32, 1, 1, 1),
            SingleConvBlock(32, 64, 2, 1, 1)
        )
        self.block1 = self._make_block(64, 1, 1)
        self.block2 = self._make_block(64, 1, 1)
        self.block3 = self._make_block(64, 1, 1)
        self.block4 = self._make_block(64, 1, 1)
        self.block5 = self._make_block(64, 1, 1)
        self.block6 = self._make_block(64, 1, 1)
        self.block7 = self._make_block(64, 1, 1)

        self.branch = SingleConvBlock(512, 128, 1, 1, 1)

    def _make_block(self, feat_dim=64, stride=1, dilation=1):
        block = nn.Sequential(
            ResBlock(feat_dim, feat_dim, stride, dilation, p=dilation),
            ResBlock(feat_dim, feat_dim, stride, dilation, p=dilation))
        return block

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        x6 = self.block6(x5)
        x7 = self.block7(x6)
        out = torch.cat((x0, x1, x2, x3, x4, x5, x6, x7), dim=1)
        out = self.branch(out)
        return out


class SingleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dilation, pad):
        super(SingleConvBlock, self).__init__()
        self.block = nn.Sequential(
            SingleConv3x3(in_channels, out_channels, stride, dilation, pad),
            nn.PReLU()
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, i, o, s, d, p):
        super(ResBlock, self).__init__()
        self.double_conv = nn.Sequential(
            SingleConv3x3(i, o, s, d, p),
            nn.PReLU(),
            SingleConv3x3(o, o, s, d, p),
        )

    def forward(self, x):
        out = self.double_conv(x)
        return out + x


def SingleConv3x3(in_channels, out_channels, stride, dilation, pad):
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=pad, dilation=dilation,
                      groups=1, bias=False, padding_mode='replicate')
    return layer