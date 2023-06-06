from .base_model import *
import torch.nn as nn
import torch.nn.functional as F
from .unet_parts import *

class UNet_baysian(BaseModel):
    def __init__(self, n_classes):
        n_channels = 1
        super(UNet_baysian, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.drop1 = nn.Dropout2d()
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.drop2 = nn.Dropout2d()
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.drop1(x4)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.drop2(x)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        return x