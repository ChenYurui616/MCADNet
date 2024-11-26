#decoder
import torch
from torch import nn
import torch.nn.functional as F

class EnLayer(nn.Module):
    def __init__(self, in_channel=64):
        super(EnLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.enlayer(x)
        return x


class LatLayer(nn.Module):
    def __init__(self, in_channel):
        super(LatLayer, self).__init__()
        self.convlayer = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.convlayer(x)
        return x


class DSLayer(nn.Module):
    def __init__(self, in_channel=64):
        super(DSLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.predlayer = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0))#, nn.Sigmoid())

    def forward(self, x):
        x = self.enlayer(x)
        x = self.predlayer(x)
        return x


class half_DSLayer(nn.Module):
    def __init__(self, in_channel=512):
        super(half_DSLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, int(in_channel/4), kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.predlayer = nn.Sequential(
            nn.Conv2d(int(in_channel/4), 1, kernel_size=1, stride=1, padding=0)) #, nn.Sigmoid())

    def forward(self, x):
        x = self.enlayer(x)
        x = self.predlayer(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.toplayer = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0))
        self.latlayer4 = LatLayer(in_channel=512)
        self.latlayer3 = LatLayer(in_channel=256)
        self.latlayer2 = LatLayer(in_channel=128)
   

        self.enlayer4 = EnLayer()
        self.enlayer3 = EnLayer()
        self.enlayer2 = EnLayer()


        self.dslayer4 = DSLayer()
        self.dslayer3 = DSLayer()
        self.dslayer2 = DSLayer()


    def _upsample_add(self, x, y):
        [_, _, H, W] = y.size()
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return x + y

    def forward(self, weighted_x5, x4, x3, x2, x1, H, W):
        preds = []
        p5 = self.toplayer(weighted_x5)
        
        p4 = self._upsample_add(p5, self.latlayer4(x4))
        p4 = self.enlayer4(p4)
        _pred = self.dslayer4(p4)
        preds.append(
            F.interpolate(_pred,
                          size=(H, W),
                          mode='bilinear', align_corners=False))

        p3 = self._upsample_add(p4, self.latlayer4(x3))
        p3 = self.enlayer3(p3)
        _pred = self.dslayer3(p3)
        preds.append(
            F.interpolate(_pred,
                          size=(H, W),
                          mode='bilinear', align_corners=False))

        p2 = self._upsample_add(p3, self.latlayer3(x2))
        p2 = self.enlayer2(p2)
        _pred = self.dslayer2(p2)
        preds.append(
            F.interpolate(_pred,
                          size=(H, W),
                          mode='bilinear', align_corners=False))

        p1 = self._upsample_add(p4, self.latlayer2(x1))
        p1 = self.enlayer2(p1)
        _pred = self.dslayer2(p1)
        preds.append(
            F.interpolate(_pred,
                          size=(H, W),
                          mode='bilinear', align_corners=False))

        return preds
