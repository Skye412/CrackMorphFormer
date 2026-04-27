import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__(
            nn.Conv2d(
                in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )


class ASPPPooling(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]
        y = self.gap(x)
        return F.interpolate(y, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch=256, atrous_rates=(6, 12, 18)):
        super().__init__()
        self.branch1 = ConvBNReLU(in_ch, out_ch, kernel_size=1, padding=0)

        self.branch2 = ConvBNReLU(
            in_ch, out_ch, kernel_size=3,
            padding=atrous_rates[0], dilation=atrous_rates[0]
        )
        self.branch3 = ConvBNReLU(
            in_ch, out_ch, kernel_size=3,
            padding=atrous_rates[1], dilation=atrous_rates[1]
        )
        self.branch4 = ConvBNReLU(
            in_ch, out_ch, kernel_size=3,
            padding=atrous_rates[2], dilation=atrous_rates[2]
        )
        self.branch5 = ASPPPooling(in_ch, out_ch)

        self.project = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        feats = [
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x),
            self.branch5(x)
        ]
        x = torch.cat(feats, dim=1)
        return self.project(x)


class DeepLabV3PlusBaseline(nn.Module):
    """
    ResNet-50 backbone + ASPP + low-level decoder
    output: 1-channel logits
    """
    def __init__(self, num_classes=1):
        super().__init__()

        backbone = resnet50(weights=None, replace_stride_with_dilation=[False, False, True])

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1   # low-level, 256 ch
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4   # high-level, 2048 ch

        self.aspp = ASPP(2048, 256, atrous_rates=(6, 12, 18))

        self.low_proj = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            ConvBNReLU(256 + 48, 256, kernel_size=3, padding=1),
            ConvBNReLU(256, 256, kernel_size=3, padding=1)
        )

        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[-2:]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        low = self.layer1(x)
        x = self.layer2(low)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.aspp(x)
        x = F.interpolate(x, size=low.shape[-2:], mode='bilinear', align_corners=False)

        low = self.low_proj(low)
        x = torch.cat([x, low], dim=1)
        x = self.decoder(x)
        x = self.classifier(x)

        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        return x