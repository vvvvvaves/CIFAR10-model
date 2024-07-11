from torch import nn
import torch


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3),
                               padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        torch.nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        torch.nn.init.constant_(self.bn1.weight, 0.5)
        torch.nn.init.zeros_(self.bn1.bias)

    def forward(self, x):
        tmp = self.conv1(x)
        tmp = self.bn1(tmp)
        out = self.relu(tmp)

        return out


class BottleNeck(nn.Module):
    def __init__(self, out_channels, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.cb1 = ConvBlock(out_channels, out_channels // 2)
        self.cb2 = ConvBlock(out_channels // 2, out_channels)

    def forward(self, x):
        tmp = self.cb1(x)
        out = self.cb2(tmp)

        if self.shortcut:
            return out + x
        else:
            return out


class FCBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()

        torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        torch.nn.init.constant_(self.bn1.weight, 0.5)
        torch.nn.init.zeros_(self.bn1.bias)

    def forward(self, x):
        tmp = self.fc1(x)
        tmp = self.bn1(tmp)
        out = self.relu(tmp)

        return out


class Net(nn.Module):
    IMAGE_SIZE = (32, 32)

    def __init__(self, n_channels1=32, n_classes=10):
        super().__init__()
        self.n_channels1 = n_channels1

        self.cb1 = ConvBlock(in_channels=3, out_channels=n_channels1)  # 3*32*32 -> 32*32*32
        self.cb2 = ConvBlock(in_channels=n_channels1, out_channels=n_channels1*2)  # 32*32*32 -> 64*32*32
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2)) # 64x32x32 -> 64x16x16

        self.cb3 = ConvBlock(in_channels=n_channels1*2, out_channels=n_channels1*2*2)  # 64x16x16 -> 128x16x16
        self.cb4 = ConvBlock(in_channels=n_channels1*2*2, out_channels=n_channels1*2*2) # 128x16x16 -> 128x16x16
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))  # 128x16x16 -> 128x8x8

        self.cb5 = ConvBlock(in_channels=n_channels1*2*2, out_channels=n_channels1*2*2*2)  # 128x8x8 -> 256x8x8
        self.cb6 = ConvBlock(in_channels=n_channels1*2*2*2, out_channels=n_channels1*2*2*2) # 256x8x8 -> 256x8x8
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2)) # -> 256x4x4

        self.fcb1 = FCBlock(256*4*4, 1024)
        self.fcb2 = FCBlock(1024, 512)
        self.fc1 = nn.Linear(512, 10)

        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        tmp = self.cb1(x)
        tmp = self.cb2(tmp)
        tmp = self.pool1(tmp)

        tmp = self.cb3(tmp)
        tmp = self.cb4(tmp)
        tmp = self.pool2(tmp)

        tmp = self.cb5(tmp)
        tmp = self.cb6(tmp)
        tmp = self.pool3(tmp)

        tmp = tmp.view(-1, self.n_channels1 * 2 * 2 * 2 * 4 * 4)

        tmp = self.fcb1(tmp)

        tmp = self.fcb2(tmp)

        tmp = self.fc1(tmp)
        out = self.logSoftmax(tmp)

        return out
