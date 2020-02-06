import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, input_channel, output_channel, stride):
        super(ResBlock, self).__init__()
        # conv1 with stride, kernel_size 3, pad with 1, kernel_size is not 1, different in and output channel;
        # conv2 without stride, kernel_size 3, pad with 1, with identical in and output channel;
        # conv3 with stride, kernel_size 1, different in and output channel
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=stride)

        self.bn1 = nn.BatchNorm2d(output_channel)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.bn3 = nn.BatchNorm2d(output_channel)

    def forward(self, x):
        temp = F.relu(self.bn1(self.conv1(x)))
        temp = self.bn2(self.conv2(temp))
        block_input = self.bn3(self.conv3(x))
        temp += block_input
        temp = F.relu(temp)
        return temp


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        # 3 input image channel RGB, 64 out channels, filter size 7, stride 2 square convolution
        # kernel
        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2)
        # 64 input 64 output
        self.bn = nn.BatchNorm2d(64)
        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2)
        self.ResBlock1 = ResBlock(input_channel=64, output_channel=64, stride=1)
        self.ResBlock2 = ResBlock(input_channel=64, output_channel=128, stride=2)
        self.ResBlock3 = ResBlock(input_channel=128, output_channel=256, stride=2)
        self.ResBlock4 = ResBlock(input_channel=256, output_channel=512, stride=2)
        # make output_size as (1, 1) as the global average
        self.GlobalAvgPool = nn.AdaptiveAvgPool2d((1, 1))
        # input channel is the output of last ResBlock, output channel is the feature number
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        x = self.max_pooling(F.relu(self.bn(self.conv(x))))

        x = self.ResBlock1(x)
        x = self.ResBlock2(x)
        x = self.ResBlock3(x)
        x = self.ResBlock4(x)

        x = self.GlobalAvgPool(x)
        x = x.view(x.size(0), -1)  # change output into (512, 1)
        x = self.fc(x)

        return x
