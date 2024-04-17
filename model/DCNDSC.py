import torch
import torchvision.models
from torch import nn


class conv_block(nn.Module):
    def __init__(self, in_channel, out_channel, bn_size, drop_rate):
        super(conv_block, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_channel)
        self.Relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channel, out_channel * bn_size, kernel_size=1)

        self.bn2 = nn.BatchNorm1d(bn_size * out_channel)
        self.Relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(bn_size * out_channel, bn_size * out_channel, kernel_size=3, padding=1, groups=bn_size * out_channel)

        self.bn3 = nn.BatchNorm1d(bn_size * out_channel)
        self.Relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv1d(bn_size * out_channel, out_channel, kernel_size=1)

        self.dropout_rate = float(drop_rate)
        self.dropout = nn.Dropout(p=self.dropout_rate)
    def bn_funtion(self, X):
        # connect_feature = torch.cat(input, 1)
        feature_output = self.conv1(self.Relu1(self.bn1(X)))
        return feature_output

    def forward(self, X):
        feature_1x1_sec1 = self.bn_funtion(X)
        feature_3x3 = self.conv2(self.Relu2(self.bn2(feature_1x1_sec1)))
        feature_1x1_sec2 = self.conv3(self.Relu3(self.bn3(feature_3x3)))
        if self.dropout_rate > 0:
            feature_1x1_sec2 = self.dropout(feature_1x1_sec2)
        return feature_1x1_sec2


class Dense_Block(nn.Module):
    def __init__(self, num_convs, in_channel, out_channel, bn_size, drop_rate):
        super(Dense_Block, self).__init__()
        Dense_layer = []
        for i in range(num_convs):
            Dense_layer.append(conv_block(in_channel + i * out_channel, out_channel, bn_size, drop_rate))
        self.net = nn.Sequential(*Dense_layer)

    def forward(self, X):
        for block in self.net:
            Y = block(X)
            X = torch.cat((X, Y), dim=1)
        return X


class transition_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(transition_block, self).__init__()
        self.bn = nn.BatchNorm1d(in_channel)
        self.Relu = nn.ReLU()
        self.avg_pooling = nn.AvgPool1d(kernel_size=2, stride=2)
        self.conv = nn.Conv1d(in_channel, out_channel,kernel_size=1)

    def forward(self, X):
        Y = self.bn(X)
        Y = self.Relu(Y)
        Y = self.conv(Y)
        Y = self.avg_pooling(Y)

        return Y


class DCNDSC(nn.Module):
    def __init__(self, kernel_num1, kernel_num2, kernel_num3, kernel_num4, kernel_output=32):
        super(DCNDSC, self).__init__()
        self.b1 = nn.Sequential(nn.Conv1d(1, kernel_num1, kernel_size=7, stride=2, padding=3, bias=False),
                                nn.BatchNorm1d(kernel_num1),
                                nn.ReLU(inplace=True),
                                nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        self.b2 = nn.Sequential(Dense_Block(6, kernel_num1, kernel_output, 4, 0))
        self.tran1 = nn.Sequential(transition_block(256, 128))
        self.b3 = nn.Sequential(Dense_Block(12, kernel_num2, kernel_output, 4, 0))
        self.tran2 = nn.Sequential(transition_block(512, 256))
        self.b4 = nn.Sequential(Dense_Block(64, kernel_num3, kernel_output, 4, 0))
        self.tran3 = nn.Sequential(transition_block(2304, 1152))
        self.b5 = nn.Sequential(Dense_Block(48, kernel_num4, kernel_output, 4, 0))
        self.bn1 = nn.BatchNorm1d(2688)
        self.ReLU = nn.ReLU()
        self.adap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1*2688, 9)

    def forward(self, X):
        Y = self.b1(X)
        Y = self.b2(Y)
        Y = self.tran1(Y)
        Y = self.b3(Y)
        Y = self.tran2(Y)
        Y = self.b4(Y)
        Y = self.tran3(Y)
        Y = self.b5(Y)
        Y = self.bn1(Y)
        Y = self.ReLU(Y)
        Y = self.adap(Y)
        Y = self.flatten(Y)
        Y = self.fc(Y)

        return Y
