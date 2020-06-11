#-*-coding:utf-8-*-
'''
@FileName:
    Network.py
@Description:
    Network for C3D
@Authors:
    Hanbo Sun(sun-hb17@mails.tsinghua.edu.cn)
@CreateTime:
    2020/06/11 23:49
'''

import torch
import torch.nn as nn

class C3DNetwork(nn.Module):
    '''
    C3D Network with BatchNorm3d
    '''
    def __init__(self, num_classes = 101):
        super(C3DNetwork, self).__init__()
        # part1
        self.conv1 = nn.Conv3d(3, 64, kernel_size = (3, 3, 3), padding = (1, 1, 1))
        self.bn1 = nn.BatchNorm3d(64)
        self.relu1 = nn.ReLU(inplace = True)
        self.pool1 = nn.MaxPool3d(kernel_size = (1, 2, 2), stride = (1, 2, 2))
        # part2
        self.conv2 = nn.Conv3d(64, 128, kernel_size = (3, 3, 3), padding = (1, 1, 1))
        self.bn2 = nn.BatchNorm3d(128)
        self.relu2 = nn.ReLU(inplace = True)
        self.pool2 = nn.MaxPool3d(kernel_size = (2, 2, 2), stride = (2, 2, 2))
        # part3
        self.conv3a = nn.Conv3d(128, 256, kernel_size = (3, 3, 3), padding = (1, 1, 1))
        self.bn3a = nn.BatchNorm3d(256)
        self.relu3a = nn.ReLU(inplace = True)
        self.conv3b = nn.Conv3d(256, 256, kernel_size = (3, 3, 3), padding = (1, 1, 1))
        self.bn3b = nn.BatchNorm3d(256)
        self.relu3b = nn.ReLU(inplace = True)
        self.pool3 = nn.MaxPool3d(kernel_size = (2, 2, 2), stride = (2, 2, 2))
        # part4
        self.conv4a = nn.Conv3d(256, 512, kernel_size = (3, 3, 3), padding = (1, 1, 1))
        self.bn4a = nn.BatchNorm3d(512)
        self.relu4a = nn.ReLU(inplace = True)
        self.conv4b = nn.Conv3d(512, 512, kernel_size = (3, 3, 3), padding = (1, 1, 1))
        self.bn4b = nn.BatchNorm3d(512)
        self.relu4b = nn.ReLU(inplace = True)
        self.pool4 = nn.MaxPool3d(kernel_size = (2, 2, 2), stride = (2, 2, 2))
        # part5
        self.conv5a = nn.Conv3d(512, 512, kernel_size = (3, 3, 3), padding = (1, 1, 1))
        self.bn5a = nn.BatchNorm3d(512)
        self.relu5a = nn.ReLU(inplace = True)
        self.conv5b = nn.Conv3d(512, 512, kernel_size = (3, 3, 3), padding = (1, 1, 1))
        self.bn5b = nn.BatchNorm3d(512)
        self.relu5b = nn.ReLU(inplace = True)
        self.pool5 = nn.MaxPool3d(kernel_size = (2, 2, 2), stride = (2, 2, 2), padding = (0, 1, 1))
        # fc
        self.fc6 = nn.Linear(8192, 4096)
        self.dropout6 = nn.Dropout(p = 0.5)
        self.relu6 = nn.ReLU(inplace = True)
        self.fc7 = nn.Linear(4096, num_classes)
    def forward(self, x):
        # part1
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        # part2
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        # part3
        x = self.relu3a(self.bn3a(self.conv3a(x)))
        x = self.relu3b(self.bn3b(self.conv3b(x)))
        x = self.pool3(x)
        # part4
        x = self.relu4a(self.bn4a(self.conv4a(x)))
        x = self.relu4b(self.bn4b(self.conv4b(x)))
        x = self.pool4(x)
        # part5
        x = self.relu5a(self.bn5a(self.conv5a(x)))
        x = self.relu5b(self.bn5b(self.conv5b(x)))
        x = self.pool5(x)
        # fc
        x = x.view(-1, 8192)
        x = self.relu6(self.dropout6(self.fc6(x)))
        x = self.fc7(x)
        return x

if __name__ == '__main__':
    net = C3DNetwork(101)
    print(net)
    inputs = torch.rand(1, 3, 16, 112, 112)
    outputs = net(inputs)
    print(outputs.size())