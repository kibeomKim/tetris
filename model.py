import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb
import numpy as np

class A3C(nn.Module):
    def __init__(self):
        super(A3C, self).__init__()

        self.conv1 = nn.Conv2d(1, 4, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(8, 8, kernel_size=4, stride=2)   #drop out or batchnorm

        self.fc = nn.Linear(512, 128)

        self.critic = nn.Linear(64, 1)
        self.actor = nn.Linear(64, 5)

    def forward(self, state, hx, cx):
        """
        :param state: [batch size, channel, width, height]
        :return:
        """
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        img_feat = x.view(x.size(0), -1)
        img_feat = F.relu(self.fc(img_feat))

        act, critic = torch.chunk(img_feat, 2, dim=1)

        return self.critic(critic), self.actor(act), hx, cx


class A3C_LSTM(nn.Module):
    def __init__(self):
        super(A3C_LSTM, self).__init__()

        self.conv1 = nn.Conv2d(4, 8, kernel_size=3, stride=1)
        # self.batchnorm1 = nn.BatchNorm2d(4, track_running_stats=False)
        self.conv2 = nn.Conv2d(8, 12, kernel_size=3, stride=1)
        # self.batchnorm2 = nn.BatchNorm2d(6, track_running_stats=False)
        self.conv3 = nn.Conv2d(12, 12, kernel_size=3, stride=1)
        # self.batchnorm3 = nn.BatchNorm2d(6, track_running_stats=False)

        self.conv_col = nn.Conv2d(4, 8, kernel_size=(1, 8), stride=1)   # kernel=(height,width)
        self.conv_row = nn.Conv2d(4, 8, kernel_size=(20, 1), stride=1)

        self.fc = nn.Linear(560, 512)

        self.lstm = nn.LSTMCell(512, 256)

        self.mlp = nn.Linear(256, 128)

        self.critic = nn.Linear(64, 1)
        self.actor = nn.Linear(64, 6)

    def forward(self, state, hx, cx):
        """
        :param state: [batch size, channel, width, height]
        :return:
        """
        # pdb.set_trace()
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # pdb.set_trace()
        img_feat = x.view(x.size(0), -1)    # [1, 336]

        col = F.relu(self.conv_col(state))
        col = col.view(col.size(0), -1)     # [1, 160]
        row = F.relu(self.conv_row(state))
        row = row.view(row.size(0), -1)     # [1, 64]

        feat = torch.cat([img_feat, col, row], dim=1)

        img_feat = F.leaky_relu(self.fc(feat))

        _hx, _cx = self.lstm(img_feat, (hx, cx))

        mlp_output = self.mlp(_hx)

        act, critic = torch.chunk(mlp_output, 2, dim=1)

        return self.critic(critic), self.actor(act), _hx, _cx