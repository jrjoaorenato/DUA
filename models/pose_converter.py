import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.linear_model import Linear

class PoseConverter(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(PoseConverter, self).__init__()
        self.linear_size = linear_size
        self.p_dropout = p_dropout

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.in_proj = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.w2 = Linear(self.linear_size_size, self.p_dropout)

        self.out_proj = nn.Linear(self.linear_size, self.output_size)

    def forward(self, x):
        y = self.in_proj(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)

        y = self.out_proj(y)

        out = x + y

        return out

class ResPoseConverter(nn.Module):
    def __init__(self, data_size, linear_size = 1024, p_dropout=0.5):
        super(ResPoseConverter, self).__init__()
        self.linear_size = linear_size
        self.data_size = data_size
        self.p_dropout = p_dropout

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

        # self.w1 = nn.Linear(self.data_size, self.linear_size)
        self.w1 = nn.Linear(self.data_size, self.data_size)
        # self.batch_norm1 = nn.BatchNorm1d(self.linear_size)
        # self.w2 = nn.Linear(self.linear_size, self.data_size)

    def forward(self, x):
        y = self.w1(x)
        # y = self.batch_norm1(y)
        # y = self.relu(y)
        # y = self.dropout(y)

        # y = self.w2(y)

        out = x + y

        return out

class MapPoseConverter(nn.Module):
    def __init__(self, data_size, linear_size, p_dropout=0.5):
        super(MapPoseConverter, self).__init__()
        self.linear_size = linear_size
        self.data_size = data_size
        self.p_dropout = p_dropout
        self.joint_num = self.data_size // 3

        self.wx = nn.Linear(self.joint_num, self.joint_num)
        self.wy = nn.Linear(self.joint_num, self.joint_num)
        self.wz = nn.Linear(self.joint_num, self.joint_num)

    def forward(self, x):
        sep_x = x.view(x.shape[0], -1, 3)
        x_x = sep_x[:, :, 0]
        x_y = sep_x[:, :, 1]
        x_z = sep_x[:, :, 2]
        
        y_x = self.wx(x_x) 
        y_y = self.wy(x_y)
        y_z = self.wz(x_z)
        y = torch.cat((y_x.unsqueeze(2), y_y.unsqueeze(2), y_z.unsqueeze(2)), dim=2)
        y = y.view(x.shape[0], -1)

        out = x + y

        return out