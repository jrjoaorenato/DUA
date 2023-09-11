from __future__ import absolute_import

import torch.nn as nn
import torch


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)

class UncertaintyLoss(nn.Module):
    def __init__(self):
        super(UncertaintyLoss, self).__init__()

    def forward(self, uncertainty_probs, predictions, ground_truth):
        # with(torch.no_grad()):
        prediction_prob_euclidian = torch.norm(predictions - ground_truth, 
                    dim=len(predictions.shape) - 1)/(torch.norm(predictions,
                    dim = len(predictions.shape)-1) + torch.norm(ground_truth, dim = len(ground_truth.shape)-1))

        unc_error = torch.mean(torch.abs(uncertainty_probs - prediction_prob_euclidian)) #do distance distribution instead of difference
        return unc_error          

class UncertaintyNetwork(nn.Module):
    def __init__(self, input_size, linear_size=1024, num_joints = 15, p_dropout=0.5, device = 'cuda:0'):
        super(UncertaintyNetwork, self).__init__()

        self.linear_size = linear_size
        self.input_size = input_size
        self.p_dropout = p_dropout
        self.num_joints = num_joints
        self.device = torch.device(device)
  
        self.joint_modules = []
        for l in range(num_joints):
            self.joint_modules.append(nn.Linear(self.linear_size, 1))
        self.joint_modules = nn.ModuleList(self.joint_modules)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = torch.Tensor([]).to(self.device)

        for l in range(self.num_joints):
            z = self.joint_modules[l](x)
            z = self.sigmoid(z)
            out = torch.cat((out, z), 1)

        return out
    def get_parameters(self, base_lr=1.0):
        params = []
        for param in self.parameters():
            params.append({"params": param})
        return params

class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y
        return out

class LinearModel(nn.Module):
    def __init__(self, input_size, output_size, linear_size=1024, num_stage=2, p_dropout=0.5):
        super(LinearModel, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # 2d joints
        self.input_size = input_size  # 16 * 2
        # 3d joints
        self.output_size = output_size  # 16 * 3


        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):
        # pre-processing
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        unc_out = y
        y = self.w2(y)  

        return y, unc_out