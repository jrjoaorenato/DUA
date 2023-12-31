from __future__ import print_function, absolute_import

import numpy as np
import torch
from torch.utils.data import Dataset
from functools import reduce


class PoseGenerator(Dataset):
    def __init__(self, poses_3d, poses_2d, actions):
        assert poses_3d is not None

        self._poses_3d = np.concatenate(poses_3d)
        self._poses_2d = np.concatenate(poses_2d)
        self._actions = reduce(lambda x, y: x + y, actions)

        assert self._poses_3d.shape[0] == self._poses_2d.shape[0] and self._poses_3d.shape[0] == len(self._actions)
        print('Generating {} poses...'.format(len(self._actions)))

    def __getitem__(self, index):
        out_pose_3d = self._poses_3d[index]
        out_pose_2d = self._poses_2d[index]
        out_action = self._actions[index]

        out_pose_3d = torch.from_numpy(out_pose_3d).float()
        out_pose_2d = torch.from_numpy(out_pose_2d).float()

        return out_pose_3d, out_pose_2d, out_action

    def __len__(self):
        return len(self._actions)

class PoseGeneratorSMPL(Dataset):
    def __init__(self, poses_3d, poses_spml, actions):
        assert poses_3d is not None

        self._poses_3d = np.concatenate(poses_3d)
        self._poses_smpl = np.concatenate(poses_spml)
        self._actions = reduce(lambda x, y: x + y, actions)

        assert self._poses_3d.shape[0] == self._poses_smpl.shape[0] and self._poses_3d.shape[0] == len(self._actions)
        print('Generating {} poses...'.format(len(self._actions)))

    def __getitem__(self, index):
        out_pose_3d = self._poses_3d[index]
        out_pose_smpl = self._poses_smpl[index]
        out_action = self._actions[index]

        out_pose_3d = torch.from_numpy(out_pose_3d).float()
        out_pose_smpl = torch.from_numpy(out_pose_smpl).float()

        return out_pose_3d, out_pose_smpl, out_action

    def __len__(self):
        return len(self._actions)

class PoseGeneratorTarget(Dataset):
    def __init__(self, poses_3d, poses_2d):
        assert poses_3d is not None

        self._poses_3d = poses_3d#np.concatenate(poses_3d, axis = 0)
        self._poses_2d = poses_2d#np.concatenate(poses_2d, axis = 0)

        assert self._poses_3d.shape[0] == self._poses_2d.shape[0]
        print('Generating {} poses...'.format(self._poses_2d.shape[0]))

    def __getitem__(self, index):
        out_pose_3d = self._poses_3d[index]
        out_pose_2d = self._poses_2d[index]

        out_pose_3d = torch.from_numpy(out_pose_3d).float()
        out_pose_2d = torch.from_numpy(out_pose_2d).float()

        return out_pose_3d, out_pose_2d

    def __len__(self):
        return self._poses_2d.shape[0]
