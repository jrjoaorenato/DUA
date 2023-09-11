from __future__ import absolute_import, division

import numpy as np
import copy
from common.skeleton import Skeleton
from common.mocap_dataset import MocapDataset
from common.camera import normalize_screen_coordinates

#todo check if its 8 or 7 before 10, 11, 12 and 13, 14, 15
surreal_skeleton = Skeleton(parents=[-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  8, 10, 11,  8, 13, 14],
                            joints_left=[4, 5, 6, 10, 11, 12],
                            joints_right=[1, 2, 3, 13, 14, 15])

surreal_skeleton_joints_group = [[2, 3], [5, 6], [1, 4], [0, 7], [8, 9], [14, 15], [11, 12], [10, 13]]
#                             [5, 8], [4, 7], [2, 8], [0, 3], [9, 15], [19, 21], [18, 20], [13, 14]
#                             [rightLeg, rightFoot], 
#                             [leftLeg, leftFoot], 
#                             [rightUpLeg, rightFoot], 
#                             [hips, spine], 
#                             [spine2, head], 
#                             [rightForeArm, rightHand],
#                             [leftForeArm, leftHand],
#                             [leftShoulder, rightShoulder]

surreal_info = {
    'name': 'SURREAL',
    'id': 'surreal',
    'res_w': 320,
    'res_h': 240,
    'center': [160.0, 120.0],
    'focal_length': [600.0, 600.0],
    'radial_distortion': [0., 0., 0.],
    'tangential_distortion': [0., 0.],
}

used_joints = [0, 2, 5, 8, 1, 4, 7,  9, 12, 15, 13, 18, 20, 14, 19, 21]

class SURREALDataset(MocapDataset):
    def __init__(self, path, remove_static_joints=True):
        super(SURREALDataset, self).__init__(skeleton=surreal_skeleton, fps=30)

        for k, v in surreal_info.items():
            if k not in ['id', 'res_w', 'res_h', 'name']:
                surreal_info[k] = np.array(v, dtype='float32')

        # Normalize camera frame
        surreal_info['center'] = normalize_screen_coordinates(surreal_info['center'], w=surreal_info['res_w'], 
            h=surreal_info['res_h']).astype('float32')
        surreal_info['focal_length'] = surreal_info['focal_length'] / surreal_info['res_w'] * 2.0
        if 'translation' in surreal_info:
            surreal_info['translation'] = surreal_info['translation'] / 1000  # mm to meters

        # Add intrinsic parameters vector
        surreal_info['intrinsic'] = [60, 160, 120, 0, 0, 0, 0, 0]
                                            # np.concatenate((np.expand_dims(surreal_info['focal_length'], 0),
                                            # np.expand_dims(surreal_info['center'], 0),
                                            # np.expand_dims(surreal_info['radial_distortion'], 0),
                                            # np.expand_dims(surreal_info['tangential_distortion'], 0) ))

        # Load serialized dataset surreal loading
        data = np.load(path, allow_pickle = True).item()['3d_joints'][:, used_joints, :]
        data_2d = np.load(path, allow_pickle = True).item()['2d_joints'][:, used_joints, :]
        # orig_annot = np.load(path, allow_pickle = True).item()['orig_annot']
        # camLoc = []
        # camDist = []
        # for i in range(len(orig_annot)):
        #     for j in range(orig_annot[i]['joints3D'].shape[2]):
        #         camLoc.append(orig_annot[i]['camLoc'])
        #         camDist.append(orig_annot[i]['camDist'])
        # camLoc = np.array(camLoc).squeeze()
        # camDist = np.array(camDist).squeeze()
        
        #data = np.load(path, allow_pickle=True)['positions_3d'].item()

        self._data = {}
        self._data['positions_3d'] = data
        self._data['positions'] = data_2d
        self._data['camera'] = surreal_info
        # self._data['camLoc'] = camLoc
        # self._data['camDist'] = camDist

        # if remove_static_joints:
        #     # Bring the skeleton to 16 joints instead of the original 32
        #     joints = []
        #     for i, x in enumerate(H36M_NAMES):
        #         if x == '' or x == 'Neck/Nose':  # Remove 'Nose' to make SH and H36M 2D poses have the same dimension
        #             joints.append(i)
        #     self.remove_joints(joints)

            # Rewire shoulders to the correct parents
            # self._skeleton._parents[10] = 8
            # self._skeleton._parents[13] = 8

            # Set joints group
        self._skeleton._joints_group = surreal_skeleton_joints_group

    def define_actions(self, action=None):
        return False
