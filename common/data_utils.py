from __future__ import absolute_import, division

import numpy as np

from .camera import world_to_camera, normalize_screen_coordinates

def get_extrinsic(T):
    # Take the first 3 columns of the matrix_world in Blender and transpose.
    # This is hard-coded since all images in SURREAL use the same.
    R_world2bcam = np.array([[0, 0, 1], [0, -1, 0], [-1, 0, 0]]).transpose()
    # *cam_ob.matrix_world = Matrix(((0., 0., 1, params['camera_distance']),
    #                               (0., -1, 0., -1.0),
    #                               (-1., 0., 0., 0.),
    #                               (0.0, 0.0, 0.0, 1.0)))

    # Convert camera location to translation vector used in coordinate changes
    T_world2bcam = -1 * np.dot(R_world2bcam, T)

    # Following is needed to convert Blender camera to computer vision camera
    R_bcam2cv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = np.dot(R_bcam2cv, R_world2bcam)
    T_world2cv = np.dot(R_bcam2cv, T_world2bcam)

    print('R_world2cv', R_world2cv)
    print('T_world2cv', T_world2cv)

    # Put into 3x4 matrix
    RT = np.concatenate([R_world2cv, T_world2cv], axis=1)
    return RT, R_world2cv, T_world2cv


def create_2d_data(data_path, dataset):
    keypoints = np.load(data_path, allow_pickle=True)
    keypoints = keypoints['positions_2d'].item()

    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                # Normalize camera frame
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps

    return keypoints


def read_3d_data(dataset):
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]

            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_3d[:, :] -= pos_3d[:, :1]  # Remove global offset
                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d

    return dataset

def fetch_surreal(dataset, stride = 1, parse_3d_poses = True):
    #have to standardize the data to be in the same format as the h36m data
    cam = dataset['camera']
    # camLoc = dataset['camLoc']

    # assert len(camLoc) == len(dataset['positions_3d']), 'Camera count mismatch'

    out_poses_3d = dataset['positions_3d']#/1000
    out_poses_2d = normalize_screen_coordinates(dataset['positions'][..., :2], w=cam['res_w'], h=cam['res_h'])
    out_poses_3d -= out_poses_3d[:, :1, :]

    # R_world2bcam = np.array([[0, 0, 1], [0, -1, 0], [-1, 0, 0]]).transpose()
    # R_bcam2cv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    # R = np.dot(R_bcam2cv, R_world2bcam)




    # for i in range(len(out_poses_3d)):
    #     T_world2bcam = -1 * np.dot(R_world2bcam, camLoc[i])
    #     T = np.dot(R_bcam2cv, T_world2bcam)
    #     out_poses_3d[i] = world_to_camera(out_poses_3d[i], R=R, t=T)
        
    # for i in range(len(dataset.data)):
    #     if parse_3d_poses:
    #         out_poses_3d.append(dataset[i]['positions_3d'][::stride])
    #     out_poses_2d.append(dataset[i]['positions'][::stride])

    # if len(out_poses_3d) == 0:
    #     out_poses_3d = None

    return out_poses_3d, out_poses_2d

def fetch(subjects, dataset, keypoints, action_filter=None, stride=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_actions = []

    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    # if action.startswith(a):
                    if action.split(' ')[0] == a:
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])
                out_actions.append([action.split(' ')[0]] * poses_2d[i].shape[0])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)):  # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    if len(out_poses_3d) == 0:
        out_poses_3d = None

    if stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            out_actions[i] = out_actions[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_poses_3d, out_poses_2d, out_actions
