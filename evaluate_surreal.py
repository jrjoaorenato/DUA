from __future__ import print_function, absolute_import, division

import os
import time
import datetime
import argparse
import numpy as np
import os.path as path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from progress.bar import Bar
from common.log import Logger, savefig
from common.utils import AverageMeter, lr_decay, save_ckpt
from common.data_utils import fetch, read_3d_data, create_2d_data, fetch_surreal
from common.generators import PoseGenerator, PoseGeneratorTarget
from common.loss import mpjpe, p_mpjpe
from models.error_prob import LinearModel, UncertaintyNetwork, UncertaintyLoss, init_weights
from models.pose_converter import MapPoseConverter

from common.camera import world_to_camera, normalize_screen_coordinates, camera_to_world

from common.util.data import ForeverDataIterator

import wandb

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch training script')
    # General arguments
    parser.add_argument('-ds', '--dataset', default='surreal', type=str, metavar='NAME', help='source dataset')
    # parser.add_argument('-dt', '--dataset_target', default='surreal', type=str, metavar='NAME', help='target dataset')
    parser.add_argument('-k', '--keypoints', default='gt', type=str, metavar='NAME', help='2D detections to use')
    parser.add_argument('-a', '--actions', default='*', type=str, metavar='LIST',
                        help='actions to train/test on, separated by comma, or * for all')
    parser.add_argument('--evaluate', default='', type=str, metavar='FILENAME',
                        help='checkpoint to evaluate (file name)')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME',
                        help='checkpoint to resume (file name)')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('--snapshot', default=10, type=int, help='save models for every #snapshot epochs (default: 20)')
    parser.add_argument('-dv', '--device', default='cuda:0', type=str, metavar='NAME',
                        help='device being used (cuda:0, cuda:1, cpu)')

    # Model arguments
    parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N',
                        help='batch size in terms of predicted frames')
    parser.add_argument('-e', '--epochs', default=200, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('--num_workers', default=8, type=int, metavar='N', help='num of workers for data loading')
    parser.add_argument('--lr', default=1.0e-3, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_decay', type=int, default=100000, help='num of steps of learning rate decay')
    parser.add_argument('--lr_gamma', type=float, default=0.96, help='gamma of learning rate decay')
    parser.add_argument('--no_max', dest='max_norm', action='store_false', help='if use max_norm clip on grad')
    parser.set_defaults(max_norm=True)

    # Experimental
    parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR', help='downsample frame rate by factor')

    args = parser.parse_args()

    # Check invalid configuration
    if args.resume and args.evaluate:
        print('Invalid flags: --resume and --evaluate cannot be set at the same time')
        exit()

    return args


def main(args):
    print('==> Using settings {}'.format(args))

    print('==> Loading dataset...')

    if args.dataset == 'surreal':
        from common.surreal_dataset import SURREALDataset
        surreal_test_path = path.join('data', 'surreal_proc_3d_test.npy')
        dataset = SURREALDataset(surreal_test_path)

    stride = args.downsample
    cudnn.benchmark = True
    device = torch.device(args.device)

    # Create model
    print("==> Creating model...")
    num_joints = 16
    model_pos = LinearModel(num_joints * 2, (num_joints - 1) * 3).to(device)
    model_uncertainty = UncertaintyNetwork(1024, linear_size=1024, num_joints=num_joints-1, device=args.device).to(device)
    model_conversion = MapPoseConverter((num_joints - 1) * 3, 1024).to(device)
    model_pos.apply(init_weights)
    model_uncertainty.apply(init_weights)
    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model_pos.parameters()) / 1000000.0))
    criterion = nn.L1Loss(reduction='mean').to(device)
    optimizer = torch.optim.Adam(model_pos.parameters(), lr=args.lr)

    # Optionally resume from a checkpoint
    ckpt_path_conv = 'checkpoint/conversion_smpl_to_h36m/2023-01-17T17:14:52.076925/ckpt_best.pth.tar'
    ckpt_path = 'checkpoint/surreal_noda/2023-06-13T15:10:41.649064/ckpt_best.pth.tar'


    if path.isfile(ckpt_path):
        print("==> Loading checkpoint '{}'".format(ckpt_path))
        ckpt = torch.load(ckpt_path, map_location=device)
        start_epoch = ckpt['epoch']
        error_best = ckpt['error']
        glob_step = ckpt['step']
        lr_now = ckpt['lr']
        model_pos.load_state_dict(ckpt['state_dict'])
        model_conversion.load_state_dict(torch.load(ckpt_path_conv, map_location=device)['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print("==> Loaded checkpoint (Epoch: {} | Error: {})".format(start_epoch, error_best))
    else:
        raise RuntimeError("==> No checkpoint found at '{}'".format(ckpt_path))

    print('==> Evaluating...')

    poses_valid, poses_valid_2d, = fetch_surreal(dataset, stride)
    valid_loader = DataLoader(PoseGeneratorTarget(poses_valid, poses_valid_2d),
                                batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
    errors_p1, errors_p2 = evaluate(valid_loader, model_pos, model_conversion, device)

    print('Protocol #1   (MPJPE) action-wise average: {:.2f} (mm)'.format(np.mean(errors_p1).item()))
    print('Protocol #2 (P-MPJPE) action-wise average: {:.2f} (mm)'.format(np.mean(errors_p2).item()))
    exit(0)

def evaluate(data_loader, model_pos, model_conversion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()
    epoch_loss_3d_pos_procrustes = AverageMeter()

    # Switch to evaluate mode
    torch.set_grad_enabled(False)
    model_pos.eval()
    end = time.time()

    bar = Bar('Eval SURREAL', max=len(data_loader))
    for i, (targets_3d, inputs_2d) in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)

        inputs_2d = inputs_2d.to(device)
        targets_3d = targets_3d.to(device)
        outputs_3d, _ = model_pos(inputs_2d.view(num_poses, -1))
        outputs_3d = outputs_3d.view(num_poses, -1, 3).cpu()
        targets_3d_conv = model_conversion(targets_3d[:,1:,:].view(num_poses, -1)).view(num_poses, -1, 3).cpu()
        targets_3d_conv = torch.cat([torch.zeros(num_poses, 1, targets_3d_conv.size(2)), targets_3d_conv], 1)  # Pad hip joint (0,0,0)
        outputs_3d = torch.cat([torch.zeros(num_poses, 1, outputs_3d.size(2)), outputs_3d], 1)  # Pad hip joint (0,0,0)
        
        epoch_loss_3d_pos.update(mpjpe(outputs_3d, targets_3d_conv).item() * 1000.0, num_poses)
        epoch_loss_3d_pos_procrustes.update(p_mpjpe(outputs_3d.numpy(), targets_3d_conv.numpy()).item() * 1000.0, num_poses)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                     '| Tgt MPJPE: {et1: .4f}, Tgt P-MPJPE: {et2: .4f}' \
            .format(batch=i + 1, size=len(data_loader), data=data_time.avg, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td, et1 = epoch_loss_3d_pos.avg, et2 = epoch_loss_3d_pos_procrustes.avg)
        bar.next()

    bar.finish()

    print(outputs_3d[0])
    print(targets_3d_conv[0])
    print(targets_3d[0])
    return epoch_loss_3d_pos.avg, epoch_loss_3d_pos_procrustes.avg


if __name__ == '__main__':
    main(parse_args())
