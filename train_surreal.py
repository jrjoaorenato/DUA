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
    parser.add_argument('-c', '--checkpoint', default='checkpoint/surreal_noda/', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('--snapshot', default=10, type=int, help='save models for every #snapshot epochs (default: 20)')
    parser.add_argument('-dv', '--device', default='cuda:0', type=str, metavar='NAME',
                        help='device being used (cuda:0, cuda:1, cpu)')
    parser.add_argument('-b', '--batch_size', default=2048, type=int, metavar='N',
                        help='batch size in terms of predicted frames')
    parser.add_argument('-e', '--epochs', default=200, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('--num_workers', default=8, type=int, metavar='N', help='num of workers for data loading')
    parser.add_argument('--lr', default=1.0e-3, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_decay', type=int, default=3125, help='num of steps of learning rate decay')
    # parser.add_argument('--lr_decay', type=int, default=100000, help='num of steps of learning rate decay')
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
        # surreal_test_path = path.join('data', 'surreal_proc_3d_test.npy')
        surreal_test_path = path.join('data', 'surreal_complete_parsed_test.npy')
        surreal_train_path = path.join('data', 'surreal_complete_parsed_train.npy')
        dataset = SURREALDataset(surreal_test_path)
        dataset_train = SURREALDataset(surreal_train_path)

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

    # criterion = nn.MSELoss(reduction='mean').to(device)
    criterion = nn.L1Loss(reduction='mean').to(device)
    optimizer = torch.optim.Adam( 
            [
                {'params': model_pos.parameters()},
                {'params': model_uncertainty.parameters()}
            ], lr=args.lr)  

    ckpt_path_conv = 'checkpoint/conversion/2023-01-17T17:14:52.076925/ckpt_best.pth.tar'
    model_conversion.load_state_dict(torch.load(ckpt_path_conv, map_location=device)['state_dict'])
    
    poses_train, poses_train_2d, = fetch_surreal(dataset_train, stride)
    train_loader = DataLoader(PoseGeneratorTarget(poses_train, poses_train_2d),
                                batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, pin_memory=True)
    poses_valid, poses_valid_2d, = fetch_surreal(dataset, stride)
    valid_loader = DataLoader(PoseGeneratorTarget(poses_valid, poses_valid_2d),
                                batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)

    ckpt_dir_path = path.join(args.checkpoint, datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f'))
    if not path.exists(ckpt_dir_path):
        os.makedirs(ckpt_dir_path)
        print('==> Making checkpoint dir: {}'.format(ckpt_dir_path))
        
    start_epoch = 0
    error_best = None
    glob_step = 0
    lr_now = args.lr

    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr_now))

        # Train for one epoch
        epoch_loss, epoch_loss_uncertainty, lr_now, glob_step = train(train_loader, model_pos, model_conversion,
                                              model_uncertainty, criterion, optimizer, device, args.lr, lr_now,
                                              epoch, glob_step, args.lr_decay, args.lr_gamma, max_norm=args.max_norm)

        # Evaluate
        error_eval_p1, error_eval_p2 = evaluate(valid_loader, model_pos, model_conversion, device)

        # wandb.watch(model_pos, log="all", log_freq=100)

        # Save checkpoint
        if error_best is None or error_best > error_eval_p1:
            error_best = error_eval_p1
            save_ckpt({'epoch': epoch + 1, 'lr': lr_now, 'step': glob_step, 'state_dict': model_pos.state_dict(),
                       'optimizer': optimizer.state_dict(), 'error': error_eval_p1}, ckpt_dir_path, suffix='best')

        if (epoch + 1) % args.snapshot == 0:
            save_ckpt({'epoch': epoch + 1, 'lr': lr_now, 'step': glob_step, 'state_dict': model_pos.state_dict(),
                       'optimizer': optimizer.state_dict(), 'error': error_eval_p1}, ckpt_dir_path)

    exit(0)

def train(data_loader, model_pos, model_conv, model_uncertainty, criterion, optimizer, device, lr_init, lr_now, epoch, step, decay, gamma, max_norm=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()
    epoch_loss_uncertainty = AverageMeter()

    # Switch to train mode
    torch.set_grad_enabled(True)
    criterion_unc = UncertaintyLoss()
    model_pos.train()
    model_conv.eval()
    end = time.time()

    bar = Bar('Train', max=len(data_loader))
    for i, (targets_3d, inputs_2d) in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)

        step += 1
        if step % decay == 0 or step == 1:
            lr_now = lr_decay(optimizer, step, lr_init, decay, gamma)

        orig_targets_3d, inputs_2d = targets_3d[:, 1:, :].to(device), inputs_2d.to(device)  # Remove hip joint for 3D poses
        target_3d = model_conv(orig_targets_3d.view(num_poses, -1)).view(num_poses, -1, 3)
        outputs_3d, unc_inputs = model_pos(inputs_2d.view(num_poses, -1))
        outputs_3d = outputs_3d.view(num_poses, -1, 3)
        
        uncertainty_values = model_uncertainty(unc_inputs)

        optimizer.zero_grad()
        loss_3d_pos = criterion(outputs_3d, target_3d)
        unc_error = criterion_unc(uncertainty_values, outputs_3d, target_3d)
        total_loss = loss_3d_pos + 0.1*unc_error

        total_loss.backward()
        if max_norm:
            nn.utils.clip_grad_norm_(model_pos.parameters(), max_norm=1)
        optimizer.step()

        epoch_loss_3d_pos.update(loss_3d_pos.item(), num_poses)
        epoch_loss_uncertainty.update(unc_error.item(), num_poses)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                     '| Loss: {loss: .4f} | UncLoss: {uncloss: .4f}' \
            .format(batch=i + 1, size=len(data_loader), data=data_time.avg, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td, loss=epoch_loss_3d_pos.avg, uncloss = epoch_loss_uncertainty.avg)
        bar.next()

    bar.finish()
    return epoch_loss_3d_pos.avg, epoch_loss_uncertainty.avg, lr_now, step

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
        # outputs_3d = model_conversion(outputs_3d).view(num_poses, -1, 3).cpu()
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
    return epoch_loss_3d_pos.avg, epoch_loss_3d_pos_procrustes.avg


if __name__ == '__main__':
    main(parse_args())
