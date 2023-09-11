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
from common.generators import PoseGenerator, PoseGeneratorTarget, PoseGeneratorSMPL
from common.loss import mpjpe, p_mpjpe, pjpe, p_pjpe
from models.error_prob import LinearModel, UncertaintyNetwork, UncertaintyLoss, init_weights
from models.pose_converter import ResPoseConverter, MapPoseConverter

from common.util.data import ForeverDataIterator

import wandb

JOINT_LIST = ['Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 
              'Spine', 'Thorax', 'Head', 'LShoulder', 'LElbow', 'LWrist', 
              'RShoulder', 'RElbow', 'RWrist']

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch training script')
    # General arguments
    parser.add_argument('-ds', '--dataset_source', default='h36m', type=str, metavar='NAME', help='source dataset')
    parser.add_argument('-dt', '--dataset_target', default='surreal', type=str, metavar='NAME', help='target dataset')
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
    parser.add_argument('-f', '--format', default='h36m', type=str, metavar='NAME', help='output pose representation format')

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
    dataset_path = path.join('data', 'data_3d_' + args.dataset_source + '.npz')
    smpl_path = path.join('data', 'data_3d_h36m_smpl.npz')

    if args.dataset_source == 'h36m':
        from common.h36m_dataset import Human36mDataset, TRAIN_SUBJECTS, TEST_SUBJECTS
        from common.smpl_dataset import SMPLDataset
        dataset = Human36mDataset(dataset_path)
        dataset_smpl = SMPLDataset(smpl_path)
        subjects_train = TRAIN_SUBJECTS
        subjects_test = TEST_SUBJECTS

    print('==> Preparing data...')
    dataset = read_3d_data(dataset)
    dataset_smpl = read_3d_data(dataset_smpl)

    print('==> Loading 2D detections...')
    keypoints = create_2d_data(path.join('data', 'data_2d_' + args.dataset_source + '_' + args.keypoints + '.npz'), dataset)
    keypoints_smpl = create_2d_data(path.join('data', 'data_2d_h36m_smpl.npz'), dataset_smpl)

    action_filter = None if args.actions == '*' else args.actions.split(',')
    if action_filter is not None:
        action_filter = map(lambda x: dataset.define_actions(x)[0], action_filter)
        print('==> Selected actions: {}'.format(action_filter))

    stride = args.downsample
    cudnn.benchmark = True
    device = torch.device(args.device)

    # Create model
    print("==> Creating model...")
    num_joints = dataset.skeleton().num_joints()
    model_pos = ResPoseConverter((num_joints - 1) * 3, 1024).to(device)
    model_pos.apply(init_weights)
    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model_pos.parameters()) / 1000000.0))

    criterion = nn.MSELoss(reduction='mean').to(device)
    optimizer = torch.optim.Adam(model_pos.parameters(), lr=args.lr)

    # Optionally resume from a checkpoint
    if args.resume or args.evaluate:
        ckpt_path = (args.resume if args.resume else args.evaluate)

        if path.isfile(ckpt_path):
            print("==> Loading checkpoint '{}'".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location=device)
            start_epoch = ckpt['epoch']
            error_best = ckpt['error']
            glob_step = ckpt['step']
            lr_now = ckpt['lr']
            model_pos.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
            print("==> Loaded checkpoint (Epoch: {} | Error: {})".format(start_epoch, error_best))

            if args.resume:
                ckpt_dir_path = path.dirname(ckpt_path)
                logger = Logger(path.join(ckpt_dir_path, 'log.txt'), resume=True)
        else:
            raise RuntimeError("==> No checkpoint found at '{}'".format(ckpt_path))
    else:
        start_epoch = 0
        error_best = None
        glob_step = 0
        lr_now = args.lr
        ckpt_dir_path = path.join(args.checkpoint, datetime.datetime.now().isoformat())

        if not path.exists(ckpt_dir_path):
            os.makedirs(ckpt_dir_path)
            print('==> Making checkpoint dir: {}'.format(ckpt_dir_path))

        logger = Logger(os.path.join(ckpt_dir_path, 'log.txt'))
        logger.set_names(['epoch', 'lr', 'loss_train', 'error_eval_p1', 'error_eval_p2'])

    if args.evaluate:
        print('==> Evaluating Predictions...')

        if action_filter is None:
            action_filter = dataset.define_actions()

        errors_p1 = np.zeros(len(action_filter))
        errors_p2 = np.zeros(len(action_filter))
        errors_p1_smpl = np.zeros(len(action_filter))
        errors_p2_smpl = np.zeros(len(action_filter))

        for i, action in enumerate(action_filter):
            poses_valid, poses_valid_2d, actions_valid = fetch(['S9'], dataset, keypoints, [action], stride)
            poses_valid_smpl, poses_valid_2d_smpl, actions_valid_smpl = fetch(['S9'], dataset_smpl, keypoints_smpl, [action], stride)
            valid_loader = DataLoader(PoseGenerator(poses_valid, poses_valid_2d, actions_valid),
                                      batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers, pin_memory=True)
            smpl_loader = DataLoader(PoseGeneratorSMPL(poses_valid_smpl, poses_valid_2d_smpl, actions_valid_smpl), batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=False, drop_last=False, persistent_workers=False)
            errors_p1[i], errors_p2[i], errors_p1_smpl[i], errors_p2_smpl[i] = evaluate(valid_loader, smpl_loader, args.format, model_pos, device)

        print('Protocol #1   (MPJPE) action-wise average: {:.2f} (mm)'.format(np.mean(errors_p1).item()))
        print('Protocol #2 (P-MPJPE) action-wise average: {:.2f} (mm)'.format(np.mean(errors_p2).item()))
        print('Protocol #1   (SMPL) action-wise average: {:.2f} (mm)'.format(np.mean(errors_p1_smpl).item()))
        print('Protocol #2   (SMPL) action-wise average: {:.2f} (mm)'.format(np.mean(errors_p2_smpl).item()))

        print('')
        print('==> Evaluating Per Joint...')

        poses_valid, poses_valid_2d, actions_valid = fetch(['S9'], dataset, keypoints, None, stride)
        poses_valid_smpl, poses_valid_2d_smpl, actions_valid_smpl = fetch(['S9'], dataset_smpl, keypoints_smpl, None, stride)
        valid_loader = DataLoader(PoseGenerator(poses_valid, poses_valid_2d, actions_valid),
                                batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
        smpl_loader = DataLoader(PoseGeneratorSMPL(poses_valid_smpl, poses_valid_2d_smpl, actions_valid_smpl), batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers, pin_memory=False, drop_last=False, persistent_workers=False)
        joint_error_pred_p1, joint_error_pred_p2, joint_error_smpl_p1, joint_error_smpl_p2 = evaluate_per_joint(valid_loader, smpl_loader, args.format, model_pos, device)

        for i in range(16):
            print('{}: MPJPE Pred: {:.2f} mm | P-MPJPE Pred {:.2f} mm | MPJPE SMPL {:.2f} mm | P-MPJPE SMPL {:.2f} mm'.format(JOINT_LIST[i], 
                  joint_error_pred_p1[i], joint_error_pred_p2[i], joint_error_smpl_p1[i], joint_error_smpl_p2[i]))

        exit(0)

    poses_train, poses_train_2d, actions_train = fetch(subjects_train, dataset, keypoints, action_filter, stride)

    poses_valid, poses_valid_2d, actions_valid = fetch(subjects_test, dataset, keypoints, action_filter, stride)
    valid_loader = DataLoader(PoseGenerator(poses_valid, poses_valid_2d, actions_valid), batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=False, drop_last=True, persistent_workers=False)

    poses_train_smpl, poses_train_2d_smpl, actions_train = fetch(subjects_train, dataset_smpl, keypoints_smpl, action_filter, stride)
    smpl_loader = DataLoader(PoseGeneratorSMPL(poses_train, poses_train_smpl, actions_train), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=False, persistent_workers=False)

    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr_now))

        # Train for one epoch
        epoch_loss, lr_now, glob_step = train(smpl_loader, model_pos, args.format, criterion, optimizer, device, args.lr, lr_now,
                                              epoch, glob_step, args.lr_decay, args.lr_gamma, max_norm=args.max_norm)

        # Save checkpoint
        if error_best is None or error_best > epoch_loss:
            error_best = epoch_loss
            save_ckpt({'epoch': epoch + 1, 'lr': lr_now, 'step': glob_step, 'state_dict': model_pos.state_dict(),
                       'optimizer': optimizer.state_dict(), 'error': epoch_loss}, ckpt_dir_path, suffix='best')

        if (epoch + 1) % args.snapshot == 0:
            save_ckpt({'epoch': epoch + 1, 'lr': lr_now, 'step': glob_step, 'state_dict': model_pos.state_dict(),
                       'optimizer': optimizer.state_dict(), 'error': epoch_loss}, ckpt_dir_path)

    logger.close()
    logger.plot(['loss_train', 'error_eval_p1'])
    savefig(path.join(ckpt_dir_path, 'log.eps'))

    return


def train(data_loader, model_pos, format, criterion, optimizer, device, lr_init, lr_now, epoch, step, decay, gamma, max_norm=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()

    # Switch to train mode
    torch.set_grad_enabled(True)
    model_pos.train()
    end = time.time()
    criterionL1 = nn.L1Loss(reduction='mean').to(device)
    lamda = 0.1

    format_H36M = True if format == 'h36m' else False

    bar = Bar('Train', max=len(data_loader))
    for i, (targets_3d, smpl_3d, _ ) in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)

        step += 1
        if step % decay == 0 or step == 1:
            lr_now = lr_decay(optimizer, step, lr_init, decay, gamma)
            
        targets_3d = targets_3d[:, 1:, :].to(device)#, inputs_2d.to(device)  # Remove hip joint for 3D poses
        smpl_3d = smpl_3d[:, 1:, :].to(device)

        if format_H36M:
             #input smpl and output h36m
            outputs_3d = model_pos(smpl_3d.view(num_poses, -1))
        else:
            #input h36m and output smpl
            outputs_3d = model_pos(targets_3d.view(num_poses, -1))

        outputs_3d = outputs_3d.view(num_poses, -1, 3)


        optimizer.zero_grad() 
        if format_H36M:
            loss_3d_pos = (1-lamda)*criterion(outputs_3d, targets_3d) + lamda*criterionL1(outputs_3d, targets_3d)
        else:
            loss_3d_pos = (1-lamda)*criterion(outputs_3d, smpl_3d) + lamda*criterionL1(outputs_3d, smpl_3d)

        total_loss = loss_3d_pos

        total_loss.backward()
        if max_norm:
            nn.utils.clip_grad_norm_(model_pos.parameters(), max_norm=1)
        optimizer.step()

        epoch_loss_3d_pos.update(loss_3d_pos.item(), num_poses)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                     '| Loss: {loss: .4f}' \
            .format(batch=i + 1, size=len(data_loader), data=data_time.avg, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td, loss=epoch_loss_3d_pos.avg)
        bar.next()

    bar.finish()
    return epoch_loss_3d_pos.avg, lr_now, step


def evaluate(data_loader, smpl_loader, format, model_pos, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()
    epoch_loss_3d_smpl = AverageMeter()
    epoch_loss_3d_pos_procrustes = AverageMeter()
    epoch_loss_3d_procrustes_smpl = AverageMeter()

    # Switch to evaluate mode
    torch.set_grad_enabled(False)
    model_pos.eval()
    end = time.time()

    smpl_iter = iter(smpl_loader)

    bar = Bar('Eval Src', max=len(data_loader))
    for i, (targets_3d, inputs_2d, _) in enumerate(data_loader):
        # Measure data loading time
        smpl_3d, _, _ = next(smpl_iter)
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)
        
        if(format == 'h36m'):
            smpl_3d = smpl_3d[:, 1:, :].to(device) #input smpl and output h36m
            outputs_3d = model_pos(smpl_3d.view(num_poses, -1))
        else:
            targets_3d = targets_3d[:, 1:, :].to(device) #input h36m and output smpl
            outputs_3d = model_pos(targets_3d.view(num_poses, -1))
        outputs_3d = outputs_3d.view(num_poses, -1, 3).cpu()

        outputs_3d = torch.cat([torch.zeros(num_poses, 1, outputs_3d.size(2)), outputs_3d], 1)  # Pad hip joint (0,0,0)
        if(format == 'h36m'):
            distinct_3d_comp = torch.cat([torch.zeros(num_poses, 1, smpl_3d.size(2)), smpl_3d.cpu()], 1)
        else:
            distinct_3d_comp = torch.cat([torch.zeros(num_poses, 1, targets_3d.size(2)), targets_3d.cpu()], 1)

        # print(outputs_3d[0], smpl_3d_comp[0], targets_3d[0])
        if(format == 'h36m'):
            epoch_loss_3d_pos.update(mpjpe(outputs_3d, targets_3d).item() * 1000.0, num_poses) #compare h36m gt and predictions
            epoch_loss_3d_pos_procrustes.update(p_mpjpe(outputs_3d.numpy(), targets_3d.numpy()).item() * 1000.0, num_poses)
            epoch_loss_3d_smpl.update(mpjpe(distinct_3d_comp, targets_3d).item() * 1000.0, num_poses) #compare h36m gt and smpl gt
            epoch_loss_3d_procrustes_smpl.update(p_mpjpe(distinct_3d_comp.numpy(), targets_3d.numpy()).item() * 1000.0, num_poses)
        else:
            epoch_loss_3d_pos.update(mpjpe(outputs_3d, smpl_3d).item() * 1000.0, num_poses)
            epoch_loss_3d_pos_procrustes.update(p_mpjpe(outputs_3d.numpy(), smpl_3d.numpy()).item() * 1000.0, num_poses)
            epoch_loss_3d_smpl.update(mpjpe(distinct_3d_comp, smpl_3d).item() * 1000.0, num_poses)
            epoch_loss_3d_procrustes_smpl.update(p_mpjpe(distinct_3d_comp.numpy(), smpl_3d.numpy()).item() * 1000.0, num_poses)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                     '| MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f} | SMPL MPJPE {e3: .4f} | SMPL P-MPJPE {e4: .4f}' \
            .format(batch=i + 1, size=len(data_loader), data=data_time.avg, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td, e1=epoch_loss_3d_pos.avg, e2=epoch_loss_3d_pos_procrustes.avg,
                    e3 = epoch_loss_3d_smpl.avg, e4 = epoch_loss_3d_procrustes_smpl.avg)
        bar.next()

    bar.finish()

    return epoch_loss_3d_pos.avg, epoch_loss_3d_pos_procrustes.avg, epoch_loss_3d_smpl.avg, epoch_loss_3d_procrustes_smpl.avg


def evaluate_per_joint(data_loader, smpl_loader, format, model_pos, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    p1_per_joint_smpl = np.zeros((len(data_loader.dataset), 16))
    p1_per_joint_pred = np.zeros((len(data_loader.dataset), 16))
    p2_per_joint_smpl = np.zeros((len(data_loader.dataset), 16))
    p2_per_joint_pred = np.zeros((len(data_loader.dataset), 16))
    print(len(data_loader.dataset))

    # Switch to evaluate mode
    torch.set_grad_enabled(False)
    model_pos.eval()
    end = time.time()

    smpl_iter = iter(smpl_loader)

    bar = Bar('Eval Src', max=len(data_loader))
    for i, (targets_3d, inputs_2d, _) in enumerate(data_loader):
        # Measure data loading time
        smpl_3d, _, _ = next(smpl_iter)
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)
        
        if(format == 'h36m'):
            smpl_3d = smpl_3d[:, 1:, :].to(device) #input smpl and output h36m
            outputs_3d = model_pos(smpl_3d.view(num_poses, -1))
        else:
            targets_3d = targets_3d[:, 1:, :].to(device) #input h36m and output smpl
            outputs_3d = model_pos(targets_3d.view(num_poses, -1))
            
        outputs_3d = outputs_3d.view(num_poses, -1, 3).cpu()

        outputs_3d = torch.cat([torch.zeros(num_poses, 1, outputs_3d.size(2)), outputs_3d], 1)  # Pad hip joint (0,0,0)
        if(format == 'h36m'):
            distinct_3d_comp = torch.cat([torch.zeros(num_poses, 1, smpl_3d.size(2)), smpl_3d.cpu()], 1)
        else:
            distinct_3d_comp = torch.cat([torch.zeros(num_poses, 1, targets_3d.size(2)), targets_3d.cpu()], 1)

        dataset_index = i * num_poses

        if(format == 'h36m'):
            p1_per_joint_smpl[dataset_index:dataset_index+num_poses] = pjpe(distinct_3d_comp, targets_3d).numpy()
            p1_per_joint_pred[dataset_index:dataset_index+num_poses] = pjpe(outputs_3d, targets_3d).numpy()
            p2_per_joint_smpl[dataset_index:dataset_index+num_poses] = p_pjpe(distinct_3d_comp.numpy(), targets_3d.numpy())
            p2_per_joint_pred[dataset_index:dataset_index+num_poses] = p_pjpe(outputs_3d.numpy(), targets_3d.numpy())
        else:
            p1_per_joint_smpl[dataset_index:dataset_index+num_poses] = pjpe(distinct_3d_comp, smpl_3d).numpy()
            p1_per_joint_pred[dataset_index:dataset_index+num_poses] = pjpe(outputs_3d, smpl_3d).numpy()
            p2_per_joint_smpl[dataset_index:dataset_index+num_poses] = p_pjpe(distinct_3d_comp.numpy(), smpl_3d.numpy())
            p2_per_joint_pred[dataset_index:dataset_index+num_poses] = p_pjpe(outputs_3d.numpy(), smpl_3d.numpy())

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} '.format(batch=i + 1, 
                    size=len(data_loader), data=data_time.avg, bt=batch_time.avg, ttl=bar.elapsed_td, eta=bar.eta_td)
        bar.next()

    bar.finish()

    return np.mean(p1_per_joint_pred, axis = 0)*1000, np.mean(p2_per_joint_pred, axis = 0)*1000, np.mean(p1_per_joint_smpl, axis = 0)*1000, np.mean(p2_per_joint_smpl, axis = 0)*1000

if __name__ == '__main__':
    main(parse_args())
