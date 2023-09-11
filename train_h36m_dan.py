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
import matplotlib.pyplot as plt


from progress.bar import Bar
from common.log import Logger, savefig
from common.utils import AverageMeter, lr_decay, save_ckpt
from common.data_utils import fetch, read_3d_data, create_2d_data, fetch_surreal
from common.generators import PoseGenerator, PoseGeneratorTarget
from common.loss import mpjpe, p_mpjpe
from models.error_prob import LinearModel, UncertaintyNetwork, UncertaintyLoss, init_weights
from models.pose_converter import MapPoseConverter, ResPoseConverter
from common.tsne import visualize as visualize_tsne

from common.camera import world_to_camera, normalize_screen_coordinates, camera_to_world
from models.domain_utils import DomainAdversarialLoss, DomainDiscriminator
from itertools import cycle

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
        from common.h36m_dataset import Human36mDataset, TRAIN_SUBJECTS, TEST_SUBJECTS
        # surreal_test_path = path.join('data', 'surreal_proc_3d_test.npy')
        surreal_test_path = path.join('data', 'surreal_parsed_val.npy')
        surreal_train_path = path.join('data', 'surreal_parsed_train.npy')
        dataset = SURREALDataset(surreal_test_path)
        dataset_train = SURREALDataset(surreal_train_path)

        dataset_path = path.join('data', 'data_3d_' + 'h36m' + '.npz')
        dataset_h36m = Human36mDataset(dataset_path)
        subjects_train = TRAIN_SUBJECTS
        subjects_test = TEST_SUBJECTS

    dataset_h36m = read_3d_data(dataset_h36m)
    keypoints = create_2d_data(path.join('data', 'data_2d_' + 'h36m' + '_' + 'gt' + '.npz'), dataset_h36m)

    action_filter = None if args.actions == '*' else args.actions.split(',')
    if action_filter is not None:
        action_filter = map(lambda x: dataset.define_actions(x)[0], action_filter)
        print('==> Selected actions: {}'.format(action_filter))

    stride = args.downsample
    cudnn.benchmark = True
    device = torch.device(args.device)

    # Create model
    print("==> Creating model...")
    num_joints = 16
    model_pos = LinearModel(num_joints * 2, (num_joints - 1) * 3).to(device)
    model_conversion = ResPoseConverter((num_joints - 1) * 3, 1024).to(device)
    model_dann = DomainDiscriminator(in_feature = (num_joints - 1) * 3, hidden_size = 128).to(device)
    model_uncertainty = UncertaintyNetwork(1024, linear_size=1024, num_joints=num_joints-1, device=args.device).to(device)
    model_pos.apply(init_weights)
    model_uncertainty.apply(init_weights)
    model_dann.apply(init_weights)
    print("==> Total parameters: {:.2f}M".format((sum(p.numel() for p in model_pos.parameters()) + sum(p.numel() for p in model_dann.parameters()))/ 1000000.0))

    criterion = nn.MSELoss(reduction='mean').to(device)
    optimizer = torch.optim.Adam( 
            [
                {'params': model_pos.parameters()},
                {'params': model_dann.parameters()},
                {'params': model_uncertainty.parameters()}
            ], lr=args.lr)  

    domain_loss = DomainAdversarialLoss(model_dann).to(device)

    ckpt_path_conv = 'checkpoint/res_conversion/current_results/ckpt_best.pth.tar'
    model_conversion.load_state_dict(torch.load(ckpt_path_conv, map_location = device)['state_dict'])
    model_conversion.eval()

    if args.evaluate:
        print('==> Evaluating...')
        ckpt_path = args.evaluate

        if path.isfile(ckpt_path):
            print("==> Loading checkpoint '{}'".format(ckpt_path))
            ckpt = torch.load(ckpt_path)
            start_epoch = ckpt['epoch']
            error_best = ckpt['error']
            glob_step = ckpt['step']
            lr_now = ckpt['lr']
            model_pos.load_state_dict(ckpt['state_dict'], map_location=device)
            print("==> Loaded checkpoint (Epoch: {} | Error: {})".format(start_epoch, error_best))
        
        else:
            raise RuntimeError("==> No checkpoint found at '{}'".format(ckpt_path))

        if action_filter is None:
            action_filter = dataset_h36m.define_actions()

        errors_p1 = np.zeros(len(action_filter))
        errors_p2 = np.zeros(len(action_filter))

        for i, action in enumerate(action_filter):
            poses_valid, poses_valid_2d, actions_valid = fetch(subjects_test, dataset_h36m, keypoints, [action], stride)
            print('==> Evaluating action: {}'.format(action))
            h36m_valid_loader = DataLoader(PoseGenerator(poses_valid, poses_valid_2d, actions_valid),
                                        batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers, pin_memory=True)
            _, _, errors_p1[i], errors_p2[i] = evaluate([], h36m_valid_loader, model_pos, model_conversion, device)

        poses_valid_sur, poses_valid_2d_sur, = fetch_surreal(dataset, stride)
        sur_valid_loader = DataLoader(PoseGeneratorTarget(poses_valid_sur, poses_valid_2d_sur),
                                batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True, drop_last=False)
        surr_p1, surr_p2, _, _ = evaluate(sur_valid_loader, [], model_pos, model_conversion, device)
        print('H36M Protocol #1   (MPJPE) action-wise average: {:.2f} (mm)'.format(np.mean(errors_p1).item()))
        print('H36M Protocol #2 (P-MPJPE) action-wise average: {:.2f} (mm)'.format(np.mean(errors_p2).item()))
        print('SURREAL Protocol #1   (MPJPE) action-wise average: {:.2f} (mm)'.format(surr_p1))
        print('SURREAL Protocol #2 (P-MPJPE) action-wise average: {:.2f} (mm)'.format(surr_p2))
        exit(0)
    
    poses_train, poses_train_2d, = fetch_surreal(dataset_train, stride)
    train_loader = DataLoader(PoseGeneratorTarget(poses_train, poses_train_2d),
                                batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, pin_memory=True, drop_last=True)

    poses_valid, poses_valid_2d, = fetch_surreal(dataset, stride)
    valid_loader = DataLoader(PoseGeneratorTarget(poses_valid, poses_valid_2d),
                                batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
    
    surreal_train_loader = ForeverDataIterator(train_loader)

    h36m_train, h36m_train_2d, actions_train = fetch(subjects_train, dataset_h36m, keypoints, action_filter, stride)
    h36m_train_loader = DataLoader(PoseGenerator(h36m_train, h36m_train_2d, actions_train), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    h36m_train_loader = ForeverDataIterator(h36m_train_loader)

    h36m_valid, h36m_valid_2d, actions_valid = fetch(subjects_test, dataset_h36m, keypoints, action_filter, stride)
    h36m_valid_loader = DataLoader(PoseGenerator(h36m_valid, h36m_valid_2d, actions_valid), batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)


    ckpt_dir_path = path.join(args.checkpoint, 'h36_dann', datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f'))
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
        epoch_loss, lr_now, glob_step = train(surreal_train_loader, h36m_train_loader, model_pos, model_conversion,
                                              model_dann, model_uncertainty, criterion, domain_loss, optimizer, device, 
                                              args.lr, lr_now, epoch, glob_step, args.lr_decay, args.lr_gamma,
                                              max_norm=args.max_norm)

        # Evaluate
        error_eval_p1, error_eval_p2, error_h36_p1, error_h36_p2 = evaluate(valid_loader, h36m_valid_loader, model_pos, model_conversion, device)

        # Save checkpoint
        if error_best is None or error_best > error_h36_p1:
            error_best = error_h36_p1
            save_ckpt({'epoch': epoch + 1, 'lr': lr_now, 'step': glob_step, 'state_dict': model_pos.state_dict(),
                       'optimizer': optimizer.state_dict(), 'error': error_eval_p1}, ckpt_dir_path, suffix='best')

        if (epoch + 1) % args.snapshot == 0:
            save_ckpt({'epoch': epoch + 1, 'lr': lr_now, 'step': glob_step, 'state_dict': model_pos.state_dict(),
                       'optimizer': optimizer.state_dict(), 'error': error_eval_p1}, ckpt_dir_path)

    exit(0)

def train(data_loader, h36m_train, model_pos, model_conv, model_dann, model_uncertainty, criterion, domain_loss, optimizer, device, lr_init, lr_now, epoch, step, decay, gamma, max_norm=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()
    epoch_loss_dom = AverageMeter()
    epoch_acc_dom = AverageMeter()

    # Switch to train mode
    torch.set_grad_enabled(True)
    model_pos.train()
    model_conv.eval()
    end = time.time()
    criterionL1 = nn.L1Loss(reduction='mean').to(device)
    criterion_unc = UncertaintyLoss().to(device)
    lamda = 0.1
    trade_off_unc = 0.001

    trade_off = 0.001
    pretrain = 0

    max_iters = len(data_loader) if len(data_loader) > len(h36m_train) else len(h36m_train)
    bar = Bar('Train', max=max_iters)
    for i in range(max_iters):
        # Measure data loading time
        
        
        h36m_3d, h36m_2d, _ = next(h36m_train)
        surreal_3d, surreal_2d = next(data_loader)
        
        data_time.update(time.time() - end)
        num_poses = surreal_3d.size(0)

        step += 1
        if step % decay == 0 or step == 1:
            lr_now = lr_decay(optimizer, step, lr_init, decay, gamma)

        orig_surreal_3d, surreal_2d = surreal_3d[:, 1:, :].to(device), surreal_2d.to(device)  # Remove hip joint for 3D poses
        h36m_3d, h36m_2d = h36m_3d[:, 1:, :].to(device), h36m_2d.to(device)
        
        optimizer.zero_grad()
        # Compute domain discrepancy
        pred_h36m, feat_h36m = model_pos(h36m_2d.view(num_poses, -1))
        pred_h36m = pred_h36m.view(num_poses, -1, 3)

        uncertainty_values = model_uncertainty(feat_h36m)

        loss_3d_pos = criterion(h36m_3d, pred_h36m) + lamda*criterionL1(h36m_3d, pred_h36m)
        unc_error = criterion_unc(uncertainty_values, pred_h36m, h36m_3d)

        if(epoch > pretrain):
            pred_surreal, _ = model_pos(surreal_2d.view(num_poses, -1))

        if(epoch > pretrain):
            dom_loss = domain_loss(pred_h36m.view(num_poses, -1), pred_surreal)

            total_loss = loss_3d_pos + trade_off * dom_loss + unc_error*trade_off_unc
        else:
            total_loss = loss_3d_pos + unc_error*trade_off_unc

        
        total_loss.backward()
        if max_norm:
            nn.utils.clip_grad_norm_(model_pos.parameters(), max_norm=1)
            nn.utils.clip_grad_norm_(model_dann.parameters(), max_norm=1)
        optimizer.step()

        epoch_loss_3d_pos.update(loss_3d_pos.item(), num_poses)
        if(epoch > pretrain):
            domain_acc = domain_loss.domain_discriminator_accuracy
            epoch_acc_dom.update(domain_acc, num_poses)
            epoch_loss_dom.update(dom_loss.item(), num_poses)
        

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                     '| Loss: {loss: .4f} | Dom Loss: {dom_loss: .4f} | Dom Acc: {dom_acc: .4f}' \
            .format(batch=i + 1, size=max_iters, data=data_time.avg, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td, loss=epoch_loss_3d_pos.avg, dom_loss = epoch_loss_dom.avg, dom_acc = epoch_acc_dom.avg)
        bar.next()

    bar.finish()
    return epoch_loss_3d_pos.avg, lr_now, step

def evaluate(data_loader, h36m_loader, model_pos, model_conversion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()
    epoch_loss_3d_pos_procrustes = AverageMeter()

    epoch_loss_3d_h36m = AverageMeter()
    epoch_loss_3d_h36m_procrustes = AverageMeter()

    # Switch to evaluate mode
    torch.set_grad_enabled(False)
    model_conversion.eval()
    model_pos.eval()
    end = time.time()

    bar = Bar('Eval H36M', max=len(h36m_loader))
    for i, (targets_3d, inputs_2d, _) in enumerate(h36m_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)

        inputs_2d = inputs_2d.to(device)
        targets_3d = targets_3d
        outputs_3d, _ = model_pos(inputs_2d.view(num_poses, -1))
        outputs_3d = outputs_3d.view(num_poses, -1, 3).cpu()
        outputs_3d = torch.cat([torch.zeros(num_poses, 1, outputs_3d.size(2)), outputs_3d], 1)  # Pad hip joint (0,0,0)
        
        epoch_loss_3d_h36m.update(mpjpe(outputs_3d, targets_3d).item() * 1000.0, num_poses)
        epoch_loss_3d_h36m_procrustes.update(p_mpjpe(outputs_3d.numpy(), targets_3d.numpy()).item() * 1000.0, num_poses)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                     '| Src MPJPE: {et1: .4f}, Src P-MPJPE: {et2: .4f}' \
            .format(batch=i + 1, size=len(h36m_loader), data=data_time.avg, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td, et1 = epoch_loss_3d_h36m.avg, et2 = epoch_loss_3d_h36m_procrustes.avg)
        bar.next()

    bar.finish()

    bar = Bar('Eval SURREAL', max=len(data_loader))
    for i, (surreal_3d, surreal_2d) in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses_sur = surreal_3d.size(0)

        surreal_2d = surreal_2d.to(device)
        surreal_3d = surreal_3d.to(device)
        surreal_pred, _ = model_pos(surreal_2d.view(num_poses_sur, -1))
        surreal_pred = surreal_pred.view(num_poses_sur, -1, 3).cpu()
        targets_3d_conv = model_conversion(surreal_3d[:,1:,:].view(num_poses_sur, -1)).view(num_poses_sur, -1, 3).cpu()
        targets_3d_conv = torch.cat([torch.zeros(num_poses_sur, 1, targets_3d_conv.size(2)), targets_3d_conv], 1)  # Pad hip joint (0,0,0)
        surreal_pred = torch.cat([torch.zeros(num_poses_sur, 1, surreal_pred.size(2)), surreal_pred], 1)  # Pad hip joint (0,0,0)
        targets_3d_conv = surreal_3d.cpu()
        
        epoch_loss_3d_pos.update(mpjpe(surreal_pred, targets_3d_conv).item() * 1000.0, num_poses_sur)
        epoch_loss_3d_pos_procrustes.update(p_mpjpe(surreal_pred.numpy(), targets_3d_conv.numpy()).item() * 1000.0, num_poses_sur)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                     '| Tgt MPJPE: {et1: .4f}, Tgt P-MPJPE: {et2: .4f}' \
            .format(batch=i + 1, size=len(data_loader), data=data_time.avg, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td, et1 = epoch_loss_3d_pos.avg, et2 = epoch_loss_3d_pos_procrustes.avg)
        bar.next()

    bar.finish()
    return epoch_loss_3d_pos.avg, epoch_loss_3d_pos_procrustes.avg, epoch_loss_3d_h36m.avg, epoch_loss_3d_h36m_procrustes.avg

if __name__ == '__main__':
    main(parse_args())
