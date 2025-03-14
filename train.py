from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import time
import datetime
import sys
from utils import Logger
from torch.utils.data import DataLoader
from dataset import TestDataset,RepeatDataset,TrainDataset_VCD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tensorboardX import SummaryWriter
import numpy as np
import random
import json
import cv2
from metrics.vos import f_boundary, jaccard
import importlib
import loss

parser = argparse.ArgumentParser(description='PyTorch VCDFormer Example')
parser.add_argument('--config', type=str, default='config/train/train.json')

systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
opt = parser.parse_args()

def set_seed(config, seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_devices']
    if not torch.cuda.is_available():
        raise Exception('No Gpu found, please run with gpu')
    else:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = True

def main():
    config = json.load(open(opt.config))
    seed = config.get('seed', random.randint(1, 10000))
    set_seed(config, seed)

    writer_path = 'experiments/' + config['name'] + '/tb_logger'
    writer = SummaryWriter(writer_path)

    save_train_log = 'experiments/' + config['name'] + '/log'
    sys.stdout = Logger(os.path.join(save_train_log, systime + 'train_' + config['name'] + '.txt'))
    print('Set random seed to ', seed)

    print(config)
    print('===> Loading Datasets')
    train_dataset = TrainDataset_VCD(config['train_data_loader'])

    IfIterations = config['trainer'].get('iterations', False)

    if IfIterations:
        train_dataset = RepeatDataset(train_dataset, config['trainer'].get('data_enlarge', 1000))
    train_loader = DataLoader(dataset=train_dataset, num_workers=config['trainer']['num_workers'],
                              batch_size=config['trainer']['batch_size'],  shuffle=True, pin_memory=True, prefetch_factor=2)

    if IfIterations and len(train_loader) < config['trainer']['iterations']:
        raise Exception('data enlarge is not enough.'
                        'Get train_loader length={},nIter={}'.format(len(train_loader), config['trainer']['iterations']))

    test_dataset = TestDataset(config['test_data_loader'])
    test_loader = DataLoader(dataset=test_dataset, num_workers=0, batch_size=1, shuffle=False)
    print('===> DataLoading Finished')
    print('TrainSet Length: ',len(train_dataset),'TestSet Length: ',len(test_dataset))
    print('TrainLoader Length: ', len(train_loader), 'TestLoader Length: ', len(test_loader))

    print('===> Loading Models')
    net = importlib.import_module('model.'+config['model']['net'])
    model = net.VCDNet()
    print('===> {} model has been initialized'.format(config['model']['net']))
    print(model)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))
    model = model.cuda()

    bce_loss = nn.BCEWithLogitsLoss().cuda()
    iou_loss = loss.IOU(size_average=True).cuda()
    optimizer = torch.optim.Adam(model.parameters(),  lr=config['trainer']['lr'],
                                 betas=(config['trainer']['beta1'], config['trainer']['beta2']))

    scheduler_opt = config['trainer']['scheduler']
    scheduler_type = scheduler_opt.pop('type')
    if scheduler_type == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=scheduler_opt['milestones'],
            gamma=scheduler_opt['gamma'])
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=scheduler_opt['periods'],
            eta_min=scheduler_opt['eta_min'])
    else:
        raise NotImplementedError(
            f'Scheduler {scheduler_type} is not implemented yet.')

    Startiter = 0
    if config['trainer']['restore']:
        data = torch.load(config['trainer']['gen_path'],map_location='cuda:0')
        model.load_state_dict(data)

    if config['trainer']['resume']:
        data = torch.load(config['trainer']['gen_path'],map_location='cuda:0')
        model.load_state_dict(data)
        data_opt = torch.load(config['trainer']['opt_path'],map_location='cuda:0')
        optimizer.load_state_dict(data_opt['optim'])
        scheduler.load_state_dict(data_opt['sche'])
        Startiter = data_opt['iteration'] + 1

    print('start training...')
    BestJFmean = 0
    BestIter = 0
    Loss_summary = {'bce_loss': 0, 'iou_loss': 0, 'total_loss': 0}

    for iter, data in enumerate(train_loader):
        model.train()
        iter = iter + Startiter
        if iter > config['trainer']['iterations']:
            break

        t2 = time.time()
        frames, masks, _ = data[0], data[1],data[2]
        center_index = frames.shape[1] // 2
        mask = masks[:, center_index, :, :, :]
        frames, mask = frames.cuda(), mask.cuda()

        t0 = time.time()
        pred = model(frames)
        t1 = time.time()

        total_loss = 0
        bce_out = bce_loss(pred[:, center_index, :, :, :], mask)
        total_loss += bce_out * config['losses']['bce_weight']

        iou_out = iou_loss(torch.sigmoid(pred[:, center_index, :, :, :]), mask)
        total_loss += iou_out * config['losses']['iou_weight']

        writer.add_scalar('loss/gen/bce_loss', bce_out.item(), iter)
        Loss_summary['bce_loss'] += bce_out.item()
        writer.add_scalar('loss/gen/iou_loss', iou_out.item(), iter)
        Loss_summary['iou_loss'] += iou_out.item()
        writer.add_scalar('loss/gen/total_loss', total_loss.item(), iter)
        Loss_summary['total_loss'] += total_loss.item()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        t3 = time.time()

        if iter % (config['trainer']['log_freq']) == 0 and iter != 0:
            for key, value in Loss_summary.items():
                Loss_summary[key] = value/config['trainer']['log_freq']

            print("===> iter[{}]/{}): bce_Loss: {:.4f}  iou_Loss: {:.4f}  "
                  "total_Loss: {:.4f}  lr: {:.6E}  Timer: {:.4f}/{:.4f} sec."
                  .format(iter, config['trainer']['iterations'], Loss_summary['bce_loss'],
                          Loss_summary['iou_loss'], Loss_summary['total_loss'],
                          optimizer.param_groups[0]['lr'], (t1 - t0), (t3 - t2)))

            writer.add_scalar('fre_loss/bce_loss', Loss_summary['bce_loss'], iter)
            writer.add_scalar('fre_loss/iou_loss', Loss_summary['iou_loss'], iter)
            writer.add_scalar('fre_loss/total_loss', Loss_summary['total_loss'], iter)
            for key,value in Loss_summary.items():
                Loss_summary[key] = 0

        if iter % (config['trainer']['save_freq']) == 0 and iter != 0:
            save_checkpoint(model, iter, optimizer, scheduler, config)
            JFmean = TestWhileTrain(test_loader, model, iter, config, writer)
            if BestJFmean <= JFmean:
                BestJFmean = JFmean
                BestIter = iter

    print('training is finished')
    print("Best Iter: {:0>2} \t Best Acc: {:.3f}".format(BestIter, BestJFmean))
    writer.close()


def TestWhileTrain(val_loader, net, iter, config, writer):
    print('-----value stage-----')
    net.eval()
    total_J, total_F, total_JFmean = [],[],[]
    with torch.no_grad():
        for index, data in enumerate(val_loader):
            frames, masks, video_name = data
            video_length = frames.size(1)
            gt_masks = [
                masks[:, i, :, :, :].permute(0, 2, 3, 1).numpy().squeeze(0).astype(np.float32) for i in range(video_length)     # h w c
            ]
            frames = frames.cuda()
            preds = [None] * video_length

            radius = config['test_data_loader']['local_stride']
            for f in range(0, video_length):
                pre_index = f - radius if f - radius >= 0 else f + 2 * radius
                aft_index = f + radius if f + radius < video_length else f - 2 * radius
                neighbor_ids = [pre_index, f, aft_index]
                selected_frames = frames[:, neighbor_ids, :, :, :]

                pred = net(selected_frames)[:, 1]
                pred = pred.sigmoid().data.cpu().permute(0, 2, 3, 1).numpy().squeeze(0)
                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                pred[np.where(pred >= 0.5)] = 1
                pred[np.where(pred < 0.5)] = 0
                preds[f] = np.array(pred).astype(np.float32)

            curv_J, curv_F, curv_JFmean = [],[],[]
            for gt, pred in zip(gt_masks, preds):
                J_score = jaccard.db_eval_iou(annotation=gt[:,:,0], segmentation=pred[:,:,0])
                F_score = f_boundary.db_eval_boundary(foreground_mask=gt[:,:,0], gt_mask=pred[:,:,0])
                JFmean = (J_score + F_score) / 2

                curv_J.append(J_score)
                curv_F.append(F_score)
                curv_JFmean.append(JFmean)

                total_J.append(J_score)
                total_F.append(F_score)
                total_JFmean.append(JFmean)

            cur_J = sum(curv_J) / len(curv_J)
            cur_F = sum(curv_F) / len(curv_F)
            cur_JFmean = sum(curv_JFmean) / len(curv_JFmean)
            print( f'[{index + 1}/{len(val_loader)}] Name: {str(video_name)} '
                   f'| J F JFMean: {cur_J:.4f} {cur_F:.4f} {cur_JFmean:.4f}')
            writer.add_scalar('scene_J/{}'.format(video_name), cur_J, iter)
            writer.add_scalar('scene_F/{}'.format(video_name), cur_F, iter)
            writer.add_scalar('scene_JFMean/{}'.format(video_name), cur_JFmean, iter)

            if config['test_data_loader']['save_results']:
                save_frame_path = os.path.join('experiments', config['name'], iter, 'pic', video_name[0])
                os.makedirs(save_frame_path, exist_ok=False)
                for i, pred in enumerate(preds):
                    cv2.imwrite(os.path.join(save_frame_path, str(i).zfill(8) + '.png'),
                                cv2.cvtColor((pred * 255.0).astype(np.uint8),  cv2.IMREAD_GRAYSCALE))

    avg_J = sum(total_J) / len(total_J)
    avg_F = sum(total_F) / len(total_F)
    avg_JFmean = sum(total_JFmean) / len(total_JFmean)
    print('Finish evaluation... Average J F JFMean: 'f'{avg_J:.4f} {avg_F:.4f} {avg_JFmean:.4f}')
    writer.add_scalar('iter_J', avg_J, iter)
    writer.add_scalar('iter_F', avg_F, iter)
    writer.add_scalar('iter_JFmean', avg_JFmean, iter)
    return avg_JFmean

def save_checkpoint(net, iter, optimizer, scheduler, config):
    save_model_path = 'experiments/' + config['name'] + '/weight'
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    model_name = 'iter_{}.pth'.format(iter)
    torch.save(net.state_dict(), os.path.join(save_model_path, model_name))

    save_state_path = 'experiments/' + config['name'] + '/state'
    if not os.path.exists(save_state_path):
        os.makedirs(save_state_path)
    opt_name = 'opt_{}.pth'.format(iter)
    torch.save( {   'iteration': iter,
                    'optim': optimizer.state_dict(),
                    'sche': scheduler.state_dict(),
                    }
                , os.path.join(save_state_path, opt_name))
    print('Checkpoint saved to {}'.format(os.path.join(save_model_path,model_name)))

if __name__ == '__main__':
    main()
