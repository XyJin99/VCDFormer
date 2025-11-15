# from __future__ import print_function
import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import sys
import datetime
from utils import Logger
import numpy as np
import random
from dataset import TestDataset
import json
import cv2
import importlib
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='PyTorch VCDFormer Example')
parser.add_argument('--config', type=str, default='config/test/test.json')

systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
opt = parser.parse_args()

def set_seed(config,seed=10):
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

    save_test_log = 'results/' + config['name'] + '/log'
    sys.stdout = Logger(os.path.join(save_test_log, systime + '_test_' + config['name'] + '.txt'))

    print(config)
    print('Set random seed to ', seed)

    # loading data
    print('===> Loading Test Datasets')
    test_dataset = TestDataset(config['test_data_loader'])
    test_loader = DataLoader(dataset=test_dataset, num_workers=0, batch_size=1, shuffle=False)
    print('===> DataLoading Finished, TestSet Length:{}, TestLoader Length:{}'
          .format(len(test_dataset), len(test_loader)))

    # initialize model
    print('===> Loading Test Model')
    net = importlib.import_module('model.' + config['model']['net'])
    netG = net.VCDNet()
    print('===> {} model has been initialized'.format(config['model']['net']))
    print(netG)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in netG.parameters()) / 1e6))
    netG = netG.cuda()

    # load pretrained model
    print('===> load pretrained model')
    if config['modelpath'] and os.path.exists(config['modelpath']):
        print('===> pretrained model {} is load'.format(config['modelpath']))
        netG.load_state_dict(torch.load(config['modelpath']))
    else:
        print('pretrain model is not exists. train with initial')
    netG.eval()


    print('===> start evaluation...')
    for index, data in enumerate(test_loader):
        with torch.no_grad():
            frames, masks, video_name = data
            video_length = frames.size(1)

            frames = frames.cuda()
            pre_masks = [None] * video_length

            radius = config['test_data_loader']['local_stride']
            for f in range(0, video_length):
                pre_index = f - radius if f - radius >= 0 else f + 2 * radius
                aft_index = f + radius if f + radius < video_length else f - 2 * radius
                neighbor_ids = [pre_index, f, aft_index]
                masked_frames = frames[:1, neighbor_ids, :, :, :]

                mod_size_h = config['test_data_loader']['h']
                mod_size_w = config['test_data_loader']['w']
                h, w = frames.shape[-2], frames.shape[-1]
                if mod_size_h != h or mod_size_w != w:
                    print('rescale before input!', 'from', h, 'to', mod_size_h)
                    masked_frames = F.interpolate(masked_frames.view(-1, 3, h, w), size=(mod_size_w, mod_size_h),
                                                  mode='bilinear', align_corners=False).view(1, -1, 3, mod_size_h,
                                                                                             mod_size_w)
                pred_img = netG(masked_frames)
                pred_img = pred_img[:, 1]
                if mod_size_h != h or mod_size_w != w:
                    pred_img = F.interpolate(pred_img, size=(h, w), mode='nearest')

                pred = pred_img.sigmoid().data.cpu().permute(0, 2, 3, 1).numpy().squeeze(0)
                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                pred[np.where(pred >= 0.5)] = 1
                pred[np.where(pred < 0.5)] = 0
                pre_masks[f] = np.array(pred).astype(np.float32)

            if config['test_data_loader']['save_results']:
                save_frame_path = os.path.join('results', config['name'], 'pic', video_name[0])
                os.makedirs(save_frame_path, exist_ok=True)
                for i, frame in enumerate(pre_masks):
                    cv2.imwrite(os.path.join(save_frame_path, f"{i:08d}.png"), (frame * 255).astype(np.uint8))

    print('Finish test')

if __name__=='__main__':
    main()
