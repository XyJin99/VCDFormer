from PIL import Image
import os
import torch.utils.data as data
import torch
from torch.utils.data import DataLoader
import numpy as np
import json
import cv2
import torchvision.transforms as transforms
import random


class RepeatDataset:
    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        return self.times * self._ori_len


def data_augmentation(cloud, mask, flip_h=True, mir=True, rot=True):
    if flip_h == True and random.random() < 0.5:
        cloud = [i.transpose(Image.Transpose.FLIP_TOP_BOTTOM) for i in cloud]
        mask = [i.transpose(Image.Transpose.FLIP_TOP_BOTTOM) for i in mask]

    if mir == True and random.random() < 0.5:
        cloud = [i.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for i in cloud]
        mask = [i.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for i in mask]

    if rot == True and random.random() < 0.5:
        cloud = [i.transpose(Image.Transpose.ROTATE_90) for i in cloud]
        mask = [i.transpose(Image.Transpose.ROTATE_90) for i in mask]

    return cloud, mask


class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        mode = img_group[0].mode
        if mode == '1':
            img_group = [img.convert('L') for img in img_group]
            mode = 'L'
        if mode == 'L':
            return np.stack([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif mode == 'RGB':
            if self.roll:
                return np.stack([np.array(x)[:, :, ::-1] for x in img_group],
                                axis=2)
            else:
                return np.stack(img_group, axis=2)
        else:
            raise NotImplementedError(f"Image mode {mode}")


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic).permute(2, 3, 0, 1).contiguous()
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(
                pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float().div(255) if self.div else img.float()
        return img


class TrainDataset_VCD(torch.utils.data.Dataset):
    ''' random select ref frames and resize(not crop) frames '''
    def __init__(self, args: dict, debug=False):
        self.args = args
        self.local_stride = args['local_stride']
        self.size = self.w, self.h = (args['w'], args['h'])
        self.img_size = (640, 640)

        json_path = args['filepath']
        with open(json_path, 'r') as f:
            self.video_dict = json.load(f)
        self.video_names = list(self.video_dict.keys())

        if debug:
            self.video_names = self.video_names[:100]

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),
        ])

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        item = self.load_item(index)
        return item

    def _sample_index(self, length, radius):
        pivot = random.randint(0, length - 1)
        pre_index = pivot - radius if pivot - radius >= 0 else pivot + 2 * radius
        aft_index = pivot + radius if pivot + radius < length else pivot - 2 * radius
        local_idx = [pre_index, pivot, aft_index]
        return local_idx

    def load_item(self, index):
        video_name = self.video_names[index]
        video_path = os.path.join(self.args['dirpath'], 'cloud', video_name)
        frame_list = sorted(os.listdir(video_path))
        mask_path = os.path.join(self.args['dirpath'], 'mask', video_name)
        mask_list = sorted(os.listdir(mask_path))

        selected_index = self._sample_index(self.video_dict[video_name], self.local_stride)
        h = random.randint(0, self.img_size[1]-self.h)
        w = random.randint(0, self.img_size[0]-self.w)

        # read video frames
        frames = []
        masks = []
        for idx in selected_index:
            img = Image.open(os.path.join(video_path, frame_list[idx])).convert('RGB')
            img = img.crop((w, h, w + self.w, h + self.h))
            frames.append(img)

            mask = Image.open(os.path.join(mask_path, mask_list[idx])).convert('L')
            mask = mask.crop((w, h, w + self.w, h + self.h))
            masks.append(mask)

        # normalizate, to tensors
        frames, masks = data_augmentation(frames, masks)
        frame_tensors = self._to_tensors(frames)
        mask_tensors = self._to_tensors(masks)
        frame_tensors = frame_tensors * 2.0 - 1.0

        return frame_tensors, mask_tensors, video_name

class TestDataset(data.Dataset):
    def __init__(self, args):
        self.args = args
        with open(args['filepath'], 'r') as f:
            self.video_dict = json.load(f)
        self.video_names = list(self.video_dict.keys())

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),
        ])

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        item = self.load_item(index)
        return item

    def load_item(self, index):
        video_name = self.video_names[index]
        frame_index = list(range(self.video_dict[video_name]))

        # read video frames and corresponding masks
        frames = []
        masks = []
        video_path = os.path.join(self.args['dirpath'], 'cloud', video_name)
        mask_path = os.path.join(self.args['dirpath'], 'mask', video_name)
        frame_list = sorted(os.listdir(video_path))
        mask_list = sorted(os.listdir(mask_path))

        for idx in frame_index:
            img = Image.open(os.path.join(video_path, frame_list[idx])).convert('RGB')
            frames.append(img)
            mask = Image.open(os.path.join(mask_path, mask_list[idx])).convert('L')
            masks.append(mask)

        # to tensors
        frame_tensors = self._to_tensors(frames)
        mask_tensors = self._to_tensors(masks)
        frame_tensors = frame_tensors * 2.0 - 1.0
        return frame_tensors, mask_tensors, video_name
