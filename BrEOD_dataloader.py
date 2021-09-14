import os
from os.path import join
import torch
from torch.utils.data import Dataset
from skimage import io
import numpy as np


class BrEODdataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        if isinstance(root_dir, list):
            self.birf = []
            self.left = []
            self.right = []
            self.disp = []
            for root in root_dir:
                print(root)
                file_list = os.listdir(root)
                self.birf += [join(root, file) for file in file_list if file.endswith("b.png")]
                self.left += [join(root, file) for file in file_list if file.endswith("l.png")]
                self.right += [join(root, file) for file in file_list if file.endswith("r.png")]
                self.disp += [join(root, file) for file in file_list if file.endswith("d.png")]
        else:
            file_list = os.listdir(root_dir)
            self.birf = [join(root_dir, file) for file in file_list if file.endswith("b.png")]
            self.left = [join(root_dir, file) for file in file_list if file.endswith("l.png")]
            self.right = [join(root_dir, file) for file in file_list if file.endswith("r.png")]
            self.disp = [join(root_dir, file) for file in file_list if file.endswith("d.png")]

        self.birf.sort()
        self.left.sort()
        self.right.sort()
        self.disp.sort()

        len_flag = len(self.left) == len(self.right) == len(self.disp) == len(self.birf)
        assert len_flag, "Sample Number Mismatch"

    def __len__(self):
        return len(self.birf)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        im_l = io.imread(self.left[idx]).astype('float32')
        im_r = io.imread(self.right[idx]).astype('float32')
        im_b = io.imread(self.birf[idx]).astype('float32')
        disp = io.imread(self.disp[idx]).astype('float32')

        sample = {
            'im_l': im_l,
            'im_r': im_r,
            'im_b': im_b,
            'disp': disp
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class BrEODToTensor(object):

    def __call__(self, sample):
        im_l, im_r, im_b, disp = sample['im_l'], sample['im_r'], sample['im_b'], sample['disp']
        im_l = im_l.transpose((2, 0, 1))
        im_r = im_r.transpose((2, 0, 1))
        im_b = im_b.transpose((2, 0, 1))
        disp = np.expand_dims(disp, axis=0)
        tensor = {
            'im_l': torch.from_numpy(im_l).float(),
            'im_r': torch.from_numpy(im_r).float(),
            'im_b': torch.from_numpy(im_b).float(),
            'disp': torch.from_numpy(disp).float()
        }
        return tensor


class BrEODNormalize(object):

    def __call__(self, sample):
        tensor = {}
        for key in sample:
            if key[:2] == "im":
                tensor[key] = sample[key] / 255.0
            else:
                tensor[key] = sample[key]
        return tensor


class BrEODDisparityConversion(object):
    def __init__(self, min_val=0, max_val=30):
        assert isinstance(min_val, (float, int))
        assert isinstance(max_val, (float, int))
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, sample):
        tensor = {}
        for key in sample:
            if key[:2] == "im":
                tensor[key] = sample[key]
            else:
                tensor[key] = sample[key] * (self.max_val - self.min_val) / 255.0 + self.min_val

        return tensor

class BrEODCrop(object):

    def __init__(self, roi):
        assert isinstance(roi, tuple)
        self.roi = roi

    def __call__(self, sample):
        top, left, bottom, right = self.roi
        tensor = {}
        for key in sample:
            tensor[key] = sample[key][top:bottom, left:right]

        return tensor


class BrEODRandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        img = sample['im_b']
        h, w = img.shape[:2]
        new_h, new_w = self.output_size

        if h == new_h:
            top = 0
        else:
            top = np.random.randint(0, h - new_h)
        if w == new_w:
            left = 0
        else:
            left = np.random.randint(0, w - new_w)

        tensor = {}

        for key in sample:
            tensor[key] = sample[key][top: top + new_h, left: left + new_w]

        return tensor