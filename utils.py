import torch
import json


def get_available_device():
    if torch.cuda.is_available():
        return torch.device('cuda', torch.cuda.current_device())
    else:
        return torch.device('cpu')


def get_device_by_id(device_num):
    if device_num < 0:
        return torch.device('cpu')
    else:
        return torch.device('cuda', device_num)


def read_configure(cfg_url):
    with open(cfg_url, 'r') as fp:
        d = json.load(fp)
        return d


def detach_tensor(img):
    return img[0].permute((1, 2, 0)).detach().cpu().numpy().astype('float32').clip(0, 1)
