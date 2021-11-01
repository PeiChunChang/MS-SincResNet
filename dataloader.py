import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import numpy as np
import scipy.io as sio
import scipy.io.wavfile
from scipy import signal
import math

def transform(data):
    data = torch.from_numpy(data).float()
    rand_scale = torch.rand(data.size()[0], 1) * 0.2 + 0.9
    rand_scale = rand_scale.repeat(1, data.size()[2])
    rand_scale.unsqueeze_(dim=1)
    rand_noise = (torch.rand(data.size()) - 0.5) * 0.02
    data = torch.mul(data, rand_scale) + rand_noise
    return data

def loader_GTZAN(filename, train=True, fs=16000, ex_sample=2, class_num=10):
    _, data = sio.wavfile.read(filename)
    data = signal.resample(data, fs * 30)
    frame_num = int(data.shape[0] // (fs // 2))
    clip_data = np.zeros((frame_num, 1, fs * (1 + ex_sample)))
    start_point = 0
    step_point = fs // 2
    for i, start in enumerate(range(0, data.shape[0], fs//2)):
        if i < frame_num:
            try:
                clip_data[i, 0, :] = data[start - ex_sample * fs : start + fs]
            except:
                tmp_data = data[start - ex_sample * fs:]
                clip_data[i, 0, :len(tmp_data)] = tmp_data

    if train:
        rand_idx = torch.randint(0, frame_num - 1, (3,))
        clip_data = clip_data[rand_idx, :]

    return clip_data

class myDataset_GTZAN(data.Dataset):
    def __init__(self, folderpath, transform=False, loader=loader_GTZAN, class_num=10, train=True):
        self.folderpath = folderpath
        self.transform = transform
        self.loader = loader
        self.data = []
        self.label = []
        self.labelName = []
        self.train = train
        self.class_num = class_num

        for (i, className) in enumerate(os.listdir(self.folderpath)):
            for data in os.listdir(self.folderpath + '/' + className):
                self.data.append(data)
                self.label.append(int(className))
                self.labelName.append(data[:(data.find('.'))])

    def __getitem__(self, index):
        filename = self.folderpath + '/' + str(self.label[index]) + '/' +self.data[index]
        data = self.loader(filename, train=self.train)

        if self.transform == True:
            data = transform(data)
        else:
            data = torch.from_numpy(data).float()
        label = torch.zeros(data.size()[0]) + self.label[index]

        return data, label.type(torch.long)

    def __len__(self):
        return len(self.data)
