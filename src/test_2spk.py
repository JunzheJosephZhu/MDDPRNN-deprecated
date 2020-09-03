#!/usr/bin/env python

# Created on 2018/12
# Author: Junzhe Zhu & Kaituo XU

import argparse

import torch

import numpy as np
from solver import Solver
from model_rnn import Dual_RNN_model
import time
import random
import os
import glob
import torch.utils.data as data
from data import load_json
import torchaudio
from pit_criterion import cal_loss
torch.manual_seed(0)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True

root = "/ws/ifp-10_3/hasegawa/junzhez2/Baseline_Model/dataset"
test_json = ["2spkr_json/tt",
            "3spkr_json/tt",
            "4spkr_json/tt",
            "5spkr_json/tt"][:1]

sample_rate = 8000
maxlen = 4
N = 64 
L = 16 
K = 100
P = 50 
H = 128 
B = 6
C = 2
shuffle = False
batch_size = 1
model_path = "pretrained/pretrained.pth"
print_freq = 10
device = 0

def load(name, sr=8000):
    audio, sr = torchaudio.load(name)
    return audio[0], sr

class TestDataset(data.Dataset):
    def __init__(self, root, json_folders, sr=8000): # segment and cv_maxlen not implemented
        """
        each line of textfile comes in the form of:
            filename1, dB1, filename2, dB2, ...
            args:
                root: folder where dataset/ is located
                json_folders: folders containing json files, **/dataset/#speakers/wav8k/min/tr/**
                sr: sample rate
                seglen: length of each segment in seconds
                minlen: minimum segment length
        """
        self.sr = sr
        self.mixes = []
        for json_folder in json_folders:
            mixfiles, wavlens = list(zip(*load_json(os.path.join(root, json_folder, 'mix.json')))) # list of 20000 filenames, and 20000 lengths
            mixfiles = [os.path.join(root, mixfile.split('dataset/')[1]) for mixfile in mixfiles]
            sig_json = [load_json(file) for file in sorted(glob.glob(os.path.join(root, json_folder, 's*.json')))] # list C, each have 20000 filenames
            for i, spkr_json in enumerate(sig_json):
                sig_json[i] = [os.path.join(root, line[0].split('dataset/')[1]) for line in spkr_json] # list C, each have 20000 filenames
            siglists = list(zip(*sig_json)) # list of 20000, each have C filenames
            self.mixes += list(zip(mixfiles, siglists, wavlens))
        #printlist(self.mixes)
        self.examples = []
        for i, mix in enumerate(self.mixes):
            self.examples.append({'mixfile': mix[0], 'sourcefiles': mix[1], 'start': 0, 'end': mix[2]})
        random.seed(0)
        self.examples = random.sample(self.examples, len(self.examples))
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        """
        Returns:
            mixture: [T]
            sources: list of C, each [T]
        """
        example = self.examples[idx]
        mixfile, sourcefiles, start, end = example['mixfile'], example['sourcefiles'], example['start'], example['end']
        mixture, sr = load(mixfile, sr=self.sr)
        assert sr == self.sr, 'need to resample'
        mixture = mixture[start:end]
        sources = [load(sourcefile, sr=sr)[0][start:end] for sourcefile in sourcefiles]
        return mixture, sources

if __name__ == '__main__':
    test_dataset = TestDataset(root, test_json)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    # model
    model = Dual_RNN_model(256, 64, 128, bidirectional=True, num_layers=6, K=250).cuda(device)
    state_dict = torch.load(model_path, map_location=torch.device('cuda:'+str(device)))['model_state_dict']
    model.load_state_dict(state_dict)
    print('epoch', torch.load(model_path, map_location=torch.device('cuda:'+str(device)))['epoch'])
    model.eval()
    total_loss = []
    with torch.no_grad():
        for i, (mixture, sources) in enumerate(test_loader):
            ilens = torch.Tensor([mixture.shape[1]]).int()
            mixture, sources = mixture.cuda(device), [torch.cat(sources, dim=0).cuda(device)]
            estimate_source = model(mixture)
            loss, _ = cal_loss(sources, estimate_source, ilens, debug=False)
            total_loss.append(loss.item())
            print(np.mean(total_loss))
        
