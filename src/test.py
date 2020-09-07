#!/usr/bin/env python

# Created on 2018/12
# Author: Junzhe Zhu & Kaituo XU
import sys
sys.path.append("configs")

import argparse

import torch

import numpy as np
from solver import Solver
from model_mulcat import Dual_RNN_model
import time
import random
import os
import glob
import torch.utils.data as data
from data import load_json
import torchaudio
from pit_criterion import cal_loss, cal_si_snr_with_pit
from duplicate_snr import duplicate_snr
from tqdm import tqdm
from config3 import kernel_size, enc, bottleneck, hidden, num_layers, K, num_spks, multiloss, mul, cat, shuffle, norm, rnn_type, dropout
torch.manual_seed(0)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True

root = "/ws/ifp-10_3/hasegawa/junzhez2/Baseline_Model/dataset"
test_json = ["2spkr_json/tt",
            "3spkr_json/tt",
            "4spkr_json/tt",
            "5spkr_json/tt"]

batch_size = 1
model_path = "models/best.pth"
print_freq = 10
device = 0
onoff_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
pick_using_snr = False

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
    model = Dual_RNN_model(enc, bottleneck, hidden, kernel_size=kernel_size, rnn_type=rnn_type, norm=norm, dropout=dropout, bidirectional=True, num_layers=num_layers, K=K, num_spks=num_spks, multiloss=multiloss, mulcat=(mul, cat)).cuda(device)
    state_dict = torch.load(model_path, map_location=torch.device('cuda:'+str(device)))['state_dict']
    model.load_state_dict(state_dict)
    print('epoch', torch.load(model_path, map_location=torch.device('cuda:'+str(device)))['epoch'])
    model.eval()
    total_snr = np.zeros((len(onoff_thresholds), len(test_json)))
    total_accuracy = np.zeros((len(onoff_thresholds), len(test_json)))
    confusion_matrix = np.zeros((len(onoff_thresholds), num_spks + 1, len(test_json))) # ylabel: [0, num_spks] xlabel: [2, 5]
    counts = np.zeros(len(test_json))
    with torch.no_grad():
        pbar = tqdm(test_loader)
        for i, (mixture, sources) in enumerate(pbar):
            counts[len(sources) - 2] += 1
            ilens = torch.Tensor([mixture.shape[1]]).int()
            # [B, T], [spks, T]
            mixture, sources = mixture.cuda(device), torch.cat(sources, dim=0).cuda(device)
            # [B, #stages, spks, T], [B, #stages, spks]
            estimate_source_list, onoff_list = model(mixture)
            # [spks, T], [spks]
            estimate_source, onoff = estimate_source_list[0, -1], onoff_list[0, -1]
            for idx, onoff_thres in enumerate(onoff_thresholds):
                onoff_pred = onoff > onoff_thres
                if sum(onoff_pred) == 0: # count snr and correctness as 0
                    confusion_matrix[idx, 0, sources.shape[0] - 2] += 1
                    continue
                variable_est = estimate_source[onoff_pred]
                sources = sources[:, :variable_est.shape[1]] # cut off extra samples
                total_accuracy[idx, sources.shape[0] - 2] += variable_est.shape == sources.shape
                if variable_est.shape >= sources.shape:
                    snr = cal_si_snr_with_pit(sources.unsqueeze(0), variable_est.unsqueeze(0), [sources.shape[1]], debug=False, log_vars=False)[0][0].item()
                    total_snr[idx, sources.shape[0] - 2] += snr
                else:
                    snr = duplicate_snr(sources, variable_est)
                    total_snr[idx, sources.shape[0] - 2] += snr
                confusion_matrix[idx, variable_est.shape[0], sources.shape[0] - 2] += 1
            if pick_using_snr:
                maxidx = np.argmax((total_snr / counts).mean(axis=1)) # threshold with maximum average accuracy
            else:
                maxidx = np.argmax((total_accuracy / counts).mean(axis=1)) # threshold with maximum average accuracy
            pbar.set_description('total_snr %s, total_acc %s, threshold %f' % (str(total_snr[maxidx] / counts), 
                                    str(total_accuracy[maxidx]/counts), onoff_thresholds[maxidx]))
        print(confusion_matrix[maxidx])