
import numpy as np
import torch
import torch.utils.data as data
from librosa import load
from time import time
import glob
import os
import random
import json
from tqdm import tqdm
def load_json(filename):
    with open(filename) as f:
        data = json.load(f)
    return data

def printlist(filelist, length=100):
    for i in range(length):
        print(filelist[i])

def pad_audio(audio, len_samples=4*8000):
    if len(audio) < len_samples:
        audio = np.concatenate([audio, np.zeros(len_samples - len(audio))])
    return audio

class MixtureDataset(data.Dataset):
    def __init__(self, root, json_folders, sr=8000, seglen=4.0, minlen=2.0): # segment and cv_maxlen not implemented
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
        seglen = int(seglen * sr)
        minlen = int(minlen * sr)
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
            if mix[2] < minlen:
                continue
            start = 0
            while start + minlen <= mix[2]:
                end = min(start + seglen, mix[2])
                self.examples.append({'mixfile': mix[0], 'sourcefiles': mix[1], 'start': start, 'end':end})
                start += minlen
        random.seed(0)
        self.examples = random.sample(self.examples, len(self.examples))[:3000]
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

def _collate_fn(batch):
    """
    Args:
        batch: list, len(batch) = batch_size, each entry is a tuple of (mixture, sources)
    Returns:
        mixtures_list: B x T, torch.Tensor, padded mixtures
        ilens : B, torch.Tensor, length of each mixture before padding
        sources_list: list of B Tensors, each C x T, where C is (variable) number of source audios
    """
    ilens = [] # shape of mixtures
    mixtures = [] # mixtures, same length as longest source in whole batch
    sources_list = [] # padded sources, same length as mixtures
    for mixture, sources in batch: # compute length to pad to
        assert len(mixture) == len(sources[0])
        assert len(mixture) <= 32000
        ilens.append(len(mixture))
        mixtures.append(pad_audio(mixture))
        sources = torch.Tensor(np.stack([pad_audio(source) for source in sources], axis=0)).float()
        sources_list.append(sources)
    mixtures = torch.Tensor(np.stack(mixtures, axis=0)).float()
    ilens = torch.Tensor(np.stack(ilens)).int()
    return mixtures, ilens, sources_list
root = "/ws/ifp-10_3/hasegawa/junzhez2/Baseline_Model"
test_json = ["2spkr_json/tt",
            "3spkr_json/tt",
            "4spkr_json/tt",
            "5spkr_json/tt"]
print(test_json[0])
dataset = MixtureDataset(root + '/dataset', test_json, 8000, 4.0, 2.0)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=_collate_fn, shuffle=False)


#!/usr/bin/env python

# Created on 2018/12
# Author: Junzhe Zhu & Kaituo XU
import sys
sys.path.append("/ws/ifp-10_3/hasegawa/junzhez2/Baseline_Model")
sys.path.append("/ws/ifp-10_3/hasegawa/junzhez2/Baseline_Model/configs")

import argparse

import torch

import numpy as np
from solver import Solver
from model_multidecoder import Dual_RNN_model
import time
import random
import os
import glob
import torch.utils.data as data
from data import TestDataset
from loss_multidecoder import cal_loss, cal_si_snr_with_pit
from duplicate_snr import duplicate_snr
from tqdm import tqdm
from config4 import kernel_size, enc, bottleneck, hidden, num_layers, K, num_spks, multiloss, mul, cat, shuffle, norm, rnn_type, dropout, maxlen, minlen
torch.manual_seed(0)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True



batch_size = 1
model_path = root + "/models/config4_best.pth"
device = 3
accuracy = 0.0

if __name__ == '__main__':
    #test_dataset = TestDataset(root + '/dataset', test_json)
    #test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    # model
    model = Dual_RNN_model(enc, bottleneck, hidden, kernel_size=kernel_size, rnn_type=rnn_type, norm=norm, dropout=dropout, bidirectional=True, num_layers=num_layers, K=K, num_spks=num_spks, multiloss=multiloss, mulcat=(mul, cat)).cuda(device)
    state_dict = torch.load(model_path, map_location=torch.device('cuda:'+str(device)))['state_dict']
    model.load_state_dict(state_dict)
    print('epoch', torch.load(model_path, map_location=torch.device('cuda:'+str(device)))['epoch'])
    model.eval()
    total_snr = np.zeros(len(test_json))
    total_accuracy = np.zeros(len(test_json))
    confusion_matrix = np.zeros((num_spks - 1, len(test_json))) # ylabel: [0, num_spks] xlabel: [2, 5]
    counts = np.zeros(len(test_json))
    accuracy_chunk = 0 ###########
    chunkcount = 0 #############
    accuracy_whole = 0
    accuracy_voted = 0
    with torch.no_grad():
        pbar = tqdm(loader)
        for i, (mixture, ilens, sources) in enumerate(pbar):
            counts[len(sources[0]) - 2] += 1
            # [B, T], [spks, T]
            mixture, sources = mixture.cuda(device), [s.cuda(device) for s in sources]
            # list of num_decoders, each [B, #stages, spks, T], [B, #stages, spks]
            estimate_source_list, vad = model(mixture)
            vad = vad[0][-1].argmax(0)
            accuracy += (vad == sources[0].shape[0] - 2).int().item()
            # [B, spks, T]
            estimate_source = estimate_source_list[vad][:, -1]
            sources = [s[:, :estimate_source.shape[2]] for s in sources] # cut off extra samples
            for b in range(len(sources)):
                total_accuracy[sources[b].shape[0] - 2] += (estimate_source[b].shape == sources[b].shape)
                if estimate_source[b].shape >= sources[b].shape:
                    snr = cal_si_snr_with_pit(sources[b].unsqueeze(0), estimate_source[b].unsqueeze(0), [sources[b].shape[1]], debug=False, log_vars=False)[0].item()
                else:
                    snr = duplicate_snr(sources[b], estimate_source[b])
                total_snr[sources[b].shape[0] - 2] += snr
                confusion_matrix[estimate_source[b].shape[0] - 2, sources[b].shape[0] - 2] += 1
            pbar.set_description('total_snr %s, total_acc %s, counts %s, acc %f' % (str(total_snr / counts), 
                                    str(total_accuracy / counts), str(counts), accuracy / (i + 1.0)))
        print(confusion_matrix, '\n', confusion_matrix/np.sum(confusion_matrix, axis=0, keepdims=True))