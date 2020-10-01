#!/usr/bin/env python

# Created on 2018/12
# Author: Junzhe Zhu & Kaituo XU
import sys
import os

root = "/ws/ifp-10_3/hasegawa/junzhez2/Baseline_Model"
sys.path.append(os.path.join(root, "configs"))

import argparse

import torch
import numpy as np
from solver import Solver
from model_multidecoder import Dual_RNN_model
import time
import random
import glob
import torch.utils.data as data
from data import TestDataset
from loss_multidecoder import cal_loss, cal_si_snr_with_pit
from duplicate_snr import duplicate_snr
from tqdm import tqdm
from config4 import kernel_size, enc, bottleneck, hidden, num_layers, K, num_spks, mul, cat, norm, rnn_type, dropout, maxlen, minlen
torch.manual_seed(0)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True

test_json = ["2spkr_json/tt",
            "3spkr_json/tt",
            "4spkr_json/tt",
            "5spkr_json/tt"]

model_path = "pretrained/raymond_pretrained_newversion.pth"
device = 3
# chop settings
sr = 8000
seglen = int(maxlen * sr)
minlen = int(minlen * sr)

def chop(mix):
    '''
        chop signal into chunks
        mix: [B, T]
    '''
    start = 0
    chunks = []
    while start + minlen <= mix.shape[1]:
        end = min(start + seglen, mix.shape[1])
        chunks.append(mix[:, start:end])
        start += minlen
    return chunks

if __name__ == '__main__':
    test_dataset = TestDataset(os.path.join(root, 'dataset'), test_json)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    # model
    model = Dual_RNN_model(enc, bottleneck, hidden, kernel_size=kernel_size, rnn_type=rnn_type, norm=norm, dropout=dropout, bidirectional=True, num_layers=num_layers, K=K, num_spks=num_spks, multiloss=False, mulcat=(mul, cat)).cuda(device)
    pkg = torch.load(os.path.join(root, model_path), map_location=torch.device('cuda:' + str(device)))
    model.load_state_dict(pkg['state_dict'])
    print('epoch %d' % pkg['epoch'])
    model.eval()
    total_snr = np.zeros(len(test_json))
    voted_accuracy = np.zeros(len(test_json))
    whole_accuracy = np.zeros(len(test_json))
    confusion_matrix = np.zeros((num_spks - 1, len(test_json))) # ylabel: [0, num_spks] xlabel: [2, 5]
    counts = np.zeros(len(test_json))

    with torch.no_grad():
        pbar = tqdm(test_loader)
        for i, (mixture, sources) in enumerate(pbar):
            counts[len(sources) - 2] += 1
            ilens = torch.Tensor([mixture.shape[1]]).int()
            # [1, T], [spks, T]
            mixture, sources = mixture.cuda(device), torch.cat(sources, dim=0).cuda(device)
            num_sources = sources.size(0)

            # run chunks
            chunks = chop(mixture)
            votes = np.zeros(num_spks - 1)
            for chunk in chunks:
                _, vad_list = model(chunk, torch.Tensor([num_sources]), False)
                vad_chunk = vad_list[0, -1].argmax(0).item()
                votes[vad_chunk] += 1
            vad_voted = np.argmax(votes, axis=0)
            voted_accuracy[num_sources - 2] += vad_voted == num_sources - 2

            # run whole model. list of 1, each [1, spks, T], [1, spks]
            # set to True to use voted vad, false to use whole vad
            estimate_source, vad_whole = model(mixture, torch.Tensor([vad_voted + 2]), True) 
            assert vad_whole.size(0) == 1
            whole_accuracy[num_sources - 2] += (vad_whole[0][-1].argmax(0) == num_sources - 2).int().item()
            
            # estimate_source: [spks, T] sources: [spks, T]
            estimate_source = estimate_source[0][0]
            sources = sources[:, :estimate_source.size(1)] # cut off extra samples
            if estimate_source.shape >= sources.shape:
                snr = cal_si_snr_with_pit(sources, estimate_source.unsqueeze(0), True)[0].item()
            else:
                snr = duplicate_snr(sources, estimate_source)
            print(snr)
            total_snr[num_sources - 2] += snr
            confusion_matrix[estimate_source.shape[0] - 2, sources.shape[0] - 2] += 1
            pbar.set_description('total_snr %s, whole_acc %s, voted_acc %s, counts %s' % (str(total_snr / counts), 
                                str(whole_accuracy / counts), str(voted_accuracy / counts), str(counts)))
        print(confusion_matrix, confusion_matrix / np.sum(confusion_matrix))