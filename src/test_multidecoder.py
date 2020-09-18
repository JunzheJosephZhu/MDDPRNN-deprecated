#!/usr/bin/env python

# Created on 2018/12
# Author: Junzhe Zhu & Kaituo XU
import sys
sys.path.append("configs")

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

root = "/ws/ifp-10_3/hasegawa/junzhez2/Baseline_Model/dataset"
test_json = ["2spkr_json/cv",
            "3spkr_json/cv",
            "4spkr_json/cv",
            "5spkr_json/cv"]

batch_size = 1
model_path = "models/config4_best.pth"
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
    test_dataset = TestDataset(root, test_json)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
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
        pbar = tqdm(test_loader)
        for i, (mixture, sources) in enumerate(pbar):
            counts[len(sources) - 2] += 1
            ilens = torch.Tensor([mixture.shape[1]]).int()
            # [B, T], [spks, T]
            mixture, sources = mixture.cuda(device), torch.cat(sources, dim=0).cuda(device)
            # list of num_decoders, each [B, #stages, spks, T], [B, #stages, spks]
            estimate_source_list, vad_whole = model(mixture)
            accuracy_whole += (vad_whole[0][-1].argmax(0) == sources.shape[0] - 2).int().item()
            chunks = chop(mixture)
            votes = np.zeros(num_spks - 1)
            for chunk in chunks:
                chunkcount+=1 ##################
                _, vad_list = model(chunk)
                vad_chunk = vad_list[0, -1].argmax(0).item()
                accuracy_chunk += len(sources) == vad_chunk + 2 #################
                votes[vad_chunk] += 1
            vad_voted = np.argmax(votes, axis=0)
            accuracy_voted += vad_voted == sources.shape[0] - 2
            print(accuracy_whole/(i + 1), accuracy_voted/(i + 1)) #####################
            # [spks, T]
            estimate_source = estimate_source_list[vad_voted][0][-1]
            sources = sources[:, :estimate_source.shape[1]] # cut off extra samples
            total_accuracy[sources.shape[0] - 2] += estimate_source.shape == sources.shape
            if estimate_source.shape >= sources.shape:
                snr = cal_si_snr_with_pit(sources.unsqueeze(0), estimate_source.unsqueeze(0), [sources.shape[1]], debug=False, log_vars=False)[0].item()
            else:
                snr = duplicate_snr(sources, estimate_source)
            total_snr[sources.shape[0] - 2] += snr
            confusion_matrix[estimate_source.shape[0] - 2, sources.shape[0] - 2] += 1
            pbar.set_description('total_snr %s, total_acc %s, counts %s' % (str(total_snr/ counts), 
                                    str(total_accuracy/counts), str(counts)))
        print(confusion_matrix, confusion_matrix/np.sum(confusion_matrix))