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
from duplicate_snr import cat_sources
from tqdm import tqdm
from config6 import kernel_size, enc, bottleneck, hidden, num_layers, K, num_spks, mul, cat, norm, rnn_type, dropout, maxlen, minlen
from scipy.io.wavfile import write

torch.manual_seed(0)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True

test_json = ["2spkr_json/tt",
            "3spkr_json/tt",
            "4spkr_json/tt",
            "5spkr_json/tt"]

model_path = "models/config6.pth"
device = 2
# chop settings
sr = 8000
seglen = int(maxlen * sr)
minlen = int(minlen * sr)

use_oracle = False
compute_signal = False
# pref = {2:-30, 3:-30, 4:-30, 5:-30}
pref = {2:-19.10105119, 3:-14.07390921, 4:-9.34538167,  5:-5.91764883}

def chop(mix):
    '''
        chop signal into chunks
        mix: [B, T]
    '''
    start = 0
    padding_len = 0
    chunks = []
    while start + minlen <= mix.shape[1]:
        end = min(start + seglen, mix.shape[1])
        chunk = mix[:, start:end]
        if chunk.shape[1] < seglen:
            padding_len = seglen - chunk.shape[1]
            padding = torch.zeros((1, padding_len)).to(chunk.get_device())
            chunk = torch.cat([chunk, padding], dim=1)
        chunks.append(chunk)
        start += minlen
    if chunks == []:
        padding_len = seglen - mix.shape[1]
        padding = torch.zeros((1, padding_len)).to(mix.get_device())
        mix = torch.cat([mix, padding], dim=1)
        chunks = [mix]
    return chunks, padding_len

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
    confusion_matrix = np.zeros((num_spks - 1, len(test_json))) # ylabel: [0, num_spks] xlabel: [2, 5]
    counts = np.zeros(len(test_json))

    with torch.no_grad():
        pbar = tqdm(test_loader)
        # [1, T], list of [1, T]
        for i, (mixture, sources) in enumerate(pbar):
            num_sources = len(sources)
            counts[num_sources - 2] += 1
            ilens = torch.Tensor([mixture.shape[1]]).int()
            # [1, T], [spks, T]
            mixture, sources = mixture.cuda(device), torch.cat(sources, dim=0).cuda(device)

            # run chunks
            chunks, padding_len = chop(mixture)
            chunks = torch.cat(chunks, dim=0)
            votes = np.zeros(num_spks - 1)
            if use_oracle:
                estimate_chunks, vad_list = model(chunks, torch.Tensor([num_sources] * len(chunks)), oracle=use_oracle)
                for vad_chunk in vad_list:
                    votes[vad_chunk[-1].argmax(0)] += 1
                vad_voted = np.argmax(votes, axis=0)
            else:
                estimate_chunks, vad_list = model(chunks, torch.Tensor([num_sources] * len(chunks)), oracle=use_oracle)
                for vad_chunk in vad_list:
                    votes[vad_chunk[-1].argmax(0)] += 1
                vad_voted = np.argmax(votes, axis=0)
                if compute_signal:
                    estimate_chunks, _ = model(chunks, torch.Tensor([vad_voted + 2] * len(chunks)), oracle=True)

            voted_accuracy[num_sources - 2] += vad_voted == num_sources - 2

            # [num_chunks, spks, T]
            if use_oracle:
                estimate_chunks = estimate_chunks[:, :, :num_sources] # VERY IMPORTANT
            else:
                estimate_chunks = estimate_chunks[:, :, :vad_voted + 2] # VERY IMPORTANT
            estimate_chunks = list(estimate_chunks[:, -1])
            if padding_len != 0: # cut off padding
                estimate_chunks[-1] = estimate_chunks[-1][:, :-padding_len]
            estimate_sources = estimate_chunks[0]
            for estimate_chunk in estimate_chunks[1:]:
                estimate_sources = cat_sources(estimate_sources, estimate_chunk, overlap=minlen)
            assert estimate_sources.size(1) == sources.size(1)

            # if estimate_sources.size() == sources.size():
            #     saved_sources = estimate_sources.cpu().numpy()
            #     saved_mixture = mixture[0].cpu().numpy()
            #     write(f"examples/{saved_sources.shape[0]}_mixture.wav", 8000, saved_mixture / saved_mixture.std())
            #     for i, saved_source in enumerate(saved_sources):
            #         write(f"examples/{saved_sources.shape[0]}_source_{i}.wav", 8000, saved_source / saved_source.std())



            # estimate_sources: [spks, T] sources: [spks, T]
            snr = cal_si_snr_with_pit(sources, estimate_sources.unsqueeze(0), allow_unequal_estimates=True)[0].item()
            snr += pref[num_sources] * np.abs(estimate_sources.size(0) - sources.size(0))
            snr /= max(sources.size(0), estimate_sources.size(0))
            total_snr[num_sources - 2] += snr
            confusion_matrix[estimate_sources.shape[0] - 2, sources.shape[0] - 2] += 1
            pbar.set_description('total_snr %s, voted_acc %s, counts %s' % (str(total_snr / counts), 
                                str(voted_accuracy / counts), str(counts)))
        print(confusion_matrix, '\n', confusion_matrix / np.sum(confusion_matrix))