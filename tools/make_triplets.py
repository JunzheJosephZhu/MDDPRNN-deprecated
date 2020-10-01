"""
Author: Junzhe Zhu
This script makes triplets for training speaker embedding using triplet loss
"""
from glob import glob
from sphfile import SPHFile
import torch
import numpy as np
import pandas as pd
files = list(glob("/ws/ifp-10_3/hasegawa/junzhez2/Variable_Speaker_Model/egs/**/*.wv1", recursive = True)) # wv1 and wv2 are the same audio
files.sort()
np.random.seed(12345)

spkr2files = {} # a dictionary that maps speakers to their files, for easier finding
for filename in files:
    spkr = SPHFile(filename).format['speaker_id']
    if spkr not in spkr2files:
        spkr2files[spkr] = [filename.replace('wv1', 'wav')]
    else:
        spkr2files[spkr].append(filename.replace('wv1', 'wav'))
for spkr, filenames in spkr2files.items():
    print(spkr, len(filenames))
spkrs = list(spkr2files.keys())
spkrs.sort()
print(len(spkrs)) # total of 132 speakers

## trainset
train_csv = [] # list of csv rows, for faster appending
train_spkrs = spkrs[:100] # make the first 100 speakers trainspeakers
# for each speaker pair, have 20 positive examples, and 20 negative examples
for spkr1 in train_spkrs: # anchor & positive speaker
    for spkr2 in train_spkrs: # negative speaker
        if spkr1 == spkr2:
            continue
        anchor_indices = np.arange(len(spkr2files[spkr1]))
        positive_indices = np.arange(len(spkr2files[spkr1]))
        negative_indices = np.arange(len(spkr2files[spkr2]))
        # randomly select anchor, positive, and negativie segments 
        np.random.shuffle(anchor_indices)
        np.random.shuffle(positive_indices)
        np.random.shuffle(negative_indices)
        for i in range(min(20, len(positive_indices), len(negative_indices))):
            row = {'anchor_spkr':spkr1, 'other_spkr':spkr2, 'anchor_audio':spkr2files[spkr1][anchor_indices[i]], 
            'positive_audio':spkr2files[spkr1][positive_indices[i]], 'negative_audio':spkr2files[spkr2][negative_indices[i]]}
            train_csv.append(row)
train_csv = pd.DataFrame(train_csv).sample(frac = 1, random_state = 12345) # randomly shuffle the csv
train_csv.to_csv('/ws/ifp-10_3/hasegawa/junzhez2/Variable_Speaker_Model/csv/train_triplets.csv', index = False) # save to csv

## testset
test_csv = []
test_spkrs = spkrs[100:]
for spkr1 in test_spkrs:
    for spkr2 in test_spkrs:
        if spkr1 == spkr2:
            continue
        anchor_indices = np.arange(len(spkr2files[spkr1]))
        positive_indices = np.arange(len(spkr2files[spkr1]))
        negative_indices = np.arange(len(spkr2files[spkr2]))
        np.random.shuffle(anchor_indices)
        np.random.shuffle(positive_indices)
        np.random.shuffle(negative_indices)
        for i in range(min(20, len(positive_indices), len(negative_indices))):
            row = {'anchor_spkr':spkr1, 'other_spkr':spkr2, 'anchor_audio':spkr2files[spkr1][anchor_indices[i]], 
            'positive_audio':spkr2files[spkr1][positive_indices[i]], 'negative_audio':spkr2files[spkr2][negative_indices[i]]}
            test_csv.append(row)
test_csv = pd.DataFrame(test_csv).sample(frac = 1, random_state = 12345)
test_csv.to_csv('/ws/ifp-10_3/hasegawa/junzhez2/Variable_Speaker_Model/csv/test_triplets.csv', index = False)

