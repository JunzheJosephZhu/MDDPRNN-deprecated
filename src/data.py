"""
Dataset classes for variable number of speakers
Author: Junzhe Zhu
"""
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
        self.examples = random.sample(self.examples, len(self.examples))

        # Count.
        example_source_files_len = [len(tmp['sourcefiles'] )for tmp in self.examples]
        unique, counts = np.unique(np.array(example_source_files_len), return_counts=True)
        self.example_weights =[]
        for tmp in example_source_files_len:
            self.example_weights.append(1./counts[tmp-2])
        self.example_weights = torch.Tensor(self.example_weights)
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
        
if __name__ == "__main__":
    root = "/ws/ifp-10_3/hasegawa/junzhez2/Baseline_Model/dataset"
    tr_json = ["2spkr_json/tr/",
                "3spkr_json/tr/",
                "4spkr_json/tr/",
                "5spkr_json/tr/"]
    val_json = ["2spkr_json/cv/",
                "3spkr_json/cv/",
                "4spkr_json/cv/",
                "5spkr_json/cv/"]
    test_json = ["2spkr_json/tt",
                "3spkr_json/tt",
                "4spkr_json/tt",
                "5spkr_json/tt"]
    dataset = MixtureDataset(root, tr_json)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, collate_fn=_collate_fn)
    print(len(dataset))
    for mixtures, ilens, sources_list in tqdm(dataloader):
        start = time()
        print(mixtures.shape, ilens.shape, [len(sources) for sources in sources_list])
        print(time() - start)
