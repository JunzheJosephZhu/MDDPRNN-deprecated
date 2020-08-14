#!/usr/bin/env python

# Created on 2018/12
# Author: Junzhe Zhu & Kaituo XU
import sys
sys.path.append("configs")
import argparse

import torch

from data import MixtureDataset, _collate_fn
from solver import Solver
from c1 import *
torch.manual_seed(0)

root = "/ws/ifp-10_3/hasegawa/junzhez2/Baseline_Model/dataset"
tr_json = ["2spkr_json/tr/",
            "3spkr_json/tr/",
            "4spkr_json/tr/",
            "5spkr_json/tr/"][:num_spks-1]
val_json = ["2spkr_json/cv/",
            "3spkr_json/cv/",
            "4spkr_json/cv/",
            "5spkr_json/cv/"][:num_spks-1]
test_json = ["2spkr_json/tt",
            "3spkr_json/tt",
            "4spkr_json/tt",
            "5spkr_json/tt"][:num_spks-1]

if use_mulcat:
    from model_mulcat import Dual_RNN_model
else:
    from model import Dual_RNN_model


if __name__ == '__main__':
    tr_dataset = MixtureDataset(root, tr_json, seglen=maxlen, minlen=minlen)
    cv_dataset = MixtureDataset(root, val_json, seglen=maxlen, minlen=minlen)
    tr_loader = torch.utils.data.DataLoader(tr_dataset, batch_size=batch_size, collate_fn=_collate_fn, shuffle=shuffle)
    cv_loader = torch.utils.data.DataLoader(cv_dataset, batch_size=batch_size, collate_fn=_collate_fn, shuffle=shuffle)
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}
    # model
    model = torch.nn.DataParallel(Dual_RNN_model(enc, bottleneck, hidden, bidirectional=True, num_layers=num_layers, K=250, num_spks=num_spks, multiloss=multiloss).cuda())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    # solver
    solver = Solver(data, model, optimizer, epochs, save_folder, checkpoint, continue_from, model_path, print_freq=print_freq,
        half_lr=half_lr, early_stop=early_stop, max_norm=max_norm, lr=lr, momentum=momentum, l2=l2, log_dir=log_dir, comment=comment)
    solver.train()
