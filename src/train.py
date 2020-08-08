#!/usr/bin/env python

# Created on 2018/12
# Author: Junzhe Zhu & Kaituo XU

import argparse

import torch

from data import MixtureDataset, _collate_fn
from solver import Solver
from model import Dual_RNN_model
import time
torch.manual_seed(0)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True

root = "/ws/ifp-10_3/hasegawa/junzhez2/Baseline_Model/dataset"
tr_json = ["2spkr_json/tr/",
            "3spkr_json/tr/",
            "4spkr_json/tr/",
            "5spkr_json/tr/"][:1]
val_json = ["2spkr_json/cv/",
            "3spkr_json/cv/",
            "4spkr_json/cv/",
            "5spkr_json/cv/"][:1]
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
epochs = 128
half_lr = True
early_stop = True
max_norm = 5
shuffle = False
batch_size = 4
lr = 1e-3
momentum = 0.0
l2 = 0.0 # weight decya
save_folder = "/ws/ifp-10_3/hasegawa/junzhez2/Baseline_Model/models"
checkpoint = 1
continue_from = save_folder+"dummy"#+"/last.pth"
model_path = "best.pth"
print_freq = 10
comment = 'changed mixing to matlab-using hungarian algorithm and variable speaker dataset, but still training on 2spkrs'
log_dir = "/ws/ifp-10_3/hasegawa/junzhez2/Baseline_Model/runs/"+time.strftime("%Y%m%d-%H%M%S")+comment


if __name__ == '__main__':
    tr_dataset = MixtureDataset(root, tr_json)
    cv_dataset = MixtureDataset(root, val_json)
    tr_loader = torch.utils.data.DataLoader(tr_dataset, batch_size=batch_size, collate_fn=_collate_fn, shuffle=shuffle)
    cv_loader = torch.utils.data.DataLoader(cv_dataset, batch_size=batch_size, collate_fn=_collate_fn, shuffle=shuffle)
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}
    # model
    model = torch.nn.DataParallel(Dual_RNN_model(256, 64, 128, bidirectional=True, num_layers=6, K=250).cuda())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    # solver
    solver = Solver(data, model, optimizer, epochs, save_folder, checkpoint, continue_from, model_path, print_freq=print_freq,
        half_lr=half_lr, early_stop=early_stop, max_norm=max_norm, lr=lr, momentum=momentum, l2=l2, log_dir=log_dir, comment=comment)
    solver.train()
