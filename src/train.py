#!/usr/bin/env python

# Created on 2018/12
# Author: Junzhe Zhu & Kaituo XU

import argparse

import torch

from data import MixtureDataset, _collate_fn
from solver import Solver
from model import MulCatModel
import time
torch.manual_seed(0)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True

train_dir = "/ws/ifp-10_3/hasegawa/junzhez2/Variable_Speaker_Model/csv/mixtures/mix_2_spk_tr.txt"
valid_dir = "/ws/ifp-10_3/hasegawa/junzhez2/Variable_Speaker_Model/csv/mixtures/mix_2_spk_cv.txt"
sample_rate = 8000
maxlen = 27.5
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
num_workers = 16
lr = 1e-3
momentum = 0.0
l2 = 0.0 # weight decya
save_folder = "/ws/ifp-10_3/hasegawa/junzhez2/Variable_Speaker_Model/models"
checkpoint = 1
continue_from = save_folder+"/epoch90.pth.tar"
model_path = "best.pth"
print_freq = 10
log_dir = "/ws/ifp-10_3/hasegawa/junzhez2/Variable_Speaker_Model/runs/"+time.strftime("%Y%m%d-%H%M%S")
comment = 'there are two epochs of OOM'



if __name__ == '__main__':
    tr_dataset = MixtureDataset(train_dir, sample_rate=sample_rate, maxlen=maxlen)
    cv_dataset = MixtureDataset(valid_dir, sample_rate=sample_rate)  # -1 -> use full audio
    tr_loader = torch.utils.data.DataLoader(tr_dataset, batch_size=batch_size, collate_fn=_collate_fn, shuffle=shuffle, num_workers=num_workers)
    cv_loader = torch.utils.data.DataLoader(cv_dataset, batch_size=batch_size, collate_fn=_collate_fn, shuffle=shuffle, num_workers=num_workers)
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}
    # model
    model = torch.nn.DataParallel(MulCatModel(N=N, L=L, K=K, P=P, H=H, B=B, C=C).cuda())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    # solver
    solver = Solver(data, model, optimizer, epochs, save_folder, checkpoint, continue_from, model_path, print_freq=print_freq,
        half_lr=half_lr, early_stop=early_stop, max_norm=max_norm, lr=lr, momentum=momentum, l2=l2, log_dir=log_dir, comment=comment)
    solver.train()
