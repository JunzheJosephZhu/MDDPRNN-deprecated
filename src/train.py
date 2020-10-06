#!/usr/bin/env python

# Created on 2018/12
# Author: Junzhe Zhu & Kaituo XU
import sys
import importlib
import os
root = "/ws/ifp-10_3/hasegawa/junzhez2/Baseline_Model"
sys.path.append(root)
import argparse
parser = argparse.ArgumentParser(description='config file')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--lroverride', action='store_true', help='override learning rate?')

args = parser.parse_args()
import torch
from torch.utils.data import WeightedRandomSampler
from data import MixtureDataset, _collate_fn
from solver import Solver
c = importlib.import_module('configs.' + args.config)
print('loading ' + args.config)

torch.manual_seed(0)

root_dataset = os.path.join(root, "dataset")
tr_json = ["2spkr_json/tr/",
            "3spkr_json/tr/",
            "4spkr_json/tr/",
            "5spkr_json/tr/"][:c.num_spks-1]
val_json = ["2spkr_json/cv/",
            "3spkr_json/cv/",
            "4spkr_json/cv/",
            "5spkr_json/cv/"][:c.num_spks-1]
test_json = ["2spkr_json/tt",
            "3spkr_json/tt",
            "4spkr_json/tt",
            "5spkr_json/tt"][:c.num_spks-1]
if c.multidecoder:
    from model_multidecoder import Dual_RNN_model
else:
    if c.use_onoff:
        from model_mulcat import Dual_RNN_model
    else:
        from model_rnn import Dual_RNN_model


if __name__ == '__main__':
    # data
    tr_dataset = MixtureDataset(root_dataset, tr_json, seglen=c.maxlen, minlen=c.minlen)
    cv_dataset = MixtureDataset(root_dataset, val_json, seglen=c.maxlen, minlen=c.minlen)
    samples_weights = tr_dataset.example_weights
    sampler = WeightedRandomSampler(weights=samples_weights,
                                    num_samples=len(samples_weights),
                                    replacement=True)
    tr_loader = torch.utils.data.DataLoader(tr_dataset, batch_size=c.batch_size, collate_fn=_collate_fn,
                                            sampler=sampler,
				                            shuffle=c.shuffle, num_workers=8)
    cv_loader = torch.utils.data.DataLoader(cv_dataset, batch_size=c.batch_size, collate_fn=_collate_fn, shuffle=c.shuffle, num_workers=8)
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}

    # model
    model = torch.nn.DataParallel(Dual_RNN_model(c.enc, c.bottleneck, c.hidden, kernel_size=c.kernel_size, rnn_type=c.rnn_type, norm=c.norm, dropout=c.dropout, bidirectional=True, num_layers=c.num_layers, K=c.K, num_spks=c.num_spks, multiloss=c.multiloss, mulcat=(c.mul, c.cat)), device_ids=c.device_ids)
    model = model.cuda(c.device_ids[0])

    # optimizer
    encoder_params = {'params': model.module.encoder.parameters(), 'lr': c.lr}
    separation_params = {'params': model.module.separation.parameters(), 'lr': c.lr}
    decoder_params = {'params': model.module.decoder.decoders.parameters(), 'lr': c.lr * 4}
    vad_params = {'params': model.module.decoder.vad.parameters(), 'lr': c.lr}
    optimizer = torch.optim.Adam([encoder_params, separation_params, decoder_params, vad_params])

    # solver
    solver = Solver(data, model, optimizer, c.epochs, c.save_folder, c.checkpoint, c.continue_from, c.model_path, print_freq=c.print_freq,
        early_stop=c.early_stop, max_norm=c.max_norm, lr=c.lr, lr_override=args.lroverride, log_dir=c.log_dir, \
        lamb=c.lamb, decay_period=c.decay_period, config=c.config_name, multidecoder=c.multidecoder, decay=c.decay)
    solver.train()
