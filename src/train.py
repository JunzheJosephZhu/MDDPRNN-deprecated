#!/usr/bin/env python

# Created on 2018/12
# Author: Junzhe Zhu & Kaituo XU
import sys
root = "/ws/ifp-10_3/hasegawa/junzhez2/Baseline_Model"
sys.path.append(root)
import argparse
parser = argparse.ArgumentParser(description='config file')
parser.add_argument('--config', type=str, default='config4', help='config file')
args = parser.parse_args()
import torch
from torch.utils.data import WeightedRandomSampler
from data import MixtureDataset, _collate_fn
from solver import Solver
exec('from configs.' + args.config + ' import *')
print('loading ' + args.config)
torch.manual_seed(0)

root_dataset = os.path.join(root, "dataset")
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
if multidecoder:
    from model_multidecoder import Dual_RNN_model
else:
    if use_onoff:
        from model_mulcat import Dual_RNN_model
    else:
        from model_rnn import Dual_RNN_model


if __name__ == '__main__':
    tr_dataset = MixtureDataset(root_dataset, tr_json, seglen=maxlen, minlen=minlen)
    cv_dataset = MixtureDataset(root_dataset, val_json, seglen=maxlen, minlen=minlen)
    samples_weights = tr_dataset.example_weights
    sampler = WeightedRandomSampler(weights=samples_weights,
                                    num_samples=len(samples_weights),
                                    replacement=True)
    tr_loader = torch.utils.data.DataLoader(tr_dataset, batch_size=batch_size, collate_fn=_collate_fn,
                                            sampler=sampler,
				                            shuffle=shuffle, num_workers=8)
    cv_loader = torch.utils.data.DataLoader(cv_dataset, batch_size=batch_size, collate_fn=_collate_fn, shuffle=shuffle, num_workers=8)
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}
    # model
    model = torch.nn.DataParallel(Dual_RNN_model(enc, bottleneck, hidden, kernel_size=kernel_size, rnn_type=rnn_type, norm=norm, dropout=dropout, bidirectional=True, num_layers=num_layers, K=K, num_spks=num_spks, multiloss=multiloss, mulcat=(mul, cat)), device_ids=device_ids)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    # solver
    solver = Solver(data, model, optimizer, epochs, save_folder, checkpoint, continue_from, model_path, print_freq=print_freq,
        half_lr=half_lr, early_stop=early_stop, max_norm=max_norm, lr=lr, lr_override=lr_override, momentum=momentum, l2=l2, log_dir=log_dir, lamb=lamb, decay_period=decay_period, config=config, multidecoder=multidecoder)
    solver.train()
