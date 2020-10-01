# Created on 2018/12
# Author: Junzhe Zhu & Kaituo XU

from itertools import permutations

import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
import time

EPS = 1e-8
CCE = torch.nn.CrossEntropyLoss(reduce=None)

def stable_mean(tensor, dim, keepdim=False):
    return torch.sum(tensor/tensor.size(dim), dim=dim, keepdim=keepdim)


def cal_loss(source, estimate_source, source_length, vad, lamb):
    """
    Args:
        source: [spks, T]
        estimate_source: [num_stages, spks, T]
        source_lengths: int
        vad: [num_stages, num_decoders]
    """
    # [B]
    num_stages, num_spks, T = estimate_source.size()
    assert source.size(0) == num_spks
    vad_target = torch.Tensor([num_spks] * num_stages).long().to(vad.get_device())
    vad_target -= 2 # start from 0
    max_snr = cal_si_snr_with_pit(source[:, :source_length], estimate_source[:, :, :source_length])
    snrloss = 0 - max_snr
    vadloss = CCE(vad, vad_target)
    acc = (torch.argmax(vad, dim=1) == vad_target).float()
    return snrloss + vadloss * lamb, snrloss, acc #, estimate_source, reorder_estimate_source


def cal_si_snr_with_pit(source, estimate_source, allow_extra_estimates=False):
    """Calculate SI-SNR with PIT training.
    Args:
        source: [C, T]
        estimate_source: [num_stages, C, T]
    """
    if allow_extra_estimates:
        assert source.size(0) <= estimate_source.size(1) and source.size(1) == estimate_source.size(2)
    else:
        assert source.size() == estimate_source.size()[1:]
    num_stages, C, num_samples = estimate_source.size()

    max_snr = torch.zeros(num_stages).to(source.get_device())

    # Step 1. Zero-mean norm
    mean_target = torch.mean(source, dim=1, keepdim=True) # [C, 1]
    mean_estimate = torch.mean(estimate_source, dim=2, keepdim=True) # [num_stages, C, 1]
    zero_mean_target = source - mean_target # [C, T]
    zero_mean_estimate = estimate_source - mean_estimate # [num_stages, C, T]

    # Step 2. SI-SNR with PIT
    # reshape to use broadcast
    s_target = zero_mean_target.unsqueeze(0).unsqueeze(0)  # [1, 1, C, T]
    s_estimate = zero_mean_estimate.unsqueeze(2)  # [num_stages, C, 1, T]
    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = stable_mean(s_estimate * s_target, dim=3, keepdim=True)  # [num_stages, C, C, 1]
    s_target_energy = stable_mean(s_target ** 2, dim=3, keepdim=True) + EPS  # [num_stages, 1, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [num_stages, C, C, T]
    # e_noise = s' - s_target
    e_noise = s_estimate - pair_wise_proj  # [num_stages, C, C, T]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = stable_mean(pair_wise_proj ** 2, dim=3) / (stable_mean(e_noise ** 2, dim=3) + EPS) # [num_stages, C, C]
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [num_stages, C, C]

    for stage_idx in range(num_stages):
        row_idx, col_idx = linear_sum_assignment(-pair_wise_si_snr[stage_idx].detach().cpu())
        max_snr[stage_idx] = pair_wise_si_snr[stage_idx][row_idx, col_idx].mean()

    return max_snr

if __name__ == "__main__":
    torch.manual_seed(123)
    num_stages = 6
    C = 3
    T = 32000
    source_length = 20000
    estimate_source = torch.randn(num_stages, C, T).cuda(1)
    source = torch.randn(C, T).cuda(1)
    source = estimate_source[0] + torch.randn(estimate_source[0].size()).cuda(1) / 2
    estimate_source, source = estimate_source[..., :28000], source[..., :28000]
    vad = torch.zeros([num_stages, 4]).cuda(1)
    vad[[0, 1, 2, 3, 4, 5], [1, 0, 3, 2, 1, 2]] = 1
    loss, snr, acc = cal_loss(source, estimate_source, source_length, vad, 0.5)
    print('loss', loss)
    print('snr', snr)
    print('acc', acc)

