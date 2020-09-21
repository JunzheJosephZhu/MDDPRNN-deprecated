# Created on 2018/12
# Author: Junzhe Zhu & Kaituo XU

from itertools import permutations

import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
import time

EPS = 1e-8

def stable_mean(tensor, dim, keepdim=False):
    return torch.sum(tensor/tensor.size(dim), dim=dim, keepdim=keepdim)


def cal_loss(source, estimate_source, source_lengths, vad, lamb, debug=False, log_vars=False):
    """
    Args:
        source: list of B, each [spks, T]
        estimate_source: list of B, each [spks, T]
        source_lengths: [B]
        vad: [B, max_spks - 1]
    """
    # [B]
    vad_target = torch.Tensor([len(s) for s in source]).long().to(vad.get_device())
    vad_target -= 2 # start from 0
    max_snr = cal_si_snr_with_pit(source, estimate_source, source_lengths, debug, log_vars)
    snrloss = 0 - torch.mean(max_snr)
    vadloss = torch.nn.CrossEntropyLoss()(vad, vad_target)
    acc = torch.mean((torch.argmax(vad, dim = 1) == vad_target).float())
    return snrloss + vadloss * lamb, snrloss, acc #, estimate_source, reorder_estimate_source
    #reorder_estimate_source = reorder_source(estimate_source, perms, max_snr_idx)


def cal_si_snr_with_pit(source, estimate_source, source_lengths, debug, log_vars):
    """Calculate SI-SNR with PIT training.
    Args:
        source: list of [B], each item is [C, T]
        estimate_source: [B, 5, T]
        source_lengths: [B], each item is between [0, T]
    """
    assert len(source) == len(estimate_source)
    assert source[0].size(1) == estimate_source[0].size(1)
    B, (_, T) = len(estimate_source), estimate_source[0].size()

    # mask padding position along T
    mask = get_mask(estimate_source, source_lengths) # [B, 1, T]
    estimate_source = [estimate_source[i] * mask[i] for i in range(B)]

    max_snr = torch.zeros(B)
    for batch_idx in range(B):
        # source[batch_idx]: [C, T]
        # estimate_source[batch_idx]: [5, T]
        # Step 1. Zero-mean norm
        C = source[batch_idx].size(0)
        num_samples = source_lengths[batch_idx]
        mean_target = torch.sum(source[batch_idx], dim=1, keepdim=True) / num_samples # [C, 1]
        mean_estimate = torch.sum(estimate_source[batch_idx], dim=1, keepdim=True) / num_samples # [5, 1]
        zero_mean_target = source[batch_idx] - mean_target
        zero_mean_estimate = estimate_source[batch_idx] - mean_estimate
        # mask padding position along T
        zero_mean_target *= mask[batch_idx]
        zero_mean_estimate *= mask[batch_idx]

        # Step 2. SI-SNR with PIT
        # reshape to use broadcast
        s_target = torch.unsqueeze(zero_mean_target, dim=0)  # [1, C, T]
        s_estimate = torch.unsqueeze(zero_mean_estimate, dim=1)  # [5, 1, T]
        # s_target = <s', s>s / ||s||^2
        pair_wise_dot = stable_mean(s_estimate * s_target, dim=2, keepdim=True)  # [5, C, 1]
        s_target_energy = stable_mean(s_target ** 2, dim=2, keepdim=True) + EPS  # [1, C, 1]
        pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [5, C, T], put the target energy term somewhere else
        # e_noise = s' - s_target
        e_noise = s_estimate - pair_wise_proj  # [5, C, T]
        # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
        pair_wise_si_snr = stable_mean(pair_wise_proj ** 2, dim=2) / (stable_mean(e_noise ** 2, dim=2) + EPS)
        pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C, C]

        if not debug:
            if log_vars:
                np.savetxt('log/pair_wise_si_snr.txt', pair_wise_si_snr.detach().cpu().numpy())
                np.save('log/source.npy', [s.detach().cpu().numpy() for s in source])
                np.save('log/estimate_source.npy', estimate_source.detach().cpu().numpy())
                np.save('log/source_lengths.npy', source_lengths.detach().cpu().numpy())
        else:
            print('source_len', source_lengths[batch_idx])
            #print('source', source[batch_idx])
            #print('estimate_source', estimate_source[batch_idx])
            print('pair_wise_dot', pair_wise_dot)
            #print('s_target', s_target)
            print('s_target_energy', s_target_energy)
            #print('pair_wise_proj', pair_wise_proj)
            #print('e_noise', e_noise)
            print('si-snr', pair_wise_si_snr)
            print('-'*30, 'batch_idx', batch_idx, '-'*30)

        row_idx, col_idx = linear_sum_assignment(-pair_wise_si_snr.detach().cpu())
        max_snr[batch_idx] = pair_wise_si_snr[row_idx, col_idx].mean()
    if debug:
        print('max_snr', max_snr)
    return max_snr


def reorder_source(source, perms, max_snr_idx):
    """
    Args:
        source: [B, C, T]
        perms: [C!, C], permutations
        max_snr_idx: [B], each item is between [0, C!)
    Returns:
        reorder_source: [B, C, T]
    """
    B, C, *_ = source.size()
    # [B, C], permutation whose SI-SNR is max of each utterance
    # for each utterance, reorder estimate source according this permutation
    max_snr_perm = torch.index_select(perms, dim=0, index=max_snr_idx)
    # print('max_snr_perm', max_snr_perm)
    # maybe use torch.gather()/index_select()/scatter() to impl this?
    reorder_source = torch.zeros_like(source)
    for b in range(B):
        for c in range(C):
            reorder_source[b, c] = source[b, max_snr_perm[b][c]]
    return reorder_source


def get_mask(source, source_lengths):
    """
    Args:
        source: [B, C, T]
        source_lengths: [B]
    Returns:
        mask: [B, 1, T]
    """
    B, (_, T) = len(source), source[0].size()
    mask = source[0].new_ones((B, 1, T))
    for i in range(B):
        mask[i, :, source_lengths[i]:] = 0
    return mask


if __name__ == "__main__":
    torch.manual_seed(123)
    estimate_source = [torch.rand((3, 100)).cuda(), torch.rand((2, 100)).cuda()]
    source = [torch.rand((3, 100)).cuda(), torch.rand((2, 100)).cuda()]
    source_lengths = [100, 100]
    vad = torch.zeros([2, 4])
    vad[[0, 1], [1, 0]] = 1
    loss, snr, acc = cal_loss(source, estimate_source, source_lengths, vad, 0.5, debug=True, log_vars=False)
    print('loss', loss)
    print('snr', snr)
    print('acc', acc)

