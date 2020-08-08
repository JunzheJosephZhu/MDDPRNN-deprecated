# Created on 2018/12
# Author: Junzhe Zhu & Kaituo XU

from itertools import permutations

import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

EPS = 1e-8


def cal_loss(source, estimate_source, source_lengths):
    """
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, 5, T]
        source_lengths: [B]
    """
    max_snr, onoff_target = cal_si_snr_with_pit(source, estimate_source, source_lengths)
    loss = 0 - torch.mean(max_snr)
    #reorder_estimate_source = reorder_source(estimate_source, perms, max_snr_idx)
    return loss, onoff_target#, estimate_source, reorder_estimate_source


def cal_si_snr_with_pit(source, estimate_source, source_lengths):
    """Calculate SI-SNR with PIT training.
    Args:
        source: list of [B], each item is [C, T]
        estimate_source: [B, 5, T]
        source_lengths: [B], each item is between [0, T]
    """
    assert len(source) == len(estimate_source)
    assert source[0].size(1) == estimate_source.size(2)

    B, max_C, T = estimate_source.size()

    # mask padding position along T
    mask = get_mask(estimate_source, source_lengths) # [B, 1, T]
    estimate_source *= mask

    max_snr = torch.zeros(B)
    onoff_target = torch.zeros(B, max_C)

    for batch_idx in range(B):
        # source[batch+idx]: [C, T]
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
        pair_wise_dot = torch.sum(s_estimate * s_target, dim=2, keepdim=True)  # [5, C, 1]
        s_target_energy = torch.sum(s_target ** 2, dim=2, keepdim=True) + EPS  # [1, C, 1]
        pair_wise_proj = pair_wise_dot * s_target   # [5, C, T], put the target energy term somewhere else
        # e_noise = s' - s_target
        e_noise = s_estimate - pair_wise_proj / s_target_energy  # [5, C, T]
        # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
        # pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + EPS)
        # pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C, C]
        log_sig_e = torch.log10(torch.sum(pair_wise_proj ** 2, dim=2) + EPS) - 2 * torch.log10(torch.sum(s_target_energy, dim=2) + EPS)
        log_noi_e = torch.log10(torch.sum(e_noise**2, dim=2) + EPS)
        pair_wise_si_snr = 10 * (log_sig_e - log_noi_e) # [5, C]
        assert not torch.isnan(log_sig_e).bool().any(), 'signal energy infinity '+str(torch.max(estimate_source))+str(torch.min(estimate_source))
        assert not torch.isnan(log_noi_e).bool().any(), 'signal energy infinity '+str(torch.max(estimate_source))+str(torch.min(estimate_source))
        
        row_idx, col_idx = linear_sum_assignment(-pair_wise_si_snr.detach().cpu())
        max_snr[batch_idx] = pair_wise_si_snr[row_idx, col_idx].sum()/C
        onoff_target[batch_idx][row_idx] = 1
    # # Get max_snr of each utterance
    # # permutations, [C!, C]
    # perms = source.new_tensor(list(permutations(range(C))), dtype=torch.long)
    # # one-hot, [C!, C, C]
    # index = torch.unsqueeze(perms, 2)
    # perms_one_hot = source.new_zeros((*perms.size(), C)).scatter_(2, index, 1)
    # # [B, C!] <- [B, C, C] einsum [C!, C, C], SI-SNR sum of each permutation
    # snr_set = torch.einsum('bij,pij->bp', [pair_wise_si_snr, perms_one_hot])
    # max_snr_idx = torch.argmax(snr_set, dim=1)  # [B]
    # # max_snr = torch.gather(snr_set, 1, max_snr_idx.view(-1, 1))  # [B, 1]
    # max_snr, _ = torch.max(snr_set, dim=1, keepdim=True)
    return max_snr, onoff_target


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
    B, _, T = source.size()
    mask = source.new_ones((B, 1, T))
    for i in range(B):
        mask[i, :, source_lengths[i]:] = 0
    return mask


if __name__ == "__main__":
    torch.manual_seed(123)
    testcase = 0
    if testcase == 0:
        B, C, T = 2, 5, 12
        # fake data
        estimate_source = torch.randint(5, (B, C, T)).float()
        #estimate_source = torch.randint(4, (B, C, T)).float()
        source = [estimate_source[0, [2, 1, 4, 3, 0]], estimate_source[1, [3, 2]]]
        #print(source, '\n', estimate_source)
        #print('source shape', source.size())
        source[1][:, -3:] = 0
        estimate_source[1, :, -3:] = 0
        source_lengths = torch.LongTensor([T, T-3])


    elif testcase == 1:
        B, C, T = 2, 3, 12
        # fake data
        source = torch.randint(4, (B, C, T)).float()
        estimate_source = torch.randint(4, (B, C, T)).float()
        source[1, :, -3:] = 0
        estimate_source[1, :, -3:] = 0
        source_lengths = torch.LongTensor([T, T-3])
        #print('source', source)
        #print('estimate_source', estimate_source)
        #print('source_lengths', source_lengths)
        
    loss, onoff_target = cal_loss(source, estimate_source, source_lengths)
    print('loss', loss)
    print('on/off target', onoff_target)
