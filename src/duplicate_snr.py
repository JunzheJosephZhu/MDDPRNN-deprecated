import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

def duplicate_snr(sources, variable_est):
    '''
        sources: [spks, T]
        variable_est: [spks, T]
    '''
    EPS = 1e-8
    assert sources.shape[0] > variable_est.shape[0]
    zero_mean_target = sources - torch.mean(sources, dim=1, keepdim=True)
    zero_mean_estimate = variable_est - torch.mean(variable_est, dim=1, keepdim=True)
    s_target = torch.unsqueeze(zero_mean_target, dim=0)  # [1, C, T]
    s_estimate = torch.unsqueeze(zero_mean_estimate, dim=1)  # [C_est, 1, T]
    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = torch.mean(s_estimate * s_target, dim=2, keepdim=True)  # [C_est, C, 1]
    s_target_energy = torch.mean(s_target ** 2, dim=2, keepdim=True) + EPS  # [1, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [C_est, C, T], put the target energy term somewhere else
    # e_noise = s' - s_target
    e_noise = s_estimate - pair_wise_proj  # [C_est, C, T]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = torch.mean(pair_wise_proj ** 2, dim=2) / (torch.mean(e_noise ** 2, dim=2) + EPS)
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [C_est, C]

    row_idx, col_idx = linear_sum_assignment(-pair_wise_si_snr.detach().cpu())
    max_snrs = pair_wise_si_snr[row_idx, col_idx].tolist()
    unmatched_sources = [i for i in range(len(sources)) if i not in col_idx]
    for source_idx in unmatched_sources:
        max_snrs.append(pair_wise_si_snr[:, source_idx].max().item())
    return np.mean(max_snrs)

if __name__ == "__main__":
    torch.manual_seed(123)
    testcase = 0
    if testcase == 0: # answer is around 80-90 ish
        C, T = 5, 12
        # fake data
        source = torch.randint(5, (C, T)).float()
        #estimate_source = torch.randint(4, (B, C, T)).float()
        estimate_source = source[[0, 2, 3]]
        #print(source, '\n', estimate_source)
        #print('source shape', source.size())
    print(duplicate_snr(source, estimate_source))