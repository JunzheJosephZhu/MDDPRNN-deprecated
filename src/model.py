# Created on 2020/7
# Author: Junzhe Zhu, Kaituo XU, Jusper Lee

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils import overlap_and_add, pad_segment, pad_to_integer

EPS = 1e-8


class MulCatModel(nn.Module):
    def __init__(self, N = 64, L = 16, K = 100, P = 50, H = 128, B = 6, C = 2):
        """
        Args:
            N: Number of latent channels
            L: Length of the encoder window
            L//2: Encoder windowing stride
            K: Chunk size
            P: Chunk stride
            S: Number of Chunks
            H: Number of LSTM hidden channels
            B: Number of Blocks
            C: Number of speakers
        """
        super().__init__()
        # Hyper-parameter
        self.N, self.L, self.K, self.P, self.H, self.B, self.C = N, L, K, P, H, B, C
        # Components
        self.encoder = Encoder(L, N)
        self.separator = nn.ModuleList([MulCat_Block(N, H, C) for _ in range(B)])
        self.decoder = Decoder(N, L, K, P, C)
        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, mixture):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            est_source: [M, C, T]
        """
        outputs = []
        mixture_w = self.encoder(mixture)
        T_origin = mixture.size(-1)
        separator_input = pad_segment(mixture_w, self.K, self.P)
        for block in self.separator:
            separator_input = block(separator_input)
            est_source = self.decoder(mixture_w, separator_input)
            est_source = est_source[..., :T_origin] # undo padding
            outputs.append(est_source)

        # T changed after conv1d in encoder, fix it here
        
        return outputs

    @classmethod
    def load_model(cls, path): # obsolete, might fix later
        # Load to CPU
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls.load_model_from_package(package)
        return model

    @classmethod # obsolete, might fix later
    def load_model_from_package(cls, package):
        model = cls(package['N'], package['L'], package['B'], package['H'],
                    package['P'], package['X'], package['R'], package['C'],
                    norm_type=package['norm_type'], causal=package['causal'],
                    mask_nonlinear=package['mask_nonlinear'])
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None, val_no_impv=None, random_state=None):
        package = {
            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
            package['val_no_impv'] = val_no_impv
            package['random_state'] = random_state
        return package


class Encoder(nn.Module):
    """Estimation of the nonnegative mixture weight by a 1-D conv layer.
    """
    def __init__(self, L, N):
        super(Encoder, self).__init__()
        # Hyper-parameter
        self.L, self.N = L, N
        self.stride = L//2
        # Components
        # 50% overlap
        self.conv1d_U = nn.Conv1d(1, N, kernel_size=L, stride=self.stride, bias=False)

    def forward(self, mixture):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            mixture_w: [M, N, K, S], where T1 = (T-L)/(L/2)+1 = 2T/L-1, S is #frames, K is frame length
        """
        mixture = pad_to_integer(mixture, self.L, self.stride)
        mixture = torch.unsqueeze(mixture, 1)  # [M, 1, T]
        mixture_w = self.conv1d_U(mixture) # [M, N, T1] # no-relu, as in original paper
        return mixture_w



        


class Decoder(nn.Module): # there's a disagreement between variable speaker separation paper and DPRNN implementation: do we transform from N to N*C, or S to S*C?
    def __init__(self, N, L, K, P, C):
        super(Decoder, self).__init__()
        # Hyper-parameter
        self.N, self.L, self.K, self.P, self.C = N, L, K, P, C
        # Components
        self.prelu = nn.PReLU()
        self.conv2d = nn.Conv2d(N, N*C, kernel_size=1) # cast to num_spkr
        self.transconv = nn.ConvTranspose1d(N, 1, L, L//2, bias = False) # second overlap and add

    def forward(self, mixture_w, est_mask):
        """
        Args:
            mixture_w: [M, N, T1]
            est_mask: [M, N, K, S]
        Returns:
            est_source: [M, C, T]
        """
        M, N, K, S = est_mask.shape
        M, N, T1 = mixture_w.shape
        assert self.K == K, "dimension mismatch"
        # [M, C*N, K, S]
        est_mask = self.prelu(est_mask)
        est_mask = self.conv2d(est_mask)
        # [M, C, N, K, S]
        est_mask = est_mask.view(M, self.C, N, K, S)
        # [M, C, N, S, K]
        est_mask = est_mask.transpose(3, 4)
        # [M, C, N, T1]
        est_mask = overlap_and_add(est_mask, self. P)
        est_mask = est_mask[..., :T1]
        est_mask = est_mask.sigmoid()
        # [M, C, N, T1]
        source_w = torch.unsqueeze(mixture_w, 1) * est_mask  
        # [M*C, N, T1]
        source_w = source_w.view(M*self.C, N, T1)
        # [M*C, T]
        est_source = self.transconv(source_w).squeeze(1)
        # [M, C, T]
        est_source = est_source.view(M, self.C, -1)
        return est_source


class MulCat_Block(nn.Module):
    '''
       Implementation of the intra-RNN and the inter-RNN
       input:
            in_channels: The number of expected features in the input x
            N: The number of features in the hidden state h
            rnn_type: RNN, LSTM, GRU
            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
            dropout: If non-zero, introduces a Dropout layer on the outputs 
                     of each LSTM layer except the last layer, 
                     with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: False
    '''

    def __init__(self, N, H, C, dropout=0, bidirectional=True):
        super().__init__()
        # RNN model
        self.intra_rnn1 = torch.nn.LSTM(
            N, H, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.intra_rnn2 = torch.nn.LSTM(
            N, H, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.inter_rnn1 = torch.nn.LSTM(
            N, H, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.inter_rnn2 = torch.nn.LSTM(
            N, H, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        # Norm
        self.intra_norm = torch.nn.LayerNorm(N)
        self.inter_norm = torch.nn.LayerNorm(N)
        # Linear
        self.intra_linear = nn.Linear(
            H*2 + N if bidirectional else H + N, N)
        self.inter_linear = nn.Linear(
            H*2 + N if bidirectional else H + N, N)
        

    def forward(self, x): ## no activation between LSTM blocks?
        '''
           x: [B, N, K, S]
           out: [Spks, B, N, K, S]
        '''
        x_skip = x
        B, N, K, S = x.shape
        # intra RNN
        # [BS, K, N]
        x = x.permute(0, 3, 2, 1).contiguous().view(B*S, K, N)
        # [BS, K, H + N]
        self.intra_rnn1.flatten_parameters()
        x1, _ = self.intra_rnn1(x)
        self.intra_rnn2.flatten_parameters()
        x2, _ = self.intra_rnn2(x)
        x = torch.cat((x1*x2, x), dim = 2)
        # [BS, K, N]
        x = self.intra_linear(x.contiguous().view(B*S*K, -1)).view(B*S, K, -1)
        # [B, S, K, N]
        x = x.view(B, S, K, N)
        x = self.intra_norm(x)
        # [B, N, K, S]
        x = x.permute(0, 3, 2, 1).contiguous()
        x = x + x_skip
        x_skip = x

        # inter RNN
        # [BK, S, N]
        x = x.permute(0, 2, 3, 1).contiguous().view(B*K, S, N)
        # [BK, S, H + N]
        self.inter_rnn1.flatten_parameters()
        x1, _ = self.inter_rnn1(x)
        self.inter_rnn2.flatten_parameters()
        x2, _ = self.inter_rnn2(x)
        x = torch.cat((x1*x2, x), dim = 2)
        # [BK, S, N]
        x = self.inter_linear(x.contiguous().view(B*S*K, -1)).view(B*K, S, -1)
        # [B, K, S, N]
        x = x.view(B, K, S, N)
        x = self.inter_norm(x)
        # [B, N, K, S]
        x = x.permute(0, 3, 1, 2).contiguous()
        out = x + x_skip

        return out



if __name__ == "__main__":
    torch.manual_seed(123)
    M = 2
    T = 40001
    mixture = torch.rand(M, T)
    print('mixture size', mixture.size())

    # N = 64 
    # L = 16 
    # K = 100
    # P = 50 
    # H = 128 
    # B = 6
    # C = 2

    # # test Encoder
    # encoder = Encoder(L, N)
    # mixture_w = encoder(mixture)
    # print('mixture encoding size', mixture_w.size())

    # separator_input = pad_segment(mixture_w, K, P)
    # print('separator_input size', separator_input.size())


    # # test TemporalConvNet
    # separator = MulCat_Block(N, H, C)
    # est_mask = separator(separator_input)
    # print('est_mask size', est_mask.size())

    # # test Decoder
    # decoder = Decoder(N, L, K, P, C)
    # est_source = decoder(mixture_w, est_mask)
    # print('est_source size', est_source.size())

    # test Conv-TasNet
    model = MulCatModel().cuda()
    output = model(mixture.cuda())
    print('model output size', [est_source.size() for est_source in output])

