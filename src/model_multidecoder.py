import sys
sys.path.append('../')

import torch.nn.functional as F
from torch import nn
import torch
import warnings
import time

warnings.filterwarnings('ignore')

class GlobalLayerNorm(nn.Module):
    '''
       Calculate Global Layer Normalization
       dim: (int or list or torch.Size) –
          input shape from an expected input of size
       eps: a value added to the denominator for numerical stability.
       elementwise_affine: a boolean value that when set to True, 
          this module has learnable per-element affine parameters 
          initialized to ones (for weights) and zeros (for biases).
    '''

    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if shape == 3:
                self.weight = nn.Parameter(torch.ones(self.dim, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1))
            if shape == 4:
                self.weight = nn.Parameter(torch.ones(self.dim, 1, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x = N x C x K x S or N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x K x S
        # gln: mean,var N x 1 x 1
        if x.dim() == 4:
            mean = torch.mean(x, (1, 2, 3), keepdim=True)
            var = torch.mean((x-mean)**2, (1, 2, 3), keepdim=True)
            if self.elementwise_affine:
                x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
            else:
                x = (x-mean)/torch.sqrt(var+self.eps)
        if x.dim() == 3:
            mean = torch.mean(x, (1, 2), keepdim=True)
            var = torch.mean((x-mean)**2, (1, 2), keepdim=True)
            if self.elementwise_affine:
                x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
            else:
                x = (x-mean)/torch.sqrt(var+self.eps)
        return x

class CumulativeLayerNorm(nn.LayerNorm):
    '''
       Calculate Cumulative Layer Normalization
       dim: you want to norm dim
       elementwise_affine: learnable per-element affine parameters 
    '''

    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine, eps=1e-8)

    def forward(self, x):
        # x: N x C x K x S or N x C x L
        # N x K x S x C
        if x.dim() == 4:
           x = x.permute(0, 2, 3, 1).contiguous()
           # N x K x S x C == only channel norm
           x = super().forward(x)
           # N x C x K x S
           x = x.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 3:
            x = torch.transpose(x, 1, 2)
            # N x L x C == only channel norm
            x = super().forward(x)
            # N x C x L
            x = torch.transpose(x, 1, 2)
        return x

def select_norm(norm, dim, shape):
    if norm == 'gln':
        return GlobalLayerNorm(dim, shape, elementwise_affine=True)
    if norm == 'cln':
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == 'ln':
        return nn.GroupNorm(1, dim, eps=1e-8)
    else:
        return nn.BatchNorm1d(dim)

class Encoder(nn.Module):
    '''
       Conv-Tasnet Encoder part
       kernel_size: the length of filters
       out_channels: the number of filters
    '''

    def __init__(self, kernel_size, out_channels):
        super(Encoder, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=out_channels,
                                kernel_size=kernel_size, stride=kernel_size//2, groups=1, bias=False)

    def forward(self, x):
        """
          Input:
              x: [B, T], B is batch size, T is times
          Returns:
              x: [B, C, T_out]
              T_out is the number of time steps
        """
        # B x T -> B x 1 x T
        x = torch.unsqueeze(x, dim=1)
        # B x 1 x T -> B x C x T_out
        x = self.conv1d(x)
        x = F.relu(x)
        return x

class Dual_RNN_Block(nn.Module):
    '''
       Implementation of the intra-RNN and the inter-RNN
       input:
            in_channels: The number of expected features in the input x
            out_channels: The number of features in the hidden state h
            rnn_type: RNN, LSTM, GRU
            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
            dropout: If non-zero, introduces a Dropout layer on the outputs 
                     of each LSTM layer except the last layer, 
                     with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: False
    '''

    def __init__(self, out_channels,
                 hidden_channels, rnn_type, norm,
                 dropout, bidirectional, mulcat):
        super(Dual_RNN_Block, self).__init__()
        # RNN model
        self.mul, self.cat = mulcat
        self.intra_rnn1 = getattr(nn, rnn_type)(
            out_channels, hidden_channels, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.inter_rnn1 = getattr(nn, rnn_type)(
            out_channels, hidden_channels, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        if self.mul:
            self.intra_rnn2 = getattr(nn, rnn_type)(
                out_channels, hidden_channels, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
            self.inter_rnn2 = getattr(nn, rnn_type)(
                out_channels, hidden_channels, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        # Norm
        self.intra_norm = select_norm(norm, out_channels, 4)
        self.inter_norm = select_norm(norm, out_channels, 4)
        # Linear
        if self.cat:
            self.intra_linear = nn.Linear(
                hidden_channels*2 + out_channels if bidirectional else hidden_channels + out_channels, out_channels)
            self.inter_linear = nn.Linear(
                hidden_channels*2 + out_channels if bidirectional else hidden_channels + out_channels, out_channels)
        else:
            self.intra_linear = nn.Linear(
                hidden_channels*2 if bidirectional else hidden_channels, out_channels)
            self.inter_linear = nn.Linear(
                hidden_channels*2 if bidirectional else hidden_channels, out_channels)
        

    def forward(self, x):
        '''
           x: [B, N, K, S]
           out: [Spks, B, N, K, S]
        '''
        B, N, K, S = x.shape
        # intra RNN
        # [BS, K, N]
        intra_rnn = x.permute(0, 3, 2, 1).contiguous().view(B*S, K, N)
        # [BS, K, H + N]
        self.intra_rnn1.flatten_parameters()
        intra_mul = self.intra_rnn1(intra_rnn)[0]
        if self.mul:
            self.intra_rnn2.flatten_parameters()
            intra_mul = intra_mul * self.intra_rnn2(intra_rnn)[0]
        if self.cat:
            intra_rnn = torch.cat([intra_rnn, intra_mul], dim=-1)
        else:
            intra_rnn = intra_mul
        # [BS, K, N]
        intra_rnn = self.intra_linear(intra_rnn.contiguous().view(B*S*K, -1)).view(B*S, K, -1)
        # [B, S, K, N]
        intra_rnn = intra_rnn.view(B, S, K, N)
        # [B, N, K, S]
        intra_rnn = intra_rnn.permute(0, 3, 2, 1).contiguous()
        intra_rnn = self.intra_norm(intra_rnn)
        
        # [B, N, K, S]
        intra_rnn = intra_rnn + x

        # inter RNN
        # [BK, S, N]
        inter_rnn = intra_rnn.permute(0, 2, 3, 1).contiguous().view(B*K, S, N)
        # [BK, S, H + N]
        self.inter_rnn1.flatten_parameters()
        inter_mul = self.inter_rnn1(inter_rnn)[0]
        if self.mul:
            self.inter_rnn2.flatten_parameters()
            inter_mul = inter_mul * self.inter_rnn2(inter_rnn)[0]
        if self.cat:
            inter_rnn = torch.cat([inter_rnn, inter_mul], dim=-1)
        else:
            inter_rnn = inter_mul

        # [BK, S, N]
        inter_rnn = self.inter_linear(inter_rnn.contiguous().view(B*S*K, -1)).view(B*K, S, -1)
        # [B, K, S, N]
        inter_rnn = inter_rnn.view(B, K, S, N)
        # [B, N, K, S]
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous()
        inter_rnn = self.inter_norm(inter_rnn)
        # [B, N, K, S]
        out = inter_rnn + intra_rnn

        return out

class Dual_Path_RNN(nn.Module):
    '''
       Implementation of the Dual-Path-RNN model
       input:
            in_channels: The number of expected features in the input x
            out_channels: The number of features in the hidden state h
            rnn_type: RNN, LSTM, GRU
            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
            dropout: If non-zero, introduces a Dropout layer on the outputs 
                     of each LSTM layer except the last layer, 
                     with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: False
            num_layers: number of Dual-Path-Block
            K: the length of chunk
            num_spks: the number of speakers
    '''

    def __init__(self, in_channels, out_channels, hidden_channels,
                 rnn_type, norm, dropout,
                 bidirectional, num_layers, K, mulcat):
        super(Dual_Path_RNN, self).__init__()
        self.K = K
        self.num_layers = num_layers
        self.norm = select_norm(norm, in_channels, 3)
        self.conv1d = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        self.dual_rnn = nn.ModuleList([])
        for i in range(num_layers):
            self.dual_rnn.append(Dual_RNN_Block(out_channels, hidden_channels,
                                     rnn_type=rnn_type, norm=norm, dropout=dropout,
                                     bidirectional=bidirectional, mulcat=mulcat))

    def forward(self, x):
        '''
           x: [B, N, L]
        '''
        # [B, N, L]
        x = self.norm(x)
        # [B, N, L]
        x = self.conv1d(x)
        # [B, N, K, S]
        x, gap = self._Segmentation(x, self.K)

        xs = []
        for i in range(self.num_layers):
            x = self.dual_rnn[i](x)
            xs.append(x)
        return xs, gap

    def _Segmentation(self, input, K):
        '''
           the segmentation stage splits
           K: chunks of length
           P: hop size
           input: [B, N, L]
           output: [B, N, K, S]
        '''
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = torch.cat([input1, input2], dim=3).view(
            B, N, -1, K).transpose(2, 3)

        return input.contiguous(), gap

    def _padding(self, input, K):
        '''
           padding the audio times
           K: chunks of length
           P: hop size
           input: [B, N, L]
        '''
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.type())
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap

class Decoder(nn.ConvTranspose1d):
    '''
        Decoder of the TasNet
        This module can be seen as the gradient of Conv1d with respect to its input. 
        It is also known as a fractionally-strided convolution 
        or a deconvolution (although it is not an actual deconvolution operation).
    '''

    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: [B, N, L]
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 3/4D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))

        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)
        return x

class SingleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, num_layers, num_spks):
        super(SingleDecoder, self).__init__()
        self.num_layers = num_layers
        self.num_spks = num_spks
        self.conv2d = nn.Conv2d(
            out_channels, out_channels*num_spks, kernel_size=1)
        self.end_conv1x1 = nn.Conv1d(out_channels, in_channels, 1, bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
         # gated output layer
        self.output = nn.Sequential(nn.Conv1d(out_channels, out_channels, 1),
                                    nn.Tanh()
                                    )
        self.output_gate = nn.Sequential(nn.Conv1d(out_channels, out_channels, 1),
                                         nn.Sigmoid()
                                         )
        self.decoder = Decoder(in_channels, out_channels=1, kernel_size=kernel_size, stride=kernel_size//2, bias=False)

    def forward(self, x, e, gap, num_stages):
        '''
            args:
                x: [#stages*B, N, K, S]
                e: [B, N, L]
            outputs:
                x: [#stages*B, spks, T]
        '''
        print('start device %d decoder %d' % (torch.cuda.current_device(), self.num_spks-2))
        x = self.prelu(x)
        x = self.conv2d(x)
        # [#stages*B*spks, N, K, S]
        stagexB, _, K, S = x.shape
        x = x.view(stagexB*self.num_spks,-1, K, S)
        # [#stages*B*spks, N, L]
        x = self._over_add(x, gap)
        x = self.output(x)*self.output_gate(x)
        # [#stages*B*spks, N, L]
        x = self.end_conv1x1(x)
        _, N, L = x.shape
        # [#stages, B, spks, N, L]
        x = x.view(num_stages, e.shape[0], self.num_spks, N, L)
        x = self.activation(x)
        # [1, B, 1, N, L]
        e = e.unsqueeze(0).unsqueeze(2)
        x *= e
        # [#stages*B*spks, N, L]
        x = x.view(stagexB*self.num_spks, N, L)
        # [B, spks, T]
        x = self.decoder(x).view(stagexB, self.num_spks, -1)
        print('ended device %d decoder %d' % (torch.cuda.current_device(), self.num_spks-2))

        return x

    def _over_add(self, input, gap):
        '''
           Merge sequence
           input: [B, N, K, S]
           gap: padding length
           output: [B, N, L]
        '''
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input

class MultiDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, num_layers, max_spks, multiloss):
        super(MultiDecoder, self).__init__()
        self.multiloss = multiloss
        self.num_layers = num_layers
        self.max_spks = max_spks
        self.num_spks = torch.arange(2, max_spks + 1)
        self.num_decoders = len(self.num_spks)
        self.conv2d = nn.Conv2d(out_channels, max_spks*out_channels*self.num_decoders, kernel_size=1)
        self.end_conv1x1 = nn.Conv1d(out_channels*self.num_decoders, in_channels*self.num_decoders, 1, groups=self.num_decoders, bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
         # gated output layer
        self.output = nn.Sequential(nn.Conv1d(out_channels*self.num_decoders, out_channels*self.num_decoders, 1, groups=self.num_decoders),
                                    nn.Tanh()
                                    )
        self.output_gate = nn.Sequential(nn.Conv1d(out_channels*self.num_decoders, out_channels*self.num_decoders, 1, groups=self.num_decoders),
                                         nn.Sigmoid()
                                         )
        self.decoder = Decoder(in_channels*self.num_decoders, out_channels=1*self.num_decoders, kernel_size=kernel_size, stride=kernel_size//2, groups=self.num_decoders, bias=False)        
        self.vad = nn.Sequential(nn.Conv2d(out_channels, in_channels, 1),
                                 nn.AdaptiveAvgPool2d(1),
                                 nn.ReLU(),
                                 nn.Conv2d(in_channels, self.num_decoders, 1)
                                 )

    def forward(self, x, e, gap):
        """
        args:
            x: list of num_layers, each being [B, N, K, S]
            gap: gap in segmentation to be removed
        returns:
            signals: list of num_decoders, each [B, #stages, num_spks[i], T]
            vad: [B, #stages, num_decoders]
        """
        B, N, K, S = x[0].size()
        num_stages = self.num_layers if self.multiloss and self.training else 1
        # [#stages*B, out_channels, K, S]
        if self.multiloss and self.training:
            x = torch.stack(x, dim=0).view(self.num_layers*B, N, K, S)
        else:
            x = x[-1]
        # [#stages*B, num_decoders]
        vad = self.vad(x).squeeze() # only logits
        # [#stages, B, num_decoders]
        vad = vad.view(num_stages, B, -1)
        # startt = time.time()

        ''' old implementation
            # list of num_decoders, each [#stages*B, spks, T]
            x = [decoder(x, e, gap, num_stages) for decoder in self.decoderlist]
            T = x[0].shape[-1]
            # list of num_decoders, each [B, #stages, spks, T]
            x = [signal.view(num_stages, B, -1, T).transpose(0, 1) for signal in x]
        '''
        # [B, #stages, max_spks, num_decoders, T]
        x = self.decode(x, e, gap, num_stages)
        signals = []
        for decoder_id in range(self.num_decoders):
            valid_spks = self.num_spks[decoder_id]
            # [B, #stages, num_spks[i], T]
            decoder_signal = x[:, :, :valid_spks, decoder_id, :]
            signals.append(decoder_signal)

        # print('device %d used %.3f' % (torch.cuda.current_device(), time.time()-startt))

        return signals, vad.transpose(0, 1)

    def decode(self, x, e, gap, num_stages):
        '''
            args:
                x: [#stages*B, N, K, S]
                e: [B, N, L]
            outputs:
                x: [B, #stages, max_spks, num_decoders, T]
        '''
        _, out_channels, K, S = x.size()
        B, in_channels, L = e.size()
        x = self.prelu(x)
        # [#stages*B, max_spks*out_channels*num_decoders, K, S]
        x = self.conv2d(x)
        # [#stages*B*max_spks, N*num_decoders, K, S]
        x = x.view(num_stages*B*self.max_spks, out_channels*self.num_decoders, K, S)
        # [#stages*B*max_spks, N*num_decoders, L]
        x = self._over_add(x, gap)
        x = self.output(x)*self.output_gate(x)
        # [#stages*B*max_spks, N*num_decoders, L]
        x = self.end_conv1x1(x)
        # [#stages, B, max_spks, N, num_decoders, L]
        x = x.view(num_stages, B, self.max_spks, in_channels, self.num_decoders, L)
        x = self.activation(x)
        # [1, B, 1, N, 1, L]
        e = e.unsqueeze(0).unsqueeze(2).unsqueeze(4)
        x *= e
        # [#stages*B*max_spks, N*num_decoders, L]
        x = x.view(num_stages*B*self.max_spks, in_channels*self.num_decoders, L)
        # [#stages*B*max_spks, num_decoders, T]
        x = self.decoder(x)
        # [#stages, B, max_spks, num_decoders, T]
        x = x.view(num_stages, B, self.max_spks, self.num_decoders, -1)

        return x.transpose(0, 1)

    def _over_add(self, input, gap):
        '''
           Merge sequence
           input: [B, N, K, S]
           gap: padding length
           output: [B, N, L]
        '''
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input

class Dual_RNN_model(nn.Module):
    '''
       model of Dual Path RNN
       input:
            in_channels: The number of expected features in the input x
            out_channels: The number of features in the hidden state h
            hidden_channels: The hidden size of RNN
            kernel_size: Encoder and Decoder Kernel size
            rnn_type: RNN, LSTM, GRU
            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
            dropout: If non-zero, introduces a Dropout layer on the outputs 
                     of each LSTM layer except the last layer, 
                     with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: False
            num_layers: number of Dual-Path-Block
            K: the length of chunk
            num_spks: the number of speakers
    '''
    def __init__(self, in_channels, out_channels, hidden_channels,
                 kernel_size, rnn_type, norm, dropout,
                 bidirectional, num_layers, K, num_spks, multiloss, mulcat):
        super(Dual_RNN_model,self).__init__()
        self.encoder = Encoder(kernel_size=kernel_size,out_channels=in_channels)
        self.separation = Dual_Path_RNN(in_channels, out_channels, hidden_channels,
                 rnn_type=rnn_type, norm=norm, dropout=dropout,
                 bidirectional=bidirectional, num_layers=num_layers, K=K, mulcat=mulcat)
        self.decoder = MultiDecoder(in_channels, out_channels, hidden_channels, kernel_size, num_layers, num_spks, multiloss)
    
    def forward(self, x):
        '''
           x: [B, L]
        '''
        # [B, N, L]
        e = self.encoder(x)
        # list of #stages, [B, N, K, S]
        s, gap = self.separation(e)
        # signals: list of num_spks - 1, each [B, #stages, spks, T]
        # vad: [B, #stages, num_spks - 1]
        signals, vad = self.decoder(s, e, gap)
            
        return signals, vad

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



if __name__ == "__main__":
    rnn = torch.nn.DataParallel(Dual_RNN_model(256, 64, 128, kernel_size=8, rnn_type='LSTM', norm='ln', dropout=0.0, bidirectional=True, num_layers=6, K=125, num_spks=5, multiloss=True, mulcat=(True, True))).cuda()
    x = torch.ones(4, 32000).cuda()
    audio, vad = rnn(x)
    print(len(audio), audio[1].shape)
    print(vad.shape)
    def check_parameters(net):
        '''
            Returns module parameters. Mb
        '''
        parameters = sum(param.numel() for param in net.parameters())
        return parameters / 10**6
    print("{:.3f}".format(check_parameters(rnn)*1000000))