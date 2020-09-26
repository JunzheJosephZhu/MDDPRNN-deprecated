import os
import torch
import sys
root = "/ws/ifp-10_3/hasegawa/junzhez2/Baseline_Model"
sys.path.append(root)
from src.model_multidecoder import Dual_RNN_model
import importlib
from configs.config6 import *
from configs.config4 import bottleneck as bottle_old
from configs.config4 import hidden as hidden_old
fill_zeros = True
small_pth_path = 'pretrained/raymond_pretrained.pth'
big_pth_path = 'pretrained/raymond_pretrained_expanded.pth'

small_pth_path = os.path.join(root, small_pth_path)
big_pth_path = os.path.join(root, big_pth_path)
small_pth = torch.load(small_pth_path)
model = Dual_RNN_model(enc, bottleneck, hidden, kernel_size=kernel_size, rnn_type=rnn_type, norm=norm, dropout=dropout, bidirectional=True, num_layers=num_layers, K=K, num_spks=num_spks, multiloss=multiloss, mulcat=(mul, cat))
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

old_dict = small_pth['state_dict']
new_dict = model.state_dict()
for ((name1, param1), (name2, param2)) in zip(old_dict.items(), new_dict.items()):
    assert name1 == name2
    # print(name1, param1.size(), param2.size())
    if fill_zeros:
        param2.fill_(0.0000000)
    if param1.shape != param2.shape:
        if name1 in ["decoder.conv2d.weight", "decoder.conv2d.bias"]: # reshaped into 20 groups
            shape1, shape2 = param1.size(), param2.size()
            param1 = param1.view(20, -1, *shape1[1:])
            param2 = param2.view(20, -1, *shape2[1:])
            if len(shape1) > 1:
                param2[:, :param1.size(1), :param1.size(2)] = param1
            else:
                param2[:, :param1.size(1)] = param1
            param2 = param2.view(shape2)
            assert param2.size() == new_dict[name2].size()
            new_dict[name2] = param2
        elif name1 in ["decoder.end_conv1x1.weight", "decoder.output.0.weight", "decoder.output.0.bias", # reshaped into 4 groups
                        "decoder.output_gate.0.weight", "decoder.output_gate.0.bias", "decoder.decoder.weight"] \
                    or ("rnn" in name1 and "ih_l0" in name1) or ("rnn" in name1 and "hh_l0" in name1):
            # rnn gate weights need to be reshaped into (4, ...) because its weight matrices are packed together
            shape1, shape2 = param1.size(), param2.size()
            param1 = param1.view(4, -1, *shape1[1:])
            param2 = param2.view(4, -1, *shape2[1:])
            if len(shape1) > 1:
                param2[:, :param1.size(1), :param1.size(2)] = param1
            else:
                param2[:, :param1.size(1)] = param1
            param2 = param2.view(shape2)
            assert param2.size() == new_dict[name2].size()
            new_dict[name2] = param2
        elif "linear" in name1: # projection layers
            shape1, shape2 = param1.size(), param2.size()
            if len(shape1) > 1:
                param2[:param1.size(0), :bottle_old] = param1[:param1.size(0), :bottle_old]
                param2[:param1.size(0), bottleneck:bottleneck + hidden_old] = param1[:param1.size(0), bottle_old:bottle_old + hidden_old]
                param2[:param1.size(0), bottleneck + hidden:bottleneck + hidden + hidden_old] = param1[:param1.size(0), bottle_old + hidden_old:bottle_old + hidden_old * 2]
            else:
                param2[:param1.size(0)] = param1
            new_dict[name2] = param2
        else:
            assert name1.startswith('separation') or name1.startswith('decoder.vad')
            shape1, shape2 = param1.size(), param2.size()
            assert shape1[2:] == shape2[2:]
            if len(shape1) > 1:
                param2[:param1.size(0), :param1.size(1)] = param1
            else:
                param2[:param1.size(0)] = param1
            new_dict[name2] = param2
    else:
        new_dict[name2] = param1

model.load_state_dict(new_dict)
package = model.serialize(model,
                            optimizer, small_pth["epoch"],
                            tr_loss=small_pth["tr_loss"],
                            cv_loss=small_pth["cv_loss"],
                            val_no_impv=small_pth["val_no_impv"],
                            random_state=small_pth["random_state"])
torch.save(package, big_pth_path)
# decoder.conv2d.weight torch.Size([1280, 64, 1, 1]) Done
# decoder.conv2d.bias torch.Size([1280]) Done
# decoder.end_conv1x1.weight torch.Size([1024, 64, 1]) Done
# decoder.prelu.weight torch.Size([1]) Dontneedtodo
# decoder.output.0.weight torch.Size([256, 64, 1]) Done
# decoder.output.0.bias torch.Size([256]) Done
# decoder.output_gate.0.weight torch.Size([256, 64, 1]) Done
# decoder.output_gate.0.bias torch.Size([256]) Done
# decoder.decoder.weight torch.Size([1024, 1, 8]) Done
# decoder.vad.0.weight torch.Size([256, 64, 1, 1])
# decoder.vad.0.bias torch.Size([256])
# decoder.vad.3.weight torch.Size([4, 256, 1, 1])
# decoder.vad.3.bias torch.Size([4])