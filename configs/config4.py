import time
import sys
import os
config = os.path.basename(__file__).split('.')[0]
lamb = 0.5
maxlen = 4
minlen = 2
kernel_size = 8
enc = 256
bottleneck = 64 
hidden = 128
num_layers = 6
K = 125
num_spks = 5
epochs = 128
half_lr = True # feature not enabled
early_stop = True
max_norm = 5
shuffle = False
batch_size = 4
norm = 'ln'
rnn_type = 'LSTM'
dropout = 0.0
lr = 5e-4
lr_override = False
momentum = 0.0
l2 = 0.0 # weight decya
save_folder = "/ws/ifp-10_3/hasegawa/junzhez2/MultiDecoder/models"
checkpoint = 1
continue_from = os.path.join(save_folder, "config4.pth")
model_path = "best.pth"
print_freq = 10
comment = config + ' multiple decoder with the same TCN. Random Initialization'
log_dir = "/ws/ifp-10_3/hasegawa/junzhez2/MultiDecoder/runs/"+time.strftime("%Y%m%d-%H%M%S")+comment
use_onoff = True # use on/off head or not; if off, use DPRNN
multiloss = True # useless if use_onoff=False
mul = False # useless if use_onoff=False
cat = False # useless if use_onoff=False
decay_period = 2
multidecoder = False