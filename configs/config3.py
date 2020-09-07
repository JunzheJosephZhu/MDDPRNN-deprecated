import time
import sys
import os
config = os.path.basename(__file__).split('.')[0]
root =  "/ws/ifp-10_3/hasegawa/junzhez2/Baseline_Model"
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
batch_size = 12
norm = 'ln'
rnn_type = 'LSTM'
dropout = 0.0
lr = 1e-3
lr_override = False
momentum = 0.0
l2 = 0.0 # weight decya
save_folder = os.path.join(root, 'models')
checkpoint = 1
continue_from = os.path.join(save_folder, "config3.pth")
model_path = config + "_best.pth"
print_freq = 10
comment = config + 'start from randomly initialized model'
log_dir = os.path.join(root, 'runs', time.strftime("%Y%m%d-%H%M%S")+comment)
use_onoff = True # use on/off head or not; if off, use DPRNN
multiloss = True # useless if use_onoff=False
mul = False # useless if use_onoff=False
cat = False # useless if use_onoff=False
decay_period = 2
multidecoder = False
device_ids = [0, 1, 2, 3]