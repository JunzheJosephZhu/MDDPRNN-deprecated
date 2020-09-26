import time
import sys
import os
config = os.path.basename(__file__).split('.')[0]
root = "/ws/ifp-10_3/hasegawa/junzhez2/Baseline_Model"
lamb = 0.5
maxlen = 4
minlen = 2
kernel_size = 8
enc = 256
bottleneck = 128
hidden = 256
num_layers = 6
K = 125
num_spks = 5
epochs = 128
half_lr = True # feature not enabled
early_stop = True
max_norm = 5
shuffle = False
batch_size = 3
norm = 'ln'
rnn_type = 'LSTM'
dropout = 0.0
lr = 5e-4
lr_override = False
momentum = 0.0
l2 = 0.0 # weight decay
save_folder = os.path.join(root, 'models')
checkpoint = 1
continue_from = os.path.join(root, "pretrained/raymond_pretrained_expanded.pth") # if not exist, randomly initialize
model_path = config + "_best.pth" # best model save path
print_freq = 10
comment = ' weighted chunk resampling'
log_dir = os.path.join(root, 'runs', time.strftime("%Y%m%d-%H%M%S") + config + comment)
use_onoff = True # hungarian model if on, DPRNN if off. useless if multidecoder=True
multiloss = False
mul = True
cat = True
decay_period = 1
multidecoder = True
device_ids = [0, 1, 2]