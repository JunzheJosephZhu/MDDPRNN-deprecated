import time
import sys
import os
config_name = os.path.basename(__file__).split('.')[0]
root = "/ws/ifp-10_3/hasegawa/junzhez2/Baseline_Model"
lamb = 0.01
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
early_stop = True
max_norm = 5
shuffle = False
batch_size = 4
norm = 'ln'
rnn_type = 'LSTM'
dropout = 0.0
lr = 3e-4
save_folder = os.path.join(root, 'models')
checkpoint = 1
continue_from = os.path.join(save_folder, "config6.pth") # if not exist, randomly initialize
model_path = config_name + "_best.pth" # best model save path
print_freq = 10
comment = ' fixed bug concerning lr schedule'
log_dir = os.path.join(root, 'runs', time.strftime("%Y%m%d-%H%M%Sd") + config_name + comment)
use_onoff = True # hungarian model if on, DPRNN if off. useless if multidecoder=True
multiloss = True
mul = True
cat = True
decay_period = 1
decay = 0.94
multidecoder = True
device_ids = [0, 1, 2, 3]