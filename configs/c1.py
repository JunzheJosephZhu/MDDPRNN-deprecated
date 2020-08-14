import time
import sys
import os
comment = os.path.basename(sys.argv[0])
maxlen = 4
minlen = 2
enc = 256 
bottleneck = 64 
hidden = 128
num_layers = 6
num_spks = 2
epochs = 128
half_lr = True
early_stop = True
max_norm = 5
shuffle = False
batch_size = 4
lr = 1e-3
momentum = 0.0
l2 = 0.0 # weight decya
save_folder = "/ws/ifp-10_3/hasegawa/junzhez2/Baseline_Model/models"
checkpoint = 1
continue_from = save_folder+'dummy'#+"/last.pth"
model_path = "best.pth"
print_freq = 10
comment += ''
log_dir = "/ws/ifp-10_3/hasegawa/junzhez2/Baseline_Model/runs/"+time.strftime("%Y%m%d-%H%M%S")+comment
use_mulcat = True
multiloss = False