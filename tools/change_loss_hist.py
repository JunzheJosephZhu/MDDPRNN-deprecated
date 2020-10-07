root = "/ws/ifp-10_3/hasegawa/junzhez2/Baseline_Model"
model = 'models/config6.pth'
import torch
import os
import numpy as np
pkg = torch.load(os.path.join(root, model))
pkg['cv_loss'][:pkg['epoch']].fill_(np.inf)
print(pkg['cv_loss'])
torch.save(pkg, os.path.join(root, model))