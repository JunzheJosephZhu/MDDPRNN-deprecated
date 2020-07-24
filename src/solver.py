# Created on 2018/12
# Author: Kaituo XU

import os
import time

import torch

from pit_criterion import cal_loss

from torch.utils.tensorboard import SummaryWriter

class Solver(object):
    def __init__(self, data, model, optimizer, epochs, save_folder, checkpoint, continue_from, model_path, print_freq=10, half_lr=True,
                early_stop=True, max_norm=5, lr=1e-3, momentum=0.0, l2=0.0, log_dir=None, comment=''):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.model = model
        self.optimizer = optimizer

        # Training config
        self.epochs = epochs
        self.half_lr = half_lr
        self.early_stop = early_stop
        self.max_norm = max_norm
        # save and load model
        self.save_folder = save_folder
        self.checkpoint = checkpoint
        self.continue_from = continue_from
        self.model_path = model_path
        # logging
        self.print_freq = print_freq
        # visualizing loss using visdom
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)

        self._reset()

        self.writer = SummaryWriter(log_dir, comment=comment)
    def _reset(self):
        # Reset
        load = self.continue_from and os.path.exists(self.continue_from)
        self.start_epoch = 0
        self.val_no_impv = 0
        self.prev_val_loss = float("inf")
        self.best_val_loss = float("inf")
        if load: # if the checkpoint model exists
            print('Loading checkpoint model %s' % self.continue_from)
            package = torch.load(self.continue_from)
            self.model.module.load_state_dict(package['state_dict'])
            self.optimizer.load_state_dict(package['optim_dict'])
            self.start_epoch = int(package.get('epoch', 1))
            self.tr_loss[:self.start_epoch] = package['tr_loss'][:self.start_epoch]
            self.cv_loss[:self.start_epoch] = package['cv_loss'][:self.start_epoch]
            self.val_no_impv = package.get('val_no_impv', 0)
            if 'random_state' in package:
                torch.set_rng_state(package['random_state'])
            
            self.prev_val_loss = self.cv_loss[self.start_epoch-1]
            self.best_val_loss = min(self.cv_loss[:self.start_epoch])

        # Create save folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.halving = False

    def train(self):
        # Train model multi-epoches
        for epoch in range(self.start_epoch, self.epochs):
            # Train one epoch
            print("Training...")
            self.model.train()  # Turn on BatchNorm & Dropout
            start = time.time()
            tr_avg_loss = self._run_one_epoch(epoch)
            print('-' * 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Train Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, tr_avg_loss))
            print('-' * 85)


            # Cross validation
            print('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            val_loss = self._run_one_epoch(epoch, cross_valid=True)
            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Valid Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, val_loss))
            print('-' * 85)
            self.writer.add_scalar('Loss/per epoch cv', val_loss, epoch)

            # Adjust learning rate (halving)
            if self.half_lr:
                if val_loss >= self.prev_val_loss:
                    self.val_no_impv += 1
                    if self.val_no_impv >= 3:
                        self.halving = True
                    if self.val_no_impv >= 10 and self.early_stop:
                        print("No improvement for 10 epochs, early stopping.")
                        break
                else:
                    self.val_no_impv = 0
            if self.halving:
                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = \
                    optim_state['param_groups'][0]['lr'] / 2.0
                self.optimizer.load_state_dict(optim_state)
                print('Learning rate adjusted to: {lr:.6f}'.format(
                    lr=optim_state['param_groups'][0]['lr']))
                self.halving = False
            self.prev_val_loss = val_loss

            # Save the best model
            self.tr_loss[epoch] = tr_avg_loss
            self.cv_loss[epoch] = val_loss
            package = self.model.module.serialize(self.model.module,
                                                       self.optimizer, epoch + 1,
                                                       tr_loss=self.tr_loss,
                                                       cv_loss=self.cv_loss,
                                                       val_no_impv = self.val_no_impv,
                                                       random_state=torch.get_rng_state())
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                file_path = os.path.join(self.save_folder, self.model_path)
                torch.save(package, file_path)
                print("Find better validated model, saving to %s" % file_path)

            # Save model each epoch, nd make a copy at last.pth
            if self.checkpoint:
                file_path = os.path.join(
                    self.save_folder, 'epoch%d.pth.tar' % (epoch + 1))
                torch.save(package, file_path)
                torch.save(package, self.continue_from)
                print('Saving checkpoint model to %s' % file_path)



    def _run_one_epoch(self, epoch, cross_valid=False):
        start = time.time()
        total_loss = 0

        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        for i, (data) in enumerate(data_loader):
            padded_mixture, mixture_lengths, padded_source = data
            padded_mixture = padded_mixture.cuda()
            mixture_lengths = mixture_lengths.cuda()
            padded_source = padded_source.cuda()
            try:
                estimate_source_list = self.model(padded_mixture)
            except:
                print(padded_mixture.shape)
                continue
            loss = []
            for estimate_source in estimate_source_list:
                step_loss, max_snr, estimate_source, reorder_estimate_source = \
                    cal_loss(padded_source, estimate_source, mixture_lengths)
                loss.append(step_loss)
            if not cross_valid: # training
                loss = torch.stack(loss).mean()
            else:
                loss = loss[-1]
            try:
                if not cross_valid:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                self.max_norm)
                    self.optimizer.step()
            except:
                print(padded_mixture.shape)
                continue
            total_loss += loss.item()

            if i % self.print_freq == 0:
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                      'Current Loss {3:.6f} | {4:.1f} ms/batch'.format(
                          epoch + 1, i + 1, total_loss / (i + 1),
                          loss.item(), 1000 * (time.time() - start) / (i + 1)),
                      flush=True)
            
            if not cross_valid:
                self.writer.add_scalar('Loss/train', loss.item(), epoch*len(data_loader)+i)
            else:
                self.writer.add_scalar('Loss/cv', loss.item(), epoch*len(data_loader)+i)


        return total_loss / (i + 1)
