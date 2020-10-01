# Created on 2018/12
# Author: Kaituo XU

import os
import time
import numpy as np
import torch


from torch.utils.tensorboard import SummaryWriter

class Solver(object):
    def __init__(self, data, model, optimizer, epochs, save_folder, checkpoint, continue_from, model_path, print_freq, half_lr,
                early_stop, max_norm, lr, lr_override, momentum, l2, log_dir, lamb, decay_period, config, multidecoder):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.model = model
        self.optimizer = optimizer

        # Training config
        self.epochs = epochs
        self.half_lr = half_lr
        self.early_stop = early_stop
        self.max_norm = max_norm
        self.lamb = lamb
        self.decay_period = decay_period
        self.multidecoder = multidecoder
        if multidecoder:
            from loss_multidecoder import cal_loss
        else:
            from loss_hungarian import cal_loss
        self.loss_func = cal_loss
        # save and load model
        self.save_folder = save_folder
        self.checkpoint = checkpoint
        self.continue_from = continue_from
        self.model_path = model_path
        self.config = config
        # logging
        self.print_freq = print_freq
        # visualizing loss using visdom
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)

        self._reset()

        self.writer = SummaryWriter(log_dir)
        # learning rate override
        if lr_override:
            optim_state = self.optimizer.state_dict()
            optim_state['param_groups'][0]['lr'] = lr
            self.optimizer.load_state_dict(optim_state)
            print('Learning rate adjusted to: {lr:.6f}'.format(
                lr=optim_state['param_groups'][0]['lr']))

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
            tr_avg_loss, tr_avg_snr, tr_avg_acc = self._run_one_epoch(epoch)
            print('-' * 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Train Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, tr_avg_loss))
            print('-' * 85)


            # Cross validation
            print('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            val_loss, val_snr, val_acc = self._run_one_epoch(epoch, cross_valid=True)
            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Valid Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, val_loss))
            print('-' * 85)
            self.writer.add_scalar('Loss/per_epoch_cv', val_loss, epoch)
            self.writer.add_scalar('SNR/per_epoch_cv', val_snr, epoch)
            self.writer.add_scalar('Accuracy/per_epoch_cv', val_acc, epoch)

            # Adjust learning rate (halving)
            if self.half_lr:
                if val_loss >= self.prev_val_loss:
                    self.val_no_impv += 1
                    if self.val_no_impv >= 10 and self.early_stop:
                        print("No improvement for 10 epochs, early stopping.")
                        break
                else:
                    self.val_no_impv = 0

            if epoch % self.decay_period == (self.decay_period-1):
                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = \
                    optim_state['param_groups'][0]['lr'] * 0.98
                self.optimizer.load_state_dict(optim_state)
                print('Learning rate adjusted to: {lr:.6f}'.format(
                    lr=optim_state['param_groups'][0]['lr']))
                    
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
                print('Saving checkpoint model to %s' % file_path)

            # update config#.pth
            torch.save(package, os.path.join(self.save_folder, self.config + '.pth'))



    def _run_one_epoch(self, epoch, cross_valid=False):
        start = time.time()
        total_loss = 0
        total_snr = 0
        total_accuracy = 0
        data_loader = self.tr_loader if not cross_valid else self.cv_loader
        current_device = next(self.model.module.parameters()).device

        for i, (padded_mixture, mixture_lengths, padded_source) in enumerate(data_loader):
            B = len(padded_source)
            padded_mixture = padded_mixture.cuda(current_device)
            padded_source = [tmp_ps.cuda(current_device) for tmp_ps in padded_source]
            num_sources = torch.Tensor([tmps_ps.size(0) for tmps_ps in padded_source]).long()
            try:
                if not cross_valid:
                    estimate_source_list, vad_list = self.model(padded_mixture, num_sources, True)
                else:
                    with torch.no_grad():
                        estimate_source_list, vad_list = self.model(padded_mixture, num_sources, True)
            except Exception as e:
                print('forward prop failed', padded_mixture.shape, e)
                continue
            if not self.multidecoder:
                # [#stages, B, ...]
                estimate_source_list = estimate_source_list.transpose(0, 1)
                vad_list = vad_list.transpose(0, 1)
                loss = []
                snr = []
                accuracy = []
                for (estimate_source, vad) in zip(estimate_source_list, vad_list):
                    step_loss, step_snr, acc = \
                            self.loss_func(padded_source, estimate_source, mixture_lengths, vad, lamb=self.lamb)
                    loss.append(step_loss)
                    snr.append(step_snr)
                    accuracy.append(acc)
                loss = torch.stack(loss)
                snr = torch.stack(snr)
                accuracy = torch.stack(accuracy)
            else: # if using multidecoder
                # list of B, each [num_stages, spks, T]
                estimate_sources = [estimate_source_list[i, :, :num_sources[i], :] for i in range(B)]
                loss = []
                snr = []
                accuracy = []
                for idx in range(B):
                    # list of [num_stages, spks, T]
                    # [num_stages, num_decoders]
                    vad = vad_list[idx]
                    step_loss, step_snr, acc = \
                            self.loss_func(padded_source[idx], estimate_sources[idx], mixture_lengths[idx], vad, self.lamb)
                    # [num_stages]
                    loss.append(step_loss)
                    snr.append(step_snr)
                    accuracy.append(acc)
                loss = torch.stack(loss, dim=0).mean(dim=0)
                snr = torch.stack(snr, dim=0).mean(dim=0)
                accuracy = torch.stack(accuracy, dim=0).mean(dim=0)
            if not cross_valid: # training
                loss = loss.mean()
                snr = snr.mean()
                accuracy = accuracy.mean()
            else:
                loss = loss[-1] 
                snr = snr[-1]
                accuracy = accuracy[-1]
            try:
                if not cross_valid:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                self.max_norm)
                    self.optimizer.step()
            except Exception as e:
                print('backprop failed', padded_mixture.shape, e)
                continue
            total_loss += loss.item()
            total_snr += snr.item()
            total_accuracy += accuracy.item()
            if i % self.print_freq == 0:
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                      'Current Loss {3:.6f} | Average SNR {4: .3f} | Average accuracy {5:.3f} | {6:.1f} ms/batch'.format(
                          epoch + 1, i + 1, total_loss / (i + 1),
                          loss.item(), total_snr / (i + 1), total_accuracy / (i + 1), 
                          1000 * (time.time() - start) / (i + 1)),
                      flush=True)
            
            if not cross_valid:
                self.writer.add_scalar('Loss/train', loss.item(), epoch*len(data_loader)+i)
                self.writer.add_scalar('SNR/train', snr.item(), epoch*len(data_loader)+i)
                self.writer.add_scalar('Accuracy/train', accuracy.item(), epoch*len(data_loader)+i)
            else:
                self.writer.add_scalar('Loss/cv', loss.item(), epoch*len(data_loader)+i)
                self.writer.add_scalar('SNR/cv', snr.item(), epoch*len(data_loader)+i)
                self.writer.add_scalar('Accuracy/cv', accuracy.item(), epoch*len(data_loader)+i)


        return total_loss / (i + 1), total_snr / (i + 1), total_accuracy / (i + 1)
