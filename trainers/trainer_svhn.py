import os
import time
import shutil
import pickle
import json
import numpy as np

import torch
import torch.nn.functional as F
from torchsummary import summary
from flops_profiler.profiler import FlopsProfiler

from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR
import wandb

from modelling.model import DeepRecurrentAttention
from trainers.utils import AverageMeter


class SVHNTrainer:
    """A Deep Recurrent Attention Model trainer.

    All hyperparameters are provided by the user in the
    config file.
    """

    def __init__(self, config, train_val_loader=None, test_loader=None, is_gridsearch=False):
        """
        Construct a new Trainer instance.

        Args:
            config (argparse): 
                Object containing command line arguments.
            train_val_loader (tuple): 
                Tuple with the train and validation dataloaders.
            test_loader (Dataloader): 
                Test dataloader (data iterator).
            is_gridsearch (bool): 
                Wether a gridsearch is being performed. Used mostly
                to control WandB logging.
        """
        self.config = config

        if config.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # glimpse network params
        self.patch_size = config.patch_size
        self.glimpse_scale = config.glimpse_scale
        self.num_patches = config.num_patches
        self.loc_hidden = config.loc_hidden
        self.glimpse_hidden = config.glimpse_hidden

        # core network params
        self.core_type = config.core_type
        self.transformer_model = config.transformer_model
        self.use_encoder = config.use_encoder
        self.num_glimpses = config.num_glimpses
        self.explore_steps = config.explore_steps
        self.hidden_size = config.hidden_size
        self.cell_size = config.cell_size
        self.inner_size = config.inner_size
        self.n_heads = config.n_heads
        
        # context network params
        self.snapshot = config.snapshot
        self.keep_context = config.keep_context

        # reinforce params            
        self.M = config.M
        self.rl_loss_coef = config.rl_loss_coef
        self.std = config.std
        
        # data params
        if train_val_loader is not None:
            self.train_loader = train_val_loader[0]
            self.valid_loader = train_val_loader[1]
            self.num_train = len(self.train_loader.dataset)
            self.num_valid = len(self.valid_loader.dataset)
        if test_loader is not None:
            self.test_loader = test_loader
            self.num_test = len(self.test_loader.dataset)
        self.num_classes = 11
        self.num_channels = 1
        self.end_class = config.end_class
        
        self.image_size = 54 if config.preprocess == "crop" else int(config.preprocess)

        # training params
        self.epochs = config.epochs
        self.start_epoch = 0
        self.momentum = config.momentum
        self.lr = config.init_lr
        self.optimizer = config.optimizer
        self.ignore_index = -100

        # misc params
        self.best = config.best
        self.ckpt_dir = config.ckpt_dir
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
            
        self.pretrained_ckpt = config.pretrained_ckpt
        if self.pretrained_ckpt is not None:
            assert os.path.exists(self.pretrained_ckpt), "Pretrained checkpoint does not exist!"
        
        self.best_valid_acc = 0.0
        self.counter = 0
        self.lr_patience = config.lr_patience
        self.train_patience = config.train_patience
        self.save_results = config.save_results
        self.wandb_name = config.wandb_name
        self.use_wandb = True if (self.wandb_name or is_gridsearch) else False
        self.resume = config.resume
        self.print_freq = config.print_freq
        self.plot_freq = config.plot_freq
        self.model_name = "dram_{}_{}x{}_{}".format(
            config.num_glimpses,
            config.patch_size,
            config.patch_size,
            config.glimpse_scale,
        )

        # configure wandb logging
        if self.use_wandb and not is_gridsearch:
            wandb.init(
                entity=config.wandb_entity,
                project=config.wandb_project, 
                name=self.wandb_name,
                config=config)
            
        # build DRAM model
        self.model = DeepRecurrentAttention(
            self.patch_size,
            self.num_patches,
            self.glimpse_scale,
            self.num_channels,
            self.glimpse_hidden,
            self.loc_hidden,
            self.std,
            self.hidden_size,
            self.cell_size,
            self.inner_size,
            self.n_heads,
            self.num_classes,
            self.core_type,
            self.device,
            self.transformer_model,
            self.use_encoder,
            self.image_size,
            self.snapshot
        )
        self.model.to(self.device)

        # initialize optimizer and scheduler
        if self.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.config.init_lr, momentum=self.config.momentum
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.config.init_lr, weight_decay=self.config.weight_decay,
            )

        self.scheduler = ExponentialLR(
            self.optimizer, gamma=0.97
        )

        # Show number of parameters
        sum = summary(self.model)
        if self.use_wandb:
            wandb.log({
                'Total parameters': sum.total_params,
                'Trainable parameters': sum.trainable_params
                })

    def train(self):
        """Train the model on the training set.

        A checkpoint of the model is saved after each epoch
        and if the validation accuracy is improved upon,
        a separate ckpt is created for use on the test set.
        """
        # load the most recent checkpoint
        if self.resume:
            self.load_checkpoint(best=False)
        # if pretrained weights are provided
        elif self.pretrained_ckpt is not None:
            self.load_pretrained_weights(self.keep_context)

        print(
            "\n[*] Train on {} samples, validate on {} samples".format(
                self.num_train, self.num_valid
            )
        )

        for epoch in range(self.start_epoch, self.epochs):

            print(
                "\nEpoch: {}/{} - LR: {:.6f}".format(
                    epoch + 1, self.epochs, self.optimizer.param_groups[0]["lr"]
                )
            )

            # train for 1 epoch
            train_loss, train_acc = self.train_one_epoch(epoch)

            # evaluate on validation set
            valid_loss, valid_acc = self.validate(epoch)

            # # reduce lr if validation loss plateaus
            self.scheduler.step()

            is_best = valid_acc >= self.best_valid_acc
            msg1 = "train loss: {:.3f} - train acc: {:.3f} "
            msg2 = "- val loss: {:.3f} - val acc: {:.3f} - val err: {:.3f}"
            if is_best:
                self.counter = 0
                msg2 += " [*]"
            msg = msg1 + msg2
            print(
                msg.format(
                    train_loss, train_acc, valid_loss, valid_acc, 100 - valid_acc
                )
            )

            # check for improvement
            if not is_best:
                self.counter += 1
            if self.counter > self.train_patience:
                print("[!] No improvement in a while, stopping training.")
                return
            self.best_valid_acc = max(valid_acc, self.best_valid_acc)
            self.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state": self.model.state_dict(),
                    "optim_state": self.optimizer.state_dict(),
                    "best_valid_acc": self.best_valid_acc,
                },
                is_best,
            )
    def train_one_epoch(self, epoch):
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.

        This is used by train() and should not be called manually.
        """
        self.model.train()
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()
        rews = AverageMeter()

        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for i, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                x, y = batch.pixel_values.to(self.device), batch.labels.to(self.device)

                # initialize location vector and hidden state
                self.batch_size = x.shape[0]                

                # save images
                imgs = []
                imgs.append(x[0:9])

                # initialize variables to store model outputs
                log_pi = []
                baselines = []
                all_log_probas = []
                predicted_digits = []
                total_reward = torch.zeros((self.batch_size, ), device=self.device)
                all_correct = torch.zeros((self.batch_size, ), device=self.device)

                # to keep track of which sequences are still running
                is_running = torch.ones((self.batch_size, ), dtype=torch.bool, device=self.device)
                
                # first iteration: Create context vector, initialize states and
                # get location for first glimpse
                h_t, l_t, b_t, p = self.model(x, None, None, first=True)
                baselines.append(b_t)
                log_pi.append(p)
                
                if x.shape[1] == 3:
                    # if we have a rgb image, convert to grayscale
                    x = x.mean(dim=1)
                    x = x[:, None, :, :]
                
                num_digits = y.shape[-1]
                for d_i in range(num_digits):
                    # exploration steps
                    for t in range(self.explore_steps):
                        # forward pass through model
                        h_t, l_t, b_t, p = self.model(x, l_t, h_t)

                        # mark with ignore_index in order to ignore them during the loss computation
                        p[~is_running], b_t[~is_running] = self.ignore_index, self.ignore_index

                        # store
                        baselines.append(b_t)
                        log_pi.append(p)
                    
                    # prediction steps
                    for t in range(self.num_glimpses - 1):
                        # forward pass through model
                        h_t, l_t, b_t, p = self.model(x, l_t, h_t)

                        # mark with ignore_index in order to ignore them during the loss computation
                        p[~is_running], b_t[~is_running] = self.ignore_index, self.ignore_index

                        # store
                        baselines.append(b_t)
                        log_pi.append(p)

                    # last iteration of current digit
                    h_t, l_t, b_t, log_probas, p = self.model(x, l_t, h_t, last=True)

                    # mark with ignore_index in order to ignore them during the loss computation
                    p[~is_running], b_t[~is_running] = self.ignore_index, self.ignore_index

                    # store
                    baselines.append(b_t)
                    log_pi.append(p)                    
                    all_log_probas.append(log_probas)

                    # get predicted label
                    pred = torch.max(log_probas, 1)[1]

                    # overwrite prediction for already finished sequences
                    pred[~is_running] = self.ignore_index

                    # store predictions
                    predicted_digits.append(pred)

                    # check if sequence has missed the prediction
                    correct = (pred.detach() == y[:, d_i])

                    # Finish sequences that have missed
                    is_running = torch.mul(is_running, correct)

                    # +1 to total reward if the correct label has been predicted
                    total_reward += is_running.float()

                    # check if sequence has reached the end
                    is_running = torch.mul(is_running, y[:, d_i] != self.end_class)

                    # check if sequence has predicted correctly the end label
                    all_correct[torch.mul(y[:, d_i] == self.end_class, correct)] = 1

                    # mark labels of finished sequences as ignore_index to be ignored
                    # during the loss computation
                    y[pred == self.ignore_index, d_i] = self.ignore_index

                    # If all sequences have finished, stop
                    if torch.all(~is_running):
                        break

                # Since the maximum number of labels is five (regardless of the real 
                # number of digits), all sequences that are still running at this point
                # are considered correct
                all_correct[is_running] = 1

                # convert list to tensors and reshape
                baselines = torch.stack(baselines).transpose(1, 0)
                log_pi = torch.stack(log_pi).transpose(1, 0)
                all_log_probas = torch.stack(all_log_probas).transpose(1, 0)
                
                # broadcast reward along new axis
                R = total_reward.unsqueeze(1).repeat(1, baselines.shape[-1])

                # compute losses for differentiable modules
                loss_action = F.nll_loss(
                    all_log_probas.reshape(-1, all_log_probas.shape[2]), 
                    y[:, :d_i+1].reshape(-1),
                    ignore_index=self.ignore_index)
                loss_baseline = F.mse_loss(baselines[baselines != self.ignore_index], R[baselines != self.ignore_index])

                # compute reinforce loss
                # summed over timesteps and averaged across batch
                adjusted_reward = R - baselines.detach()
                loss_mat = -log_pi * adjusted_reward
                # ignore values with ignore_index
                loss_mat[log_pi == self.ignore_index] = 0

                # sum over timesteps
                loss_reinforce = torch.sum(loss_mat, dim=1)                
                loss_reinforce = torch.mean(loss_reinforce, dim=0)

                # sum up into a hybrid loss
                loss = loss_action + loss_baseline + loss_reinforce * self.rl_loss_coef

                # compute accuracy
                acc = 100 * (all_correct.float().sum() / y.shape[0])
                
                # store
                losses.update(loss.item(), self.batch_size)
                accs.update(acc.item(), self.batch_size)
                rews.update(total_reward.mean().item(), self.batch_size)

                # compute gradients and update SGD
                loss.backward()
                self.optimizer.step()

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc - tic)
                
                pbar.set_description(
                    (
                        "{:.1f}s - loss: {:.3f} - acc: {:.3f}".format(
                            (toc - tic), loss.item(), acc.item()
                        )
                    )
                )
                pbar.update(self.batch_size)

            # log to wandb
            if self.use_wandb:
                wandb.log({
                'Train Loss': losses.avg, 'Train Accuracy': accs.avg, 'Train Reward': rews.avg
                })

            return losses.avg, accs.avg
        
    
    @torch.no_grad()
    def validate(self, epoch):
        """Evaluate the DRAM model on the validation set.
        """
        self.model.eval()
        
        # ensure std is set correctly
        self.model.std = self.std
        self.model.locator.std = self.std

        losses = AverageMeter()
        accs = AverageMeter()
        rews = AverageMeter()

        for i, batch in enumerate(self.valid_loader):
            x, y = batch.pixel_values.to(self.device), batch.labels.to(self.device)

            # initialize location vector and hidden state
            self.batch_size = x.shape[0]
            
            # duplicate M times (Monte Carlo sampling)
            x = x.repeat(self.M, 1, 1, 1)    

            # initialize variables to store model outputs
            locs = []
            log_pi = []
            baselines = []
            all_log_probas = []
            predicted_digits = []
            total_reward = torch.zeros((self.batch_size, ), device=self.device)
            all_correct = torch.zeros((self.batch_size, ), device=self.device)

            # to keep track of which sequences are still running
            is_running = torch.ones((self.batch_size, ), dtype=torch.bool, device=self.device)

            # first iteration: Create context vector, initialize states and
            # get location for first glimpse
            h_t, l_t, _, _ = self.model(x, None, None, first=True)
            
            # if we have a rgb image, convert to grayscale
            if x.shape[1] == 3:
                x = x.mean(dim=1)
                x = x[:, None, :, :]
            
            num_digits = y.shape[-1]
            for d_i in range(num_digits):
                # exploration steps
                for t in range(self.explore_steps):
                    # forward pass through model
                    h_t, l_t, b_t, p = self.model(x, l_t, h_t)

                    # mark with ignore_index in order to ignore them during the loss computation
                    p[~is_running], b_t[~is_running] = self.ignore_index, self.ignore_index

                    # store
                    baselines.append(b_t)
                    log_pi.append(p)
                    
                # prediction steps
                for t in range(self.num_glimpses - 1):
                    # forward pass through model
                    h_t, l_t, b_t, p = self.model(x, l_t, h_t)

                    # mark with ignore_index in order to ignore them during the loss computation
                    p[(~is_running).repeat(self.M)], b_t[(~is_running).repeat(self.M)] = self.ignore_index, self.ignore_index

                    # store
                    locs.append(l_t[0:9])
                    baselines.append(b_t)
                    log_pi.append(p)

                # last iteration of current digit
                h_t, l_t, b_t, log_probas, p = self.model(x, l_t, h_t, last=True)

                # mark with ignore_index in order to ignore them during the loss computation
                p[(~is_running).repeat(self.M)], b_t[(~is_running).repeat(self.M)] = self.ignore_index, self.ignore_index

                # store
                locs.append(l_t[0:9])
                baselines.append(b_t)
                log_pi.append(p)
                
                # Monte Carlo sampling for class log-probabilities
                log_probas = log_probas.view(self.M, -1, log_probas.shape[-1])
                log_probas = torch.mean(log_probas, dim=0)
                all_log_probas.append(log_probas)

                # get predicted label
                pred = torch.max(log_probas, 1)[1]

                # overwrite prediction for already finished sequences
                pred[~is_running] = self.ignore_index

                # store predictions
                predicted_digits.append(pred)

                # check if sequence has missed the prediction
                correct = (pred.detach() == y[:, d_i])

                # Finish sequences that have missed
                is_running = torch.mul(is_running, correct)

                # +1 to total reward if the correct label has been predicted
                total_reward += is_running.float()

                # check if sequence has reached the end
                is_running = torch.mul(is_running, y[:, d_i] != self.end_class)

                # check if sequence has predicted correctly the end label
                all_correct[torch.mul(y[:, d_i] == self.end_class, correct)] = 1

                # mark labels of finished sequences as ignore_index to be ignored
                # during the loss computation
                y[pred == self.ignore_index, d_i] = self.ignore_index

                # If all sequences have finished, stop
                if torch.all(~is_running):
                    break

            # Since the maximum number of labels is five (regardless of the real 
            # number of digits), all sequences that are still running at this point
            # are considered correct
            all_correct[is_running] = 1

            # convert list to tensors and reshape
            baselines = torch.stack(baselines).transpose(1, 0)
            log_pi = torch.stack(log_pi).transpose(1, 0)
            all_log_probas = torch.stack(all_log_probas).transpose(1, 0)
            
            # Monte Carlo sampling for baselines and location log probabilities
            baselines = baselines.contiguous().view(self.M, -1, baselines.shape[-1])
            baselines = torch.mean(baselines, dim=0)

            log_pi = log_pi.contiguous().view(self.M, -1, log_pi.shape[-1])
            log_pi = torch.mean(log_pi, dim=0)

            # broadcast reward along new axis
            R = total_reward.unsqueeze(1).repeat(1, baselines.shape[-1])

            # compute losses for differentiable modules
            loss_action = F.nll_loss(
                all_log_probas.reshape(-1, all_log_probas.shape[-1]), 
                y[:, :d_i+1].reshape(-1),
                ignore_index=self.ignore_index)
            loss_baseline = F.mse_loss(baselines[baselines != self.ignore_index], R[baselines != self.ignore_index])

            # compute reinforce loss
            # summed over timesteps and averaged across batch
            adjusted_reward = R - baselines.detach()
            loss_mat = -log_pi * adjusted_reward
            # ignore values with self.ignore_index
            loss_mat[log_pi == self.ignore_index] = 0
            loss_reinforce = torch.sum(loss_mat, dim=1)
            loss_reinforce = torch.mean(loss_reinforce, dim=0)

            # sum up into a hybrid loss
            loss = loss_action + loss_baseline + loss_reinforce * self.rl_loss_coef

            # compute accuracy
            acc = 100 * (all_correct.float().sum() / y.shape[0])

            # store
            losses.update(loss.item(), self.batch_size)
            accs.update(acc.item(), self.batch_size)
            rews.update(total_reward.mean().item(), self.batch_size)

        # log to wandb
        if self.use_wandb:
            wandb.log({
                'Val Loss': losses.avg, 'Val Accuracy': accs.avg, 'Val Reward': rews.avg
                })

        return losses.avg, accs.avg

    @torch.no_grad()
    def test(self):
        """Test the DRAM model.

        This function should only be called at the very
        end once the model has finished training.
        """
        self.model.eval()
        
        # ensure std is set correctly
        self.model.std = self.std
        self.model.locator.std = self.std
        
        accs = AverageMeter()
        rews = AverageMeter()
        total_correct = 0
        results = []

        # load the best checkpoint
        self.load_checkpoint(best=self.best)
        
        # compute the number of floating point operations per forward pass of the model
        # we assume a batch size of 1 and the maximum number of glimpses
        flops = self.compute_flops()
        print(f"Number of FLOPS (GFLOPS): {flops} ({flops/10**9})")        

        for i, batch in enumerate(self.test_loader):
            x, y = batch.pixel_values.to(self.device), batch.labels.clone().to(self.device)

            # initialize location vector and hidden state
            self.batch_size = x.shape[0]
            
            # duplicate M times (Monte Carlo sampling)
            x = x.repeat(self.M, 1, 1, 1) 

            # initialize variables to store model outputs
            all_log_probas = []
            locs = []
            predicted_digits = []
            total_reward = torch.zeros((self.batch_size, ), device=self.device)
            all_correct = torch.zeros((self.batch_size, ), device=self.device)

            # to keep track of which sequences are still running
            is_running = torch.ones((self.batch_size, ), dtype=torch.bool, device=self.device)

            # first iteration: Create context vector, initialize states and
            # get location for first glimpse
            h_t, l_t, _, _ = self.model(x, None, None, first=True)
            # save predicted location
            locs.append(l_t)
            
            # if we have a rgb image, convert to grayscale
            if x.shape[1] == 3:
                x = x.mean(dim=1)
                x = x[:, None, :, :]
            
            num_digits = y.shape[-1]
            for d_i in range(num_digits):
                # exploration steps
                for t in range(self.explore_steps):
                    # forward pass through model
                    h_t, l_t, b_t, p = self.model(x, l_t, h_t)
                    # save predicted location
                    locs.append(l_t)
                
                # prediction steps
                for t in range(self.num_glimpses - 1):
                    # forward pass through model
                    h_t, l_t, b_t, p = self.model(x, l_t, h_t)
                    # save predicted location
                    locs.append(l_t)

                # last iteration of current digit
                h_t, l_t, b_t, log_probas, p = self.model(x, l_t, h_t, last=True)
                # save predicted location
                locs.append(l_t)

                # Monte Carlo sampling for class log-probabilities
                log_probas = log_probas.view(self.M, -1, log_probas.shape[-1])
                log_probas = torch.mean(log_probas, dim=0)
                all_log_probas.append(log_probas)

                # get predicted label
                pred = torch.max(log_probas, 1)[1]

                # overwrite prediction for already finished sequences
                pred[~is_running] = self.ignore_index
                
                # save prediction
                predicted_digits.append(pred)

                # check if sequence has missed the prediction
                correct = (pred.detach() == y[:, d_i])

                # Finish sequences that have missed
                is_running = torch.mul(is_running, correct)

                # +1 to total reward if the correct label has been predicted
                total_reward += is_running.float()

                # check if sequence has reached the end
                is_running = torch.mul(is_running, y[:, d_i] != self.end_class)

                # check if sequence has predicted correctly the end label
                all_correct[torch.mul(y[:, d_i] == self.end_class, correct)] = 1

                # mark labels of finished sequences as ignore_index to be ignored
                # during the loss computation
                y[pred == self.ignore_index, d_i] = self.ignore_index
                
                # If all sequences have finished, stop
                if torch.all(~is_running):
                    break
            
            # Since the maximum number of labels is five (regardless of the real 
            # number of digits), all sequences that are still running at this point
            # are considered correct
            all_correct[is_running] = 1

            # convert list to tensors and reshape
            all_log_probas = torch.stack(all_log_probas).transpose(1, 0)

            # compute accuracy
            acc = 100 * (all_correct.float().sum() / y.shape[0])

            # store
            accs.update(acc.item(), self.batch_size)
            rews.update(total_reward.mean().item(), self.batch_size)
            total_correct += all_correct.float().sum().item()

            # save results
            predicted_digits = torch.stack(predicted_digits).transpose(1, 0)
            
            # Monte Carlo sampling for locations. The predicted location at each
            # time step is the mean of the M locations
            locs = torch.stack(locs).transpose(1, 0)
            locs = locs.contiguous().view(self.M, -1, locs.shape[-2], locs.shape[-1])
            locs = torch.mean(locs, dim=0)
            
            for i in range(self.batch_size):
                results.append({
                    'locs': locs[i].tolist(),
                    'pred': predicted_digits[i].tolist(),
                    'labels': batch.labels[i].tolist(),
                    'reward': total_reward[i].item(),
                    'all_correct': all_correct[i].float().item(),
                    'pixel_values': x[i].tolist()
                })

        perc = accs.avg
        error = 100 - perc
        print(
            "[*] Test Acc: {}/{} ({:.2f}% - {:.2f}%) | Test Reward {}".format(
                total_correct, self.num_test, perc, error, rews.avg
            )
        )
        # log to wandb
        if self.use_wandb:
            wandb.log({
                'Test Accuracy': perc, 'Test Reward': rews.avg
                })

        # Save resuls to file
        if self.save_results:
            self.write_results(results)

        return perc, rews.avg
    
    
    @torch.no_grad()
    def compute_flops(self):
        """Compute the number of flops required for a single sample.
        """
        # start counting FLOPS
        prof = FlopsProfiler(self.model)
        prof.start_profile()
        
        x = torch.zeros((1, 3 if self.image_size == 224 else 1, self.image_size, self.image_size), device=self.device)

        # initialize location vector and hidden state
        self.batch_size = x.shape[0]
        
        # duplicate M times (Monte Carlo sampling)
        x = x.repeat(self.M, 1, 1, 1) 

        # first iteration: Create context vector, initialize states and
        # get location for first glimpse
        h_t, l_t, _, _ = self.model(x, None, None, first=True)
        
        # if we have a rgb image, convert to grayscale
        if x.shape[1] == 3:
            x = x.mean(dim=1)
            x = x[:, None, :, :]
        
        for d_i in range(6):
            # exploration steps
            for t in range(self.explore_steps):
                # forward pass through model
                h_t, l_t, _, _ = self.model(x, l_t, h_t)
                
            # prediction steps
            for t in range(self.num_glimpses - 1):
                # forward pass through model
                h_t, l_t, _, _ = self.model(x, l_t, h_t)

            # last iteration of current digit
            h_t, l_t, _, _, _ = self.model(x, l_t, h_t, last=True)
            
        prof.stop_profile()
        flops = prof.get_total_flops()
        prof.end_profile()

        return flops
    
    
    def write_results(self, results):
        filename = self.model_name + "_results.json"
        res_path = os.path.join(self.ckpt_dir, filename)
        
        with open(res_path, "w") as f:
            json.dump(results, f)

            
    def save_checkpoint(self, state, is_best):
        """Saves a checkpoint of the model.

        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        filename = self.model_name + "_ckpt.pth.tar"
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)
        if is_best:
            filename = self.model_name + "_model_best.pth.tar"
            shutil.copyfile(ckpt_path, os.path.join(self.ckpt_dir, filename))

    def load_checkpoint(self, best=False):
        """Load the best copy of a model.

        This is useful for 2 cases:
        - Resuming training with the most recent model checkpoint.
        - Loading the best validation model to evaluate on the test data.

        Args:
            best: if set to True, loads the best model.
                Use this if you want to evaluate your model
                on the test data. Else, set to False in which
                case the most recent version of the checkpoint
                is used.
        """
        print("[*] Loading model from {}".format(self.ckpt_dir))

        filename = self.model_name + "_ckpt.pth.tar"
        if best:
            filename = self.model_name + "_model_best.pth.tar"
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)

        # load variables from checkpoint
        self.start_epoch = ckpt["epoch"]
        self.best_valid_acc = ckpt["best_valid_acc"]
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])

        if best:
            print(
                "[*] Loaded {} checkpoint @ epoch {} "
                "with best valid acc of {:.3f}".format(
                    filename, ckpt["epoch"], ckpt["best_valid_acc"]
                )
            )
        else:
            print("[*] Loaded {} checkpoint @ epoch {}".format(filename, ckpt["epoch"]))

    def load_pretrained_weights(self, keep_context=False):
        """Load the best copy of a model.

        This is useful for 2 cases:
        - Resuming training with the most recent model checkpoint.
        - Loading the best validation model to evaluate on the test data.

        Args:
            best: if set to True, loads the best model.
                Use this if you want to evaluate your model
                on the test data. Else, set to False in which
                case the most recent version of the checkpoint
                is used.
        """
        print("[*] Loading pretrained weights from {}".format(self.pretrained_ckpt))

        filename = self.model_name + "_model_best.pth.tar"
        ckpt_path = os.path.join(self.pretrained_ckpt, filename)
        ckpt = torch.load(ckpt_path)["model_state"]
        
        # Ignore context weights
        new_ckpt = ckpt.copy() # Need to make a copy since ckpt is an ordered Dict
        if not keep_context:
            for k in ckpt.keys():
                if "context" in k:
                    new_ckpt.pop(k, None)

        # load weights from checkpoint
        self.model.load_state_dict(new_ckpt, strict=False)