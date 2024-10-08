import os
import time
import shutil
import pickle
import json
from flops_profiler.profiler import FlopsProfiler

import torch
import torch.nn.functional as F
from torchsummary import summary

from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

from modelling.model import RecurrentAttention
from trainers.utils import AverageMeter


class MNISTTrainer:
    """A Recurrent Attention Model trainer.

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
        self.num_glimpses = config.num_glimpses
        self.hidden_size = config.hidden_size

        # reinforce params
        self.std = config.std
        self.M = config.M

        # data params
        if train_val_loader is not None:
            self.train_loader = train_val_loader[0]
            self.valid_loader = train_val_loader[1]
            self.num_train = len(self.train_loader.sampler.indices)
            self.num_valid = len(self.valid_loader.sampler.indices)
        if test_loader is not None:
            self.test_loader = test_loader
            self.num_test = len(self.test_loader.dataset)
        self.num_classes = 10
        self.num_channels = 1

        # training params
        self.epochs = config.epochs
        self.start_epoch = 0
        self.momentum = config.momentum
        self.lr = config.init_lr

        # misc params
        self.best = config.best
        self.ckpt_dir = config.ckpt_dir
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

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
        self.model_name = "ram_{}_{}x{}_{}".format(
            config.num_glimpses,
            config.patch_size,
            config.patch_size,
            config.glimpse_scale,
        )

        # configure wandb logging
        if self.use_wandb and not is_gridsearch:
            # WANDB CONFIG
            wandb.init(
                entity=config.wandb_entity,
                project=config.wandb_project, 
                name=self.wandb_name,
                config=config)

        # build RAM model
        self.model = RecurrentAttention(
            self.patch_size,
            self.num_patches,
            self.glimpse_scale,
            self.num_channels,
            self.loc_hidden,
            self.glimpse_hidden,
            self.std,
            self.hidden_size,
            self.num_classes,
            self.core_type,
            self.transformer_model        
        )
        self.model.to(self.device)

        # initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.init_lr, weight_decay=config.weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, "min", patience=self.lr_patience
        )

        # Show number of parameters
        summary(self.model)
        # Show number of parameters
        sum = summary(self.model)
        if self.use_wandb:
            wandb.log({
                'Total parameters': sum.total_params,
                'Trainable parameters': sum.trainable_params
                })

    def reset(self):
        h_t = torch.zeros(
            self.batch_size,
            self.hidden_size,
            dtype=torch.float,
            device=self.device,
            requires_grad=True,
        )
        l_t = torch.FloatTensor(self.batch_size, 2).uniform_(-1, 1).to(self.device)
        l_t.requires_grad = True

        return h_t, l_t

    def train(self):
        """Train the model on the training set.

        A checkpoint of the model is saved after each epoch
        and if the validation accuracy is improved upon,
        a separate ckpt is created for use on the test set.
        """
        # load the most recent checkpoint
        if self.resume:
            self.load_checkpoint(best=False)

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
            self.scheduler.step(-valid_acc)

            is_best = valid_acc > self.best_valid_acc
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

        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for i, (x, y) in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                x, y = x.to(self.device), y.to(self.device)

                plot = False
                if (epoch % self.plot_freq == 0) and (i == 0):
                    plot = True

                # initialize location vector and hidden state
                self.batch_size = x.shape[0]
                
                # initialize variables to store model outputs
                log_pi = []
                baselines = []
                
                # get initial hidden state and random location
                h_t, l_t = self.reset()

                # initialize transformer's buffer for past glimpses
                if self.core_type == "transformer":
                    self.model.reset_glimpse_buffer(self.batch_size, self.device)
                
                for t in range(self.num_glimpses - 1):
                    # forward pass through model
                    h_t, l_t, b_t, p = self.model(x, l_t, h_t)

                    # store
                    baselines.append(b_t)
                    log_pi.append(p)

                # last iteration
                h_t, l_t, b_t, log_probas, p = self.model(x, l_t, h_t, last=True)
                log_pi.append(p)
                baselines.append(b_t)

                # convert list to tensors and reshape
                baselines = torch.stack(baselines).transpose(1, 0)
                log_pi = torch.stack(log_pi).transpose(1, 0)

                # calculate reward
                predicted = torch.max(log_probas, 1)[1]
                R = (predicted.detach() == y).float()
                R = R.unsqueeze(1).repeat(1, self.num_glimpses)

                # compute losses for differentiable modules
                loss_action = F.nll_loss(log_probas, y)
                loss_baseline = F.mse_loss(baselines, R)

                # compute reinforce loss
                # summed over timesteps and averaged across batch
                adjusted_reward = R - baselines.detach()
                loss_reinforce = torch.sum(-log_pi * adjusted_reward, dim=1)
                loss_reinforce = torch.mean(loss_reinforce, dim=0)

                # sum up into a hybrid loss
                loss = loss_action + loss_baseline + loss_reinforce * 0.01

                # compute accuracy
                correct = (predicted == y).float()
                acc = 100 * (correct.sum() / len(y))

                # store
                losses.update(loss.item(), x.size()[0])
                accs.update(acc.item(), x.size()[0])

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
                'Train Loss': losses.avg, 'Train Accuracy': accs.avg
                })

            return losses.avg, accs.avg

    @torch.no_grad()
    def validate(self, epoch):
        """Evaluate the RAM model on the validation set.
        """
        losses = AverageMeter()
        accs = AverageMeter()

        for i, (x, y) in enumerate(self.valid_loader):
            x, y = x.to(self.device), y.to(self.device)

            # duplicate M times
            x = x.repeat(self.M, 1, 1, 1)

            # initialize location vector and hidden state
            self.batch_size = x.shape[0]
            h_t, l_t = self.reset()

            # initialize transformer's buffer for past glimpses
            if self.core_type == "transformer":
                self.model.reset_glimpse_buffer(self.batch_size, self.device)

            # extract the glimpses
            log_pi = []
            baselines = []
            for t in range(self.num_glimpses - 1):
                # forward pass through model
                h_t, l_t, b_t, p = self.model(x, l_t, h_t)

                # store
                baselines.append(b_t)
                log_pi.append(p)

            # last iteration
            h_t, l_t, b_t, log_probas, p = self.model(x, l_t, h_t, last=True)
            log_pi.append(p)
            baselines.append(b_t)

            # convert list to tensors and reshape
            baselines = torch.stack(baselines).transpose(1, 0)
            log_pi = torch.stack(log_pi).transpose(1, 0)

            # average
            log_probas = log_probas.view(self.M, -1, log_probas.shape[-1])
            log_probas = torch.mean(log_probas, dim=0)

            baselines = baselines.contiguous().view(self.M, -1, baselines.shape[-1])
            baselines = torch.mean(baselines, dim=0)

            log_pi = log_pi.contiguous().view(self.M, -1, log_pi.shape[-1])
            log_pi = torch.mean(log_pi, dim=0)

            # calculate reward
            predicted = torch.max(log_probas, 1)[1]
            R = (predicted.detach() == y).float()
            R = R.unsqueeze(1).repeat(1, self.num_glimpses)

            # compute losses for differentiable modules
            loss_action = F.nll_loss(log_probas, y)
            loss_baseline = F.mse_loss(baselines, R)

            # compute reinforce loss
            adjusted_reward = R - baselines.detach()
            loss_reinforce = torch.sum(-log_pi * adjusted_reward, dim=1)
            loss_reinforce = torch.mean(loss_reinforce, dim=0)

            # sum up into a hybrid loss
            loss = loss_action + loss_baseline + loss_reinforce * 0.01

            # compute accuracy
            correct = (predicted == y).float()
            acc = 100 * (correct.sum() / len(y))

            # store
            losses.update(loss.item(), x.size()[0])
            accs.update(acc.item(), x.size()[0])

        # log to wandb
        if self.use_wandb:
            wandb.log({
                'Val Loss': losses.avg, 'Val Accuracy': accs.avg
                })

        return losses.avg, accs.avg

    @torch.no_grad()
    def test(self):
        """Test the RAM model.

        This function should only be called at the very
        end once the model has finished training.
        """
        correct = 0
        results = []

        # load the best checkpoint
        self.load_checkpoint(best=self.best)
        
        # compute the number of floating point operations per forward pass of the model
        # we assume a batch size of 1 and the maximum number of glimpses
        flops = self.compute_flops()
        print(f"Number of FLOPS (GFLOPS): {flops} ({flops/10**9})")

        for i, (x, y) in enumerate(self.test_loader):
            x, y = x.to(self.device), y.to(self.device)

            # duplicate M times
            x = x.repeat(self.M, 1, 1, 1)
            
            # initialize list to store predicted locations
            locs = []

            # initialize location vector and hidden state
            self.batch_size = x.shape[0]
            h_t, l_t = self.reset()
            locs.append(l_t)

            # initialize transformer's buffer for past glimpses
            if self.core_type == "transformer":
                self.model.reset_glimpse_buffer(self.batch_size, self.device)

            # extract the glimpses
            for t in range(self.num_glimpses - 1):
                # forward pass through model
                h_t, l_t, b_t, p = self.model(x, l_t, h_t)
                locs.append(l_t)

            # last iteration
            h_t, l_t, b_t, log_probas, p = self.model(x, l_t, h_t, last=True)

            log_probas = log_probas.view(self.M, -1, log_probas.shape[-1])
            log_probas = torch.mean(log_probas, dim=0)

            pred = log_probas.data.max(1, keepdim=True)[1]
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()
            
            # Monte Carlo sampling for locations. The predicted location at each
            # time step is the mean of the M locations
            locs = torch.stack(locs).transpose(1, 0)
            locs = locs.contiguous().view(self.M, -1, locs.shape[-2], locs.shape[-1])
            locs = torch.mean(locs, dim=0)
            
            for i in range(self.batch_size):
                results.append({
                    "locs": locs[i].tolist(),
                    "pred": pred[i].item(),
                    "label": y[i].item(),
                    "pixel_values": x[i].tolist()
                })

        perc = (100.0 * correct) / (self.num_test)
        error = 100 - perc
        print(
            "[*] Test Acc: {}/{} ({:.2f}% - {:.2f}%)".format(
                correct, self.num_test, perc, error
            )
        )
        # log to wandb
        if self.use_wandb:
            wandb.log({
                'Test Accuracy': perc
                })
            
        # Save resuls to file
        if self.save_results:
            self.write_results(results)
      
            
    @torch.no_grad()
    def compute_flops(self):
        """Compute the number of flops required for a single sample.
        """
        # start counting FLOPS
        prof = FlopsProfiler(self.model)
        prof.start_profile()
        
        x = torch.zeros((1, 1, 28, 28), device=self.device)

        # duplicate M times
        x = x.repeat(self.M, 1, 1, 1)

        # initialize location vector and hidden state
        self.batch_size = x.shape[0]
        h_t, l_t = self.reset()

        # initialize transformer's buffer for past glimpses
        if self.core_type == "transformer":
            self.model.reset_glimpse_buffer(self.batch_size, self.device)

        for t in range(self.num_glimpses - 1):
            # forward pass through model
            h_t, l_t, _, _ = self.model(x, l_t, h_t)
        # last iteration
        self.model(x, l_t, h_t, last=True)
        
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
