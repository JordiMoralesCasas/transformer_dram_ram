import os
import time
import shutil
import pickle
import json

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR
import wandb

from modelling.model import DeepRecurrentAttention
from trainers.utils import AverageMeter, NLSScorer, AccScorer

class DocILETrainer:
    def __init__(
        self, 
        config, 
        train_loader=None, 
        val_loader=None, 
        test_loader=None, 
        device=None,
        tokenizer=None,
        is_gridsearch=False
    ):
        """
        Construct a new Trainer instance.

        Args:
            config: object containing command line arguments.
            data_loader: A data iterator.
        """
        self.config = config

        self.device = device

        # glimpse network params
        self.patch_size = config.patch_size

        # core network params
        self.num_glimpses = config.num_glimpses
        self.hidden_size = config.hidden_size

        # reinforce params
        self.std = config.std

        # data params
        if train_loader is not None:
            self.train_loader = train_loader
            self.valid_loader = val_loader
            self.num_train = len(self.train_loader.dataset)
            self.num_valid = len(self.valid_loader.dataset)
            
        self.test_loader = test_loader
        self.num_test = len(self.test_loader.dataset)

        self.vocab_size = config.vocab_size
        self.num_channels = 1
        self.ignore_index = config.ignore_index
        self.pad_token_id = config.pad_token_id
        self.decoder_start_token_id = config.start_token_id
        self.eos_token_id = config.eos_token_id

        # training params
        self.epochs = config.epochs
        self.start_epoch = 0
        self.lr = config.init_lr
        self.rl_loss_coef = config.rl_loss_coef

        # misc params
        self.best = config.best
        self.ckpt_dir = config.ckpt_dir
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        self.best_valid_nls = 0.0
        self.counter = 0
        self.train_patience = config.train_patience
        self.save_results = config.save_results
        self.wandb_name = config.wandb_name
        self.use_wandb = True if (self.wandb_name or is_gridsearch) else False
        self.resume = config.resume
        self.print_freq = config.print_freq
        self.plot_freq = config.plot_freq
        self.model_name = "LMDRAM_{}_{}".format(
            config.num_glimpses, config.patch_size
        )
        
        # Since we are working with a huge dataset (DocILE), we log metrics after
        # x% of training samples instead of only every epoch.
        self.log_mid_training = False
        self.logging_percentage = 0.1

        self.plot_dir = "./plots/" + self.model_name + "/"
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        # configure wandb logging
        if self.use_wandb and not is_gridsearch:
            wandb.init(
                entity="mcv_jordi",
                project="DocILE_LMDRAM", 
                name=self.wandb_name,
                config=config)

        # create scoring function
        self.threshold = 1.0
        self.nls_scorer = NLSScorer(
            tokenizer=tokenizer, 
            threshold=0
        )

        self.acc_scorer = AccScorer(
            pad_token_id=self.pad_token_id
        )

        # Build ZoomVQA
        self.model = DeepRecurrentAttention(
            self.patch_size,
            config.num_patches,
            config.glimpse_scale,
            self.num_channels,
            config.glimpse_hidden,
            config.loc_hidden,
            self.std,
            self.hidden_size,
            config.cell_size,
            config.inner_size,
            config.n_heads,
            None,
            config.core_type,
            self.device,
            config.transformer_model,
            config.max_length
        )
        self.model.to(self.device)

        # initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.init_lr, weight_decay=self.config.weight_decay,
        )

        self.scheduler_rate = 0.97
        self.scheduler = ExponentialLR(
            self.optimizer, gamma=self.scheduler_rate
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

        print(
            "\n[*] Train on {} samples, validate on {} samples".format(
                self.num_train, self.num_valid
            )
        )

        for epoch in range(self.start_epoch, self.epochs):

            print(
                "\nEPOCH: {}/{} - LR: {:.6f}".format(
                    epoch + 1, self.epochs, self.optimizer.param_groups[0]["lr"]
                )
            )

            # train for 1 epoch
            train_loss, train_acc = self.train_one_epoch(epoch)

            # evaluate on validation set
            valid_nls = self.validate(epoch)

            # # reduce lr if validation loss plateaus
            self.scheduler.step()

            is_best = valid_nls > self.best_valid_nls
            msg1 = "\t train loss: {:.3f} - train acc: {:.3f} \n"
            msg2 = "\t val NLS: {:.3f} \n"
            if is_best:
                self.counter = 0
                msg2 += " [*]"
            msg = msg1 + msg2
            print(
                msg.format(
                    train_loss, train_acc, valid_nls
                )
            )

            # check for improvement
            if not is_best:
                self.counter += 1
            if self.counter > self.train_patience:
                print("[!] No improvement in a while, stopping training.")
                return
            self.best_valid_nls = max(valid_nls, self.best_valid_nls)
            self.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state": self.model.state_dict(),
                    "optim_state": self.optimizer.state_dict(),
                    "best_valid_nls": self.best_valid_nls,
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
        losses_epoch = AverageMeter()
        accs_epoch = AverageMeter()
        rewards_epoch = AverageMeter()
        
        if self.log_mid_training and self.use_wandb:
            losses = AverageMeter()
            accs = AverageMeter()
            rewards = AverageMeter()

        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for i, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                # Move inputs to device
                pixel_values = batch.pixel_values.to(self.device)
                label_ids = batch.label_ids.to(self.device)
                decoder_attention_mask = batch.decoder_attention_mask.to(self.device)
                answer_bbox = batch.bbox.to(self.device)

                # initialize location vector and hidden state
                self.batch_size = pixel_values.shape[0]

                # save images
                imgs = []
                imgs.append(pixel_values[0:9])

                # extract the glimpses
                locs = []
                log_pi = []
                baselines = []

                # Reset buffers
                self.model.reset_transformers_buffer(self.batch_size)

                # Init encoders
                h_t, l_t, b_t, p = self.model(pixel_values, None, None, first=True)
                locs.append(l_t)
                baselines.append(b_t)
                log_pi.append(p)

                for t in range(self.num_glimpses):
                    # forward pass through model
                    h_t, l_t, b_t, p = self.model(pixel_values, l_t, h_t)

                    # store
                    locs.append(l_t)
                    baselines.append(b_t)
                    log_pi.append(p)

                # last iteration
                lm_logits = self.model.lm_answer(
                    label_ids=label_ids, 
                    decoder_attention_mask=decoder_attention_mask, 
                    pad_token_id=self.pad_token_id,
                    decoder_start_token_id=self.decoder_start_token_id
                    )

                # convert list to tensors and reshape
                baselines = torch.stack(baselines).transpose(1, 0)
                log_pi = torch.stack(log_pi).transpose(1, 0)

                # Get accuracy
                pred_ids = torch.argmax(lm_logits, dim=-1)
                acc = self.acc_scorer(pred_ids, label_ids)
                
                # Answer bbox area covered by glimpses
                locs = torch.stack(locs).transpose(1, 0)
                locs = locs[:, :-1, :]

                # compute reward
                reward = torch.where(acc > self.threshold, acc, 0)
                R = reward.unsqueeze(1).repeat(1, self.num_glimpses+1)

                # compute losses for differentiable modules
                label_ids[label_ids == self.pad_token_id] = self.ignore_index
                loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction="mean")
                loss_lm = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), label_ids.view(-1))

                loss_baseline = F.mse_loss(baselines, R)

                # compute reinforce loss
                # summed over timesteps and averaged across batch
                adjusted_reward = R - baselines.detach()
                loss_reinforce = torch.sum(-log_pi * adjusted_reward, dim=1)
                loss_reinforce = torch.mean(loss_reinforce, dim=0)

                # sum up into a hybrid loss
                loss = loss_lm + loss_baseline + loss_reinforce * self.rl_loss_coef

                # store
                losses_epoch.update(loss.item(), self.batch_size)
                accs_epoch.update(acc.mean().item(), self.batch_size)
                rewards_epoch.update(reward.mean().item(), self.batch_size)
                
                if self.log_mid_training and self.use_wandb:
                    losses.update(loss.item(), self.batch_size)
                    accs.update(acc.mean().item(), self.batch_size)
                    rewards.update(reward.mean().item(), self.batch_size)
                    
                    if (i+1) % int(len(self.train_loader)*self.logging_percentage) == 0 and self.use_wandb:
                        # after x% of training samlpes, log metrics to wandb
                        wandb.log({
                            'Train Loss': losses.avg, 'Train Accuracy': accs.avg, 'Train Reward': rewards.avg
                        })
                        
                        # reset meters
                        losses = AverageMeter()
                        accs = AverageMeter()
                        rewards = AverageMeter()

                # compute gradients and update SGD
                loss.backward()
                self.optimizer.step()

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc - tic)

                pbar.set_description(
                    (
                        "TRAINING | {:.1f}s - loss: {:.3f} - acc: {:.3f} - reward: {:.3f}".format(
                            (toc - tic), loss.item(), acc.mean().item(), reward.mean().item()
                        )
                    )
                )
                pbar.update(self.batch_size)

            # log to wandb
            if self.use_wandb:
                wandb.log({
                'Train Loss (Whole epoch)': losses_epoch.avg, 'Train Accuracy (Whole epoch)': accs_epoch.avg, 'Train Reward (Whole epoch)': rewards_epoch.avg,
                })                        

            return losses_epoch.avg, accs_epoch.avg

    @torch.no_grad()
    def validate(self, epoch):
        """Evaluate the model on the validation set.
        """
        self.model.eval()

        accs = AverageMeter()
        nlss = AverageMeter()
        rewards = AverageMeter()

        with tqdm(total=self.num_valid) as pbar:
            for i, batch in enumerate(self.valid_loader):
                # Move inputs to device
                pixel_values = batch.pixel_values.to(self.device)
                label_ids = batch.label_ids.to(self.device)
                answer_bbox = batch.bbox.to(self.device)

                # initialize location vector and hidden state
                self.batch_size = pixel_values.shape[0]

                # Reset buffers
                self.model.reset_transformers_buffer(self.batch_size)
                
                # save glimpse location
                locs = []

                # Init encoders
                h_t, l_t, b_t, p = self.model(pixel_values, None, None, first=True)
                locs.append(l_t)

                for t in range(self.num_glimpses):
                    # forward pass through model
                    h_t, l_t, b_t, p = self.model(pixel_values, l_t, h_t)
                    locs.append(l_t)

                # last iteration
                pred_ids, _ = self.model.infer_answer(self.batch_size, self.decoder_start_token_id, self.eos_token_id, self.pad_token_id)
                
                # Answer bbox area covered by glimpses
                locs = torch.stack(locs).transpose(1, 0)
                locs = locs[:, :-1, :] # last glimpse location can be ignored (not used by the model)

                # Get accuracy and Normalized Levenshtein Score
                acc = self.acc_scorer(pred_ids, label_ids)
                nls = self.nls_scorer(pred_ids, label_ids)
                
                # calculate reward
                reward = torch.where(acc >= self.threshold, acc, 0)

                # store
                rewards.update(reward.mean().item(), self.batch_size)
                accs.update(acc.mean().item(), self.batch_size)
                nlss.update(nls.mean().item(), self.batch_size)
                
                # update progress bar
                pbar.set_description("VALIDATION")
                pbar.update(self.batch_size)

            # log to wandb
            if self.use_wandb:
                wandb.log({
                'Val Accuracy': accs.avg, 'Val Reward': rewards.avg, 'Val NLS': nlss.avg
                })
                

        return nlss.avg

    @torch.no_grad()
    def test(self):
        """Test the model on the test set.

        This function should only be called at the very
        end once the model has finished training.
        """
        self.model.eval()

        accs = AverageMeter()
        nlss = AverageMeter()
        rewards = AverageMeter()
        results = []

        
        with tqdm(total=self.num_test) as pbar:
            for i, (batch, original_paths) in enumerate(self.test_loader):
                # Move inputs to device
                pixel_values = batch.pixel_values.to(self.device)
                label_ids = batch.label_ids.to(self.device)
                answer_bbox = batch.bbox.to(self.device)
                # initialize location vector and hidden state
                self.batch_size = pixel_values.shape[0]

                # extract the glimpses
                locs = []

                # Reset buffers
                self.model.reset_transformers_buffer(self.batch_size)

                # Init encoders
                h_t, l_t, b_t, p = self.model(pixel_values, None, None, first=True)
                # store
                locs.append(l_t.clone())

                for t in range(self.num_glimpses):
                    # forward pass through model
                    h_t, l_t, b_t, p = self.model(pixel_values, l_t, h_t)
                    # store
                    locs.append(l_t.clone())

                # last iteration
                pred_ids, _ = self.model.infer_answer(self.batch_size, self.decoder_start_token_id, self.eos_token_id, self.pad_token_id)
                
                # Answer bbox area covered by glimpses
                locs = torch.stack(locs).transpose(1, 0)
                locs = locs[:, :-1, :] # last glimpse location can be ignored (not used by the model)

                # Get accuracy and Normalized Levenshtein Score
                acc = self.acc_scorer(pred_ids, label_ids)
                nls = self.nls_scorer(pred_ids, label_ids)
                
                # calculate reward
                reward = torch.where(acc >= self.threshold, acc, 0)

                # store
                rewards.update(reward.mean().item(), self.batch_size)
                accs.update(acc.mean().item(), self.batch_size)
                nlss.update(nls.mean().item(), self.batch_size)

                # save results
                for i in range(self.batch_size):
                    results.append({
                        "locs": locs[i].tolist(), # last location produced by the model is not actually used to input a glimpse
                        "pred": pred_ids[i].tolist(),
                        "labels": label_ids[i].tolist(),
                        "acc": acc[i].item(),
                        "reward": reward[i].item(),
                        "nls": nls[i].item(),
                        "img_path": original_paths[i],
                        "answer_bbox": answer_bbox[i].tolist()
                    })
                    
                # update progress bar
                pbar.set_description("TESTING")
                pbar.update(self.batch_size)

        print(
            "[*] Test NLS: {:.2f}% - {:.2f}%".format(
                nlss.avg*100, (1 - nlss.avg)*100
            )
        )

        # log to wandb
        if self.use_wandb:
            wandb.log({
            'Test Reward': rewards.avg, 'Test NLS': nlss.avg, 'Test Accuracy': accs.avg
            })

        # Save resuls to file
        if self.save_results:
            self.write_results(results)

        return nlss.avg
    

    def write_results(self, results):
        filename = self.model_name + "_results.json"
        res_path = os.path.join(self.ckpt_dir, filename)
        
        with open(res_path, "w") as f:
            json.dump(results, f)


    def save_checkpoint(self, state, is_best):
        """Saves a checkpoint of the model.

        If this model has reached the best validation NLS thus
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
        self.best_valid_nls = ckpt["best_valid_nls"]
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])

        if best:
            print(
                "[*] Loaded {} checkpoint @ epoch {} "
                "with best valid NLS of {:.3f}".format(
                    filename, ckpt["epoch"], ckpt["best_valid_nls"]
                )
            )
        else:
            print("[*] Loaded {} checkpoint @ epoch {}".format(filename, ckpt["epoch"]))
