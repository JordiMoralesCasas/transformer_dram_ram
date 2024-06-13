import torch

import trainers.utils as utils
import data.data_loaders as data_loaders

from trainers.trainer_mnist import MNISTTrainer
from trainers.trainer_svhn import SVHNTrainer
from config import get_config

import wandb
import pprint


def main():
    # Init WanDB
    wandb.init(project="svhn_zoom", entity="mcv_jordi")

    # Get config from gridsearch
    config, unparsed = get_config()
    wandb_config = wandb.config

    config.init_lr = wandb_config.init_lr
    config.momentum = wandb_config.momentum
    config.epochs = wandb_config.epochs
    config.optimizer = wandb_config.optimizer
    config.std = wandb_config.std
    config.preprocess = wandb_config.preprocess
    config.weight_decay = wandb_config.weight_decay
    config.core_type = wandb_config.core_type
    config.batch_size = wandb_config.batch_size
    config.cell_size = wandb_config.cell_size
    config.hidden_size = wandb_config.hidden_size
    config.patch_size = wandb_config.patch_size
    config.inner_size = wandb_config.inner_size
    config.n_heads = wandb_config.n_heads
    config.num_glimpses = wandb_config.num_glimpses
    config.rl_loss_coef = wandb_config.rl_loss_coef


    # Ensure dirs exist
    utils.prepare_dirs(config)

    # ensure reproducibility
    torch.manual_seed(config.random_seed)
    kwargs = {}
    if config.use_gpu:
        torch.cuda.manual_seed(config.random_seed)
        kwargs = {"num_workers": config.num_workers, "pin_memory": True}

    # instantiate data loaders
    if config.dataset == "mnist":
        # Load MNIST 
        train_dloader = None
        if config.is_train:
            train_dloader = data_loaders.get_train_valid_loader_mnist(
                config.data_dir,
                config.batch_size,
                config.random_seed,
                config.valid_size,
                config.shuffle,
                config.show_sample,
                **kwargs,
            )
        test_dloader = data_loaders.get_test_loader_mnist(
            config.data_dir, config.batch_size, **kwargs,
        )
        # Initialize Trainer for MNIST Dataset
        trainer = MNISTTrainer(config, train_loader=train_dloader, test_loader=test_dloader)
    else:
        # Load SVHN
        config.data_dir = "/data/users/jmorales/svhn/"
        train_dloader = None
        if config.is_train:
            train_dloader = data_loaders.get_train_valid_loader_svhn(
                config.data_dir,
                config.batch_size,
                config.random_seed,
                config.show_sample,
                do_preprocessing=config.preprocess,
                **kwargs,
            )
        test_dloader = data_loaders.get_test_loader_svhn(
            config.data_dir, config.batch_size, do_preprocessing=config.preprocess,**kwargs,
        )
        # Initialize Trainer for SVHN Dataset
        trainer = SVHNTrainer(config, train_loader=train_dloader, test_loader=test_dloader, is_gridsearch=True)

    # either train
    if config.is_train:
        utils.save_config(config)
        trainer.train()
        acc, reward = trainer.test()
        return reward
    # or load a pretrained model and test
    else:
        trainer.test()


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)

    # WanDB login
    wandb.login()
    
    # Hyperparameter search config
    sweep_config = {
        'name': 'LSTM SVHN Sweep',
        'method': 'random',
        'metric': {
            'name': 'Test Reward',
            'goal': 'maximize'   
            },
        'parameters':  {
            'core_type': {
                'value': "rnn"},
            'epochs': {
                'value': 10},
            'batch_size': {
                'value': 128},
            'cell_size': {
                'value': 512},
            'hidden_size': {
                'value': 1024},
            'preprocess': {
                'value': True},
            'num_glimpses': {
                'value': 3},
            'optimizer': {
                'values': ["sgd", "adamw"]},
            'weight_decay': {
                'value': 0.01},
            'std': {
                'value': 0.01},
            'rl_loss_coef': {
                'values': [0.01, 0.5, 1]},
            'momentum': {
                'value': 0.9},
            'patch_size': {
                'value': 28},
            'init_lr': {
                'values': [
                    0.01, 0.025, 0.0075
                    ]}, 
            'inner_size': {
                'values': [
                    1024, # Not used
                    ]},
            'n_heads': {
                'values': [
                    1, # Not used
                    ]}
        } 
    }
    
    
    """sweep_config = {
        'name': 'Transformer TrXL SVHN 2nd Sweep',
        'method': 'random',
        'metric': {
            'name': 'Test Reward',
            'goal': 'maximize'   
            },
        'parameters':  {
            'core_type': {
                'value': "transformer"},
            'epochs': {
                'value': 5},
            'batch_size': {
                'value': 128},
            'cell_size': {
                'value': 512},
            'hidden_size': {
                'value': 1024},
            'preprocess': {
                'value': True},
            'num_glimpses': {
                'value': 3},
            'optimizer': {
                'value': "adamw"},
            'weight_decay': {
                'values': [0.01, 0.001, 0]},
            'std': {
                'values': [0.01, 0.03, 0.05]},
            'momentum': {
                'value': 0.0}, # Not used
            'patch_size': {
                'value': 28},
            'init_lr': {
                'values': [
                    0.00002, 0.00001, 0.000005, 0.000001,
                    ]}, 
            'inner_size': {
                'values': [
                    1024,
                    ]},
            'n_heads': {
                'values': [
                    1,
                    ]}
        } 
    }"""

    """sweep_config = {
        'name': 'Transformer GPT2 SVHN 2nd Sweep',
        'method': 'random',
        'metric': {
            'name': 'Test Reward',
            'goal': 'maximize'   
            },
        'parameters':  {
            'core_type': {
                'value': "transformer"},
            'epochs': {
                'value': 5},
            'batch_size': {
                'value': 128},
            'cell_size': {
                'value': 512},
            'hidden_size': {
                'value': 1024},
            'preprocess': {
                'value': True},
            'num_glimpses': {
                'value': 3},
            'optimizer': {
                'value': "adamw"},
            'weight_decay': {
                'values': [0.01, 0.001, 0]},
            'std': {
                'values': [0.01, 0.03, 0.05]},
            'momentum': {
                'value': 0.0}, # Not used
            'patch_size': {
                'value': 28},
            'init_lr': {
                'values': [
                    0.00002, 0.00001, 0.000005, 0.000001,
                    ]}, 
            'inner_size': {
                'values': [
                    1024,
                    ]},
            'n_heads': {
                'values': [
                    1,
                    ]}
        } 
    }"""
    
    
    
    pprint.pprint(sweep_config)

    sweep_id = wandb.sweep(sweep_config, project="svhn_zoom", entity="mcv_jordi")
    wandb.agent(sweep_id, function=main, count=40)

    """
    import multiprocessing
    def call_agent(id):
        wandb.agent(id, function=main, count=2)

    pool_obj = multiprocessing.Pool(2)
    pool_obj.map()
    """