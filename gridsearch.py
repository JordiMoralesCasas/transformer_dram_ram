import torch

import trainers.utils as utils
import data.data_loaders as data_loaders

from trainers.trainer_mnist import MNISTTrainer
from trainers.trainer_svhn import SVHNTrainer
from trainers.trainer_multinumber import MultiNumberTrainer
from config import get_config

import wandb
import pprint

# WandB config
PROJECT = None
ENTITY = None


def main():
    # Get config from gridsearch
    config, unparsed = get_config()
    
    # Init WanDB
    wandb.init(project=config.wandb_project, entity=config.wandb_entity)
    wandb_config = wandb.config

    for k, v in wandb_config.items():
        config.__dict__[k] = v


    # Ensure dirs exist
    utils.prepare_dirs(config)

    # ensure reproducibility
    torch.manual_seed(config.random_seed)
    kwargs = {}
    if config.use_gpu:
        torch.cuda.manual_seed(config.random_seed)
        kwargs = {"num_workers": config.num_workers, "pin_memory": True}

    # instantiate data loaders
    if config.task == "mnist":
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
        trainer = MNISTTrainer(config, train_loader=train_dloader, test_loader=test_dloader, is_gridsearch=True)
        
    elif config.task == "svhn":
        # Load SVHN
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
        
    elif config.task == "multinumber":
        # Load synthetic MultiNumber dataset on SVHN
        train_loader, val_loader = None, None
        if config.is_train:
            train_loader = data_loaders.get_loader_multinumber(
                config.data_dir,
                "train",
                config.batch_size,
                config.end_class,
                config.separator_class,
                debug_run=config.debug_run,
                use_encoder=config.use_encoder,
                snapshot=config.snapshot,
                **kwargs,
            )
            val_loader = data_loaders.get_loader_multinumber(
                config.data_dir,
                "val",
                config.batch_size,
                config.end_class,
                config.separator_class,
                debug_run=config.debug_run,
                use_encoder=config.use_encoder,
                snapshot=config.snapshot,
                **kwargs,
            )
        test_dloader = data_loaders.get_loader_multinumber(
                config.data_dir,
                "test",
                config.batch_size,
                config.end_class,
                config.separator_class,
                debug_run=config.debug_run,
                use_encoder=config.use_encoder,
                snapshot=config.snapshot,
                **kwargs,
            )
            
        # Initialize Trainer for SVHN Dataset
        trainer = MultiNumberTrainer(config, train_loader=train_loader, val_loader=val_loader, test_loader=test_dloader)

    # Start training
    utils.save_config(config)
    trainer.train()
    trainer.test()


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)

    # WanDB login
    wandb.login()
    
    # Hyperparameter search config   
    sweep_config = {
        'name': 'Transformer GTrXL SVHN',
        'method': 'random',
        'metric': {
            'name': 'Test Reward',
            'goal': 'maximize'   
            },
        'parameters':  {
            'transformer_model': {
                'value': "gtrxl"},
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
    }    
    
    # Print gridsearch info
    pprint.pprint(sweep_config)
    
    # Start gridsearch
    sweep_id = wandb.sweep(sweep_config, project=PROJECT, entity=ENTITY)
    wandb.agent(sweep_id, function=main, count=35)