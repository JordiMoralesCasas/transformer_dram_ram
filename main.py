import torch

import trainers.utils as utils
import data.data_loaders as data_loaders

from trainers.trainer_mnist import MNISTTrainer
from trainers.trainer_svhn import SVHNTrainer
from trainers.trainer_multinumber import MultiNumberTrainer
from config import get_config


def main(config):
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
        trainer = MNISTTrainer(config, train_loader=train_dloader, test_loader=test_dloader)
        
    elif config.task == "svhn":
        # Load SVHN
        train_val_loader = None
        if config.is_train:
            train_val_loader = data_loaders.get_train_valid_loader_svhn(
                config.data_dir,
                config.batch_size,
                config.end_class,
                config.random_seed,
                debug_run=config.debug_run,
                do_preprocessing=config.preprocess,
                use_encoder=config.use_encoder,
                snapshot=config.snapshot,
                **kwargs,
            )
        test_dloader = data_loaders.get_test_loader_svhn(
            config.data_dir, 
            config.batch_size, 
            config.end_class,
            debug_run=config.debug_run, 
            do_preprocessing=config.preprocess, 
            use_encoder=config.use_encoder,
            snapshot=config.snapshot,
            **kwargs,
        )
            
        # Initialize Trainer for SVHN Dataset
        trainer = SVHNTrainer(config, train_val_loader=train_val_loader, test_loader=test_dloader)
        
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

    # either train
    if config.is_train:
        utils.save_config(config)
        trainer.train()
        trainer.test()
    # or load a pretrained model and test
    else:
        trainer.test()


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    config, unparsed = get_config()
    main(config)

