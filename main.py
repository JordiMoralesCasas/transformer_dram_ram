import torch

import trainers.utils as utils
import data.data_loaders as data_loaders

from trainers.trainer_mnist import MNISTTrainer
from trainers.trainer_svhn import SVHNTrainer
from trainers.trainer_docile_reading import DocILETrainerReading
from trainers.trainer_docile_find_text import DocILETrainerFindText
from config import get_config

from transformers import AutoTokenizer


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
            config.data_dir, config.batch_size, do_preprocessing=config.preprocess, **kwargs,
        )
        # Initialize Trainer for SVHN Dataset
        trainer = SVHNTrainer(config, train_loader=train_dloader, test_loader=test_dloader)

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

