import torch

import trainers.utils as utils
import data.data_loaders as data_loaders

from trainers.trainer_mnist import MNISTTrainer
from trainers.trainer_svhn import SVHNTrainer
from trainers.trainer_docile import DocILETrainer
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
    elif config.dataset == "svhn":
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
    else:
        # Load DocILE
        # Initialize model/processor/tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # set pad token for GPT2, doesn't really matter what to use
        # since it will be masked by the attention mask anyways
        tokenizer.pad_token = tokenizer.eos_token
        
        config.vocab_size = tokenizer.vocab_size
        config.pad_token_id = tokenizer.pad_token_id
        config.eos_token_id = tokenizer.eos_token_id
        # for GPT2
        config.start_token_id = tokenizer.bos_token_id
        
        """# for T5, the start token is also the padding token
        config.start_token_id = tokenizer.pad_token_id"""
        
        # get device
        if config.use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # instantiate data loaders
        config.data_dir = "/data/users/jmorales/docile"
        config.image_folder = "/data2/users/CLEF/docile/data/docile/images_dpi300"
        train_loader, val_loader = None, None
        if config.is_train:
            train_loader = data_loaders.get_docile_loader(
                data_dir=config.data_dir,
                image_dir=config.image_folder,
                batch_size=config.batch_size,
                split="train",
                shuffle=True,
                device=device,
                tokenizer=tokenizer,
                show_sample=config.show_sample,
                debug_run=config.debug_run,
                **kwargs,
            )
            val_loader = data_loaders.get_docile_loader(
                data_dir=config.data_dir,
                image_dir=config.image_folder,
                batch_size=config.batch_size,
                split="val",
                shuffle=False,
                device=device,
                tokenizer=tokenizer,
                show_sample=config.show_sample,
                debug_run=config.debug_run,
                **kwargs,
            )
            
        """for i in train_loader:
            print(i.keys())
            import cv2
            import numpy as np
            
            img = ((i["pixel_values"][0] + 1) / 2 * 255).numpy()
            cv2.imwrite("test_docile.png", np.moveaxis(img, 0, 2))
            exit(0)"""
            
        test_loader = data_loaders.get_docile_loader(
                data_dir=config.data_dir,
                image_dir=config.image_folder,
                batch_size=config.batch_size,
                split="test",
                shuffle=False,
                device=device,
                tokenizer=tokenizer,
                show_sample=config.show_sample,
                debug_run=config.debug_run,
                **kwargs,
            )
        
        # Initialize Trainer for SVHN Dataset
        trainer = DocILETrainer(config, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, device=device, tokenizer=tokenizer)

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

