import numpy as np
from trainers.utils import plot_images
import random
import os
import pickle
import json

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

from data.svhn.svhn_dataset import SVHNDataset
from data.svhn.svhn_collator import DataCollatorForSVHN
from data.svhn.svhn_utils import DigitStructFile

from data.svhn.multinumber_collator import DataCollatorForMultiNumberSVHN
from data.svhn.multinumber_dataset import MultiNumberSVHNDataset


def get_train_valid_loader_mnist(
    data_dir,
    batch_size,
    random_seed,
    valid_size=0.1,
    shuffle=True,
    num_workers=4,
    pin_memory=False,
):
    """Train and validation data loaders for the MNIST Dataset.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args:
        data_dir (path): 
            Path directory to the dataset.
        batch_size (int): 
            How many samples per batch to load.
        random_seed (int): 
            Fix seed for reproducibility.
        valid_size (float): 
            Percentage split of the training set used for the validation set. 
            Should be a float in the range [0, 1]. In the paper, this number is
            set to 0.1.
        shuffle (bool):
            Whether to shuffle the train/validation indices.
        num_workers (int): 
            Number of subprocesses to use when loading the dataset.
        pin_memory (bool): 
            whether to copy tensors into CUDA pinned memory. Set it to True if
            using GPU.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert (valid_size >= 0) and (valid_size <= 1), error_msg

    # define transforms
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    trans = transforms.Compose([transforms.ToTensor(), normalize])

    # load dataset
    dataset = datasets.MNIST(data_dir, train=True, download=True, transform=trans)

    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    valid_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return (train_loader, valid_loader)


def get_test_loader_mnist(data_dir, batch_size, num_workers=4, pin_memory=False):
    """Test dataloader for the MNIST Dataset.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args:
        data_dir (str): 
            Path directory to the dataset.
        batch_size (int): 
            How many samples per batch to load.
        num_workers (int): 
            Number of subprocesses to use when loading the dataset.
        pin_memory (bool): 
            Whether to copy tensors into CUDA pinned memory. Set it to True if using GPU.
    """
    # define transforms
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    trans = transforms.Compose([transforms.ToTensor(), normalize])

    # load dataset
    dataset = datasets.MNIST(data_dir, train=False, download=True, transform=trans)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return data_loader


def get_train_valid_loader_svhn(
    data_dir,
    batch_size,
    end_class,
    random_seed,
    num_workers=4,
    pin_memory=False,
    debug_run=False,
    do_preprocessing=None,
    use_encoder=False,
    snapshot=False
):
    """Train and validation data loaders for the SVHN Dataset.

    Args:
        data_dir (str): 
            Path directory to the dataset.
        batch_size (int): 
            How many samples per batch to load.
        end_class (int): 
            Label corresponding to the "End" of prediction.
        random_seed (int): 
            Fix seed for reproducibility.
        num_workers (ing): 
            Number of subprocesses to use when loading the dataset.
        pin_memory (bool): 
            Whether to copy tensors into CUDA pinned memory. Set it to True if using GPU.
        debug_run (bool): 
            Wether we are running a DEBUG run. A smaller dataset will be used.
        do_preprocessing (bool): 
            Wether to apply the preprocessing method (crop around digits,
            resize and random/center crop).
        use_encoder (bool):
            Wether a Transformer encoder will be used for context.
        snapshot (bool):
            Wether we are in snapshot mode.
        
    """ 
    if do_preprocessing != "crop":   
        try:
            do_preprocessing = int(do_preprocessing)
        except:
            assert False, '"do_preprocessing" must be "crop" or an integer.'
    assert do_preprocessing == "crop" or type(do_preprocessing) == int, '"do_preprocessing" must be "crop" or an integer.'
        
    
    # Get Train and Validation data
    if not os.path.exists(data_dir + "train_data.pkl"):
        random.seed(random_seed)
        # Load .mat data
        train_set_data = DigitStructFile(data_dir + "train/digitStruct.mat").getAllDigitStructure_ByDigit()
        extra_set_data = DigitStructFile(data_dir + "extra/digitStruct.mat").getAllDigitStructure_ByDigit()

        # Join all data and randomly create the validation and train sets following a 10/90 split.
        all_data = train_set_data + extra_set_data
        shuffle_data = random.sample(all_data, len(all_data))
        val_length = int(0.1*len(all_data))
        val_data = shuffle_data[:val_length]
        train_data = shuffle_data[val_length:]

        # save the resulting set data
        with open(data_dir + "train_data.pkl", "wb") as f:
            pickle.dump(train_data, f)
        with open(data_dir + "val_data.pkl", "wb") as f:
            pickle.dump(val_data, f)
    else:
        with open(data_dir + "train_data.pkl", "rb") as f:
            train_data = pickle.load(f)
        with open(data_dir + "val_data.pkl", "rb") as f:
            val_data = pickle.load(f)

    # define transforms
    image_processor, trans = None, None
    if do_preprocessing == "crop":
        trans = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((64, 64)),
            transforms.RandomCrop((54, 54)),
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))])
        
    elif do_preprocessing == 224 and not snapshot:
        # For large images, we can use either an image Transformer encoder or
        # a ResNet feature extractor
        if use_encoder:
            image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        else:
            trans = transforms.Compose([
                transforms.Resize((do_preprocessing, do_preprocessing)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    else:
        trans = transforms.Compose([
            transforms.Resize((do_preprocessing, do_preprocessing)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
    
    # load dataset
    train_dataset = SVHNDataset(data_dir, train_data, end_class, transforms=trans, debug_run=debug_run, do_preprocessing=do_preprocessing, snapshot=snapshot)
    val_dataset = SVHNDataset(data_dir, val_data, end_class, transforms=trans, debug_run=debug_run, do_preprocessing=do_preprocessing, snapshot=snapshot)
    
    data_collator = DataCollatorForSVHN(image_processor=image_processor)
    train_loader = DataLoader(
        train_dataset, 
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    
    valid_loader = DataLoader(
        val_dataset, 
        shuffle=False,
        collate_fn=data_collator, 
        batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, )

    return (train_loader, valid_loader)


def get_test_loader_svhn(data_dir, batch_size, end_class, num_workers=4, pin_memory=False, debug_run=False, do_preprocessing=False, use_encoder=False, snapshot=False):
    """Test dataloader for the SVHN Dataset.

    Args:
        data_dir (str): 
            Path directory to the dataset.
        batch_size (int): 
            How many samples per batch to load.
        end_class (int): 
            Label corresponding to the "End" of prediction.
        num_workers (ing): 
            Number of subprocesses to use when loading the dataset.
        pin_memory (bool): 
            Whether to copy tensors into CUDA pinned memory. Set it to True if using GPU.
        debug_run (bool): 
            Wether we are running a DEBUG run. A smaller dataset will be used.
        do_preprocessing (bool): 
            Wether to apply the preprocessing method (crop around digits,
            resize and random/center crop).
        use_encoder (bool):
            Wether a Transformer encoder will be used for context.
        snapshot (bool):
            Wether we are in snapshot mode.
            
    """
    if do_preprocessing != "crop":   
        try:
            do_preprocessing = int(do_preprocessing)
        except:
            assert False, '"do_preprocessing" must be "crop" or an integer.'
    assert do_preprocessing == "crop" or type(do_preprocessing) == int, '"do_preprocessing" must be "crop" or an integer.'

    # Get Test data
    if not os.path.exists(data_dir + "test_data.pkl"):
        test_data = DigitStructFile(data_dir + "test/digitStruct.mat").getAllDigitStructure_ByDigit()
        with open(data_dir + "test_data.pkl", "wb") as f:
            pickle.dump(test_data, f)
    else:
        with open(data_dir + "test_data.pkl", "rb") as f:
            test_data = pickle.load(f)

    image_processor, trans = None, None
    if do_preprocessing == "crop":
        trans = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((64, 64)),
            transforms.RandomCrop((54, 54)),
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))])
        
    elif do_preprocessing == 224 and not snapshot:
        # For large images, we can use either an image Transformer encoder or
        # a ResNet feature extractor
        if use_encoder:
            image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        else:
            trans = transforms.Compose([
                transforms.Resize((do_preprocessing, do_preprocessing)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    else:
        trans = transforms.Compose([
            transforms.Resize((do_preprocessing, do_preprocessing)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
    
    # load dataset
    test_dataset = SVHNDataset(data_dir, test_data, end_class, trans, debug_run=debug_run, do_preprocessing=do_preprocessing, snapshot=snapshot)

    test_loader = DataLoader(
        test_dataset, shuffle=False,
        collate_fn=DataCollatorForSVHN(image_processor=image_processor),
        batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    return test_loader


def get_loader_multinumber(
    data_dir,
    split,
    batch_size,
    end_class,
    separator_class,
    num_workers=4,
    pin_memory=False,
    debug_run=False,
    use_encoder=False,
    snapshot=False
):
    """
    Train and validation data loaders for the synthetic multiple number 
    dataset.

    Args:   
        data_dir (str): 
            Path directory to the dataset.
        split (str):
            Current split name.
        batch_size (int): 
            How many samples per batch to load.
        end_class (int): 
            Label corresponding to the "End" of prediction.
        separator_class (int): 
            Label corresponding to the "Separator".
        num_workers (ing): 
            Number of subprocesses to use when loading the dataset.
        pin_memory (bool): 
            Whether to copy tensors into CUDA pinned memory. Set it to True if using GPU.
        debug_run (bool): 
            Wether we are running a DEBUG run. A smaller dataset will be used.
        use_encoder (bool):
            Wether a Transformer encoder will be used for context.
        snapshot (bool):
            Wether we are in snapshot mode.
        
    """
    # define transforms
    image_processor, trans = None, None        
    if not snapshot:
        # For large images, we can use either an ViT encoder or a ResNet feature extractor
        if use_encoder:
            image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        else:
            trans = transforms.Compose([
                transforms.Resize((224, 224,)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    else:
        trans = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((54, 54)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
    
    # load dataset
    dataset = MultiNumberSVHNDataset(data_dir, split, end_class, separator_class, transforms=trans, debug_run=debug_run)
    
    data_collator = DataCollatorForMultiNumberSVHN(image_processor=image_processor)
    loader = DataLoader(
        dataset, 
        shuffle=True if split == "train" else False,
        collate_fn=data_collator,
        batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    return loader