import numpy as np
from trainers.utils import plot_images
import random
import os
import pickle

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

from data.svhn.svhn_dataset import SVHNDataset
from data.svhn.svhn_collator import DataCollatorForSVHN
from data.svhn.svhn_utils import DigitStructFile


def get_train_valid_loader_mnist(
    data_dir,
    batch_size,
    random_seed,
    valid_size=0.1,
    shuffle=True,
    show_sample=False,
    num_workers=4,
    pin_memory=False,
):
    """Train and validation data loaders for the MNIST Dataset.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args:
        data_dir (path): path directory to the dataset.
        batch_size (int): how many samples per batch to load.
        random_seed (int): fix seed for reproducibility.
        valid_size (float): percentage split of the training set used for
            the validation set. Should be a float in the range [0, 1].
            In the paper, this number is set to 0.1.
        shuffle (bool): whether to shuffle the train/validation indices.
        show_sample (bool): plot 9x9 sample grid of the dataset.
        num_workers (int): number of subprocesses to use when loading the dataset.
        pin_memory (bool): whether to copy tensors into CUDA pinned memory. Set it to
            True if using GPU.
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

    # visualize some images
    if show_sample:
        sample_loader = DataLoader(
            dataset,
            batch_size=9,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy()
        X = np.transpose(X, [0, 2, 3, 1])
        plot_images(X, labels)

    return (train_loader, valid_loader)


def get_test_loader_mnist(data_dir, batch_size, num_workers=4, pin_memory=False):
    """Test dataloader for the MNIST Dataset.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args:
        data_dir: path directory to the dataset.
        batch_size: how many samples per batch to load.
        num_workers: number of subprocesses to use when loading the dataset.
        pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
            True if using GPU.
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
    random_seed,
    show_sample=False,
    num_workers=4,
    pin_memory=False,
    do_preprocessing=False
):
    """Train and validation data loaders for the SVHN Dataset.

    Args:
        data_dir (str): path directory to the dataset.
        batch_size (int): how many samples per batch to load.
        random_seed (int): fix seed for reproducibility.
        show_sample (bool): plot 9x9 sample grid of the dataset.
        num_workers (ing): number of subprocesses to use when loading 
            the dataset.
        pin_memory (bool): hether to copy tensors into CUDA pinned 
            memory. Set it to True if using GPU.
        do_preprocessing (bool): Wether to apply the preprocessing
            method (crop around digits, resize and random/center crop).
        
    """
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
    if do_preprocessing:
        train_trans = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((64, 64)),
            transforms.RandomCrop((54, 54)),
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))])
        val_trans = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((64, 64)),
            transforms.CenterCrop((54, 54)),
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))])
    else:
        train_trans = val_trans = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))])
    
    # load dataset
    train_dataset = SVHNDataset(data_dir, train_data, train_trans, do_preprocessing=do_preprocessing)
    val_dataset = SVHNDataset(data_dir, val_data, val_trans, do_preprocessing=do_preprocessing)

    valid_loader = DataLoader(
        val_dataset, shuffle=False,
        collate_fn=DataCollatorForSVHN(), 
        num_workers=num_workers, batch_size=batch_size)
    
    train_loader = DataLoader(
        train_dataset, shuffle=True,
        collate_fn=DataCollatorForSVHN(),
        batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    # visualize some images
    if show_sample:
        sample_loader = DataLoader(
            train_dataset,
            batch_size=9,
            shuffle=True,
            collate_fn=DataCollatorForSVHN(),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        data_iter = iter(sample_loader)
        batch = next(data_iter)
        X = batch.pixel_values.numpy()
        X = np.transpose(X, [0, 2, 3, 1])
        plot_images(X, batch.labels)

    return (train_loader, valid_loader)


def get_test_loader_svhn(data_dir, batch_size, num_workers=4, pin_memory=False, do_preprocessing=False):
    """Test dataloader for the SVHN Dataset.

    Args:
        data_dir (str): path directory to the dataset.
        batch_size (int): how many samples per batch to load.
        num_workers (int): number of subprocesses to use when loading the dataset.
        pin_memory (bool): whether to copy tensors into CUDA pinned memory. Set it to
            True if using GPU.
        do_preprocessing (bool): Wether to apply the preprocessing
            method (crop around digits, resize and random/center crop).
    """

    # Get Test data
    if not os.path.exists(data_dir + "test_data.pkl"):
        test_data = DigitStructFile(data_dir + "test/digitStruct.mat").getAllDigitStructure_ByDigit()
        with open(data_dir + "test_data.pkl", "wb") as f:
            pickle.dump(test_data, f)
    else:
        with open(data_dir + "test_data.pkl", "rb") as f:
            test_data = pickle.load(f)

    # define transforms
    if do_preprocessing:
        trans = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((64, 64)),
            transforms.CenterCrop((54, 54)),
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))])
    else:
        trans = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))])
    
    # load dataset
    test_dataset = SVHNDataset(data_dir, test_data, trans, do_preprocessing=do_preprocessing)

    train_loader = DataLoader(
        test_dataset, shuffle=True,
        collate_fn=DataCollatorForSVHN(),
        batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader
