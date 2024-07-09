import torch
import os
from PIL import Image
import numpy as np
import cv2
import json

class MultiNumberSVHNDataset(torch.utils.data.Dataset):
    """
    Dataset object for the multiple digit recognition
    task on the SVHN dataset.
    """
    
    def __init__(self, root_folder: str, split: str, end_class: int, separator_class: int, transforms=None, debug_run=False, do_preprocessing=False, snapshot=False):
        """
        Initialize a Synthethic Dataset object
        Args:
            root_folder (str): path to the directory where SVHN
                images are.
            split_data (str): path to the pickle file containing
                data of the current split.
            transforms (torchvision.transform): Transforms to be 
                applied to the images.
            do_preprocessing (bool): Wether to do preprocessing step
                on the images. See 'preprocess_crop' method for more details.
            
        """
        self.root_folder = root_folder
        self.image_folder = os.path.join(root_folder, split)
        
        # Get split data
        assert os.path.exists(os.path.join(self.root_folder, f"{split}_split.json")), "Split data file does not exist!"
        with open(os.path.join(self.root_folder, f"{split}_split.json"), "r") as f:
            self.split_data = json.load(f)
        
        if debug_run:
            self.split_data = self.split_data[:256]
        self.transforms = transforms
        self.do_preprocessing = do_preprocessing
        self.snapshot = snapshot
        self.end_class = end_class
        self.separator_class = separator_class

    def __len__(self):
        return len(self.split_data)
    

    def __getitem__(self, index):
        """
        Get dataset example given an index.

        Args:
            index (int): index of the desired dataset example

        Returns:
            dict: Dataset example
        """
        current_sample = self.split_data[index]
        
        # Sample ID (for the test split results)
        sample_id = current_sample["id"]
        
        # Read image
        img_path = os.path.join(self.image_folder, f"{sample_id}.png")
        img = Image.open(img_path)

        # Apply transforms
        if self.transforms is not None:
            img = self.transforms(img)

        # Join labels together. Add a special label between the first and second numbers, which serves both as 
        # a separator and for the model to understand when it has finished predicting the first number and has to go
        # find the second one. An "end" label is added at the end.
        labels = [int(i) for i in current_sample["first"]] + [self.separator_class] + [int(i) for i in current_sample["second"]] + [self.end_class]

        return {"image": img, "labels": labels, "id": sample_id}