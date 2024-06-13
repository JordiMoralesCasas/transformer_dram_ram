import torch
import numpy as np
from PIL import Image
import os
import random
import json


class DocILEDataset(torch.utils.data.Dataset):
    """
    Pytorch Dataset for the Reading task on the DocILE's
    synthetic split dataset.
    
    The task is performed on preprocessed images, cropped around
    the Ground Truth answer bounding box, following the preprocessing
    techinique proposed for the SVHN dataset in [1].
    
    [1] Ian J. Goodfellow et al. "Multi-digit Number Recognition
              from Street View Imagery using Deep Convolutional Neural Networks"
    """
    def __init__(self, data_path: str, image_dir: str, transforms, debug_run: bool):
        """
        Initialize a DocILEDataset Dataset object
        Args:
            data_path (str): path to the directory that contains the json files
                for the training, validation and testing data.
            image_dir (str): path directory where the dataset images are stored.
            transforms (torchvision.transform): Transforms to be applied to the images.
            debug_run (bool): This is a debugging run. Use only a small sample of data.
        """
        # Load Dataset
        with open(data_path, "r") as f:
            self.data = json.load(f)
        
        # When doing a debug run, use only a few samples
        if debug_run:
            self.data = self.data[:64]
        
        # set transforms 
        self.transforms = transforms
        # set image directory
        self.image_dir = image_dir

    def __len__(self):
        return len(self.data)
    
    
    def preprocess(self, answer_bbox, img):
        """
            Follow the preprocessing mentioned in [1]: Expand 
            the answer bounding box by 30% in both the x and 
            the y direction and crop the image.
            
            Then, the images are resized and randomly cropped (
            or center cropped if doing inference) as defined by
            the dataset's transforms.
            

            [1] Ian J. Goodfellow et al. "Multi-digit Number Recognition
              from Street View Imagery using Deep Convolutional Neural Networks"
        """
        w, h = img.size
        
        # covert answer bbox to image coordinates
        x1_, y1_, x2_, y2_ = int(answer_bbox[0]*w), int(answer_bbox[1]*h), int(answer_bbox[2]*w), int(answer_bbox[3]*h)

        # Expand this bbox by 30% in both x and y axis
        x1 = int(max(0, x1_ - 0.15*(x2_-x1_)))
        y1 = int(max(0, y1_ - 0.15*(y2_-y1_)))
        x2 = int(min(w, x2_ + 0.15*(x2_-x1_)))
        y2 = int(min(h, y2_ + 0.15*(y2_-y1_)))

        # Crop image
        img = img.crop((x1, y1, x2, y2))

        return img

    def __getitem__(self, index: int):
        """
        Get dataset example given an index.

        Args:
            index (int): index of the desired dataset example

        Returns:
            Dataset example (dict)
        """
        current = self.data[index]
        
        # There is always a single question per sample
        answer = current["answers"][0]
        
        # Get answer bounding box
        bbox = current["answer_boxes"][0]

        # Load image
        img_path = os.path.join(self.image_dir, f"{current['image_name']}.jpg")
        img = Image.open(img_path)

        img = self.preprocess(bbox, img)
        
        # Apply transforms (Resize + Normalize)
        img = self.transforms(img)
        
        return {"img_path": img_path, "image": img, "answer": answer, "bbox": bbox}
    
    
if __name__ == "__main__":
    from torchvision import transforms
    
    # define transforms
    transf = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((128, 128)),
            transforms.CenterCrop((96, 96)),
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))])

    dataset = DocILEDataset(
        data_path="/data/users/jmorales/docile/train_set.json",
        image_dir="/data2/users/CLEF/docile/data/docile/images_dpi300",
        transforms=transf,
        debug_run=True
    )

    print(len(dataset))

    sample = dataset[0]
    print(sample.keys())
    print(sample["question"])
    print(sample["answer"])
    print(sample["bbox"])

    sample["image"]