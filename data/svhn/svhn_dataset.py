import torch
import os
from PIL import Image

class SVHNDataset(torch.utils.data.Dataset):
    """
    Dataset object for the "Synthetic box detection dataset".

    Attributes:
        image_folder (str):
            Path to image folder
        label_folder (str):
            Path to GT labels folder

    """
    def __init__(self, root_folder: str, split_data: list[dict], transforms, do_preprocessing=False):
        """
        Initialize a Synthethic Dataset object
        Args:
            root_folder (str):
                Path to image folder
            label_folder (str):
                Path to GT labels folder
        """
        self.root_folder = root_folder
        self.split_data = split_data
        self.transforms = transforms
        self.do_preprocessing = do_preprocessing

    def __len__(self):
        return len(self.split_data)
    
    def preprocess(self, bboxes, img):
        """
            Follow the preprocessing mentioned in [1]:

            "Find the small rectangular bounding box that will 
            contain individual character bounding boxes. We then
            expand this bounding box by 30% in both the x and 
            the y direction, crop the image to that bounding box
            and resize the crop to 64x64 pixels. We then crop a 
            54x54 pixel image from a random location within the 
            64x64 pixel image."

            [1] Ian J. Goodfellow et al. "Multi-digit Number Recognition
              from Street View Imagery using Deep Convolutional Neural Networks"
        """
        w, h = img.size

        # Get smaller bounding box that contain all digit bounding boxes
        x1 = min([i["left"] for i in bboxes])
        y1 = min([i["top"] for i in bboxes])
        x2 = max([i["left"] + i["width"] for i in bboxes])
        y2 = max([i["top"] + i["height"] for i in bboxes])

        # Expand this bbox by 30% in both x and y axis
        x1 = int(max(0, x1 - 0.15*(x2-x1)))
        y1 = int(max(0, y1 - 0.15*(y2-y1)))
        x2 = int(min(w, x2 + 0.15*(x2-x1)))
        y2 = int(min(h, y2 + 0.15*(y2-y1)))

        # Crop image
        img = img.crop((x1, y1, x2, y2))

        return img
        

    def __getitem__(self, index):
        """
        Get dataset example given an index.

        Args:
            index (int):
                index of the desired dataset example

        Returns:
            dict: Dataset example
        """
        current_sample = self.split_data[index]
        # Read image
        img_path = os.path.join(self.root_folder, current_sample["dataset_split"], current_sample["filename"])
        img = Image.open(img_path)

        if self.do_preprocessing:
            img = self.preprocess(current_sample["boxes"], img)

        # Apply transforms (Resize + Normalize)
        img = self.transforms(img)

        # Get labels, we add an "end sequence" label (0) at the end
        labels = [int(box["label"]) for box in current_sample["boxes"]] + [0]

        # Max length is 5 (as in the paper)
        labels = labels[:5]

        return {"image": img, "labels": labels}