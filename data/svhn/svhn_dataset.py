import torch
import os
from PIL import Image
import numpy as np
import cv2

class SVHNDataset(torch.utils.data.Dataset):
    """
    Dataset object for the multiple digit recognition
    task on the SVHN dataset.
    """
    
    def __init__(self, root_folder: str, split_data: list[dict], end_class: int, transforms=None, debug_run=False, do_preprocessing=False, snapshot=False):
        """
        Initialize a Synthethic Dataset object
        Args:
            root_folder (str): 
                Path to the directory where SVHN images are.
            split_data (str): 
                Path to the pickle file containing data of the current split.
            end_class (int): 
                Label corresponding to the "End" of prediction.
            transforms (torchvision.transform): 
                Transforms to be applied to the images.
            debug_run (bool): 
                Wether we are running a DEBUG run. If True, the resulting dataset
                will have a size of 256. 
            do_preprocessing (bool): 
                Wether to do preprocessing step on the images. See 
                'preprocess_crop' method for more details.
            snapshot (bool): 
                Wether we are in Snapshot mode.
            
        """
        self.root_folder = root_folder
        self.split_data = split_data
        if debug_run:
            self.split_data = self.split_data[:256]
        self.transforms = transforms
        self.do_preprocessing = do_preprocessing
        self.snapshot = snapshot
        self.end_class = end_class

    def __len__(self):
        return len(self.split_data)
    
    def preprocess_crop(self, bboxes, img):
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
            
        Args:
            bboxes (List[List]): 
                List containing the bounding boxes (left, top, width and height) of all
                the digits of the current number.
            img (PIL Image): 
                Image object.
                
        Returns:
            Processed image
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
    
    def preprocess_expand(self, bboxes, img, scale=0.2, split=None):
        """
        Preprocessing step where the original image is extended with additional background.
        Numbers are tighly cropped and the randomly pasted in a background image.
        
        Args:
            bboxes (List[List]): 
                List containing the bounding boxes (left, top, width and height) of all
                the digits of the current number.
            img (PIL Image): 
                Image object.
            scale (float): 
                Standard deviation used to create the random noise background image.
            split (str):
                Current split
                
        Returns:
            Processed image
        """
        
        # convert to grayscale
        img = img.convert('L')
        
        # this proportion ensures that all samples' digits have the same size as in the preprocessing
        # method used in "preprocess_crop".
        proportion_img = 1.3*self.do_preprocessing/64
        
        # Get smaller bounding box that contain all digit bounding boxes
        x1 = int(min([i["left"] for i in bboxes]))
        y1 = int(min([i["top"] for i in bboxes]))
        x2 = int(max([i["left"] + i["width"] for i in bboxes]))
        y2 = int(max([i["top"] + i["height"] for i in bboxes]))
        
        # get matrix containing the bbox
        bbox_img = np.array(img)[y1:y2, x1:x2]
        h, w = bbox_img.shape
        
        # compute the mean background color and create an empty background image with random noise
        mean_background_color = np.median(
            np.concatenate([bbox_img[:, 0], bbox_img[:, -1], bbox_img[0, :], bbox_img[-1, :]]))
        new_img = (
            np.random.normal(
                loc=mean_background_color/255, scale=scale, 
                size=(int(h*proportion_img), int(w*proportion_img))
                )*255).astype("uint8")
        new_img = np.clip(new_img, a_min=0, a_max=255)
        
        # randomly paste the bbox on the background image or place in the center for inference
        if split == "train":
            rand_x, rand_y = np.random.randint(0, new_img.shape[1]-w), np.random.randint(0, new_img.shape[0]-h)
        else:
            rand_x, rand_y= int((new_img.shape[1]-w)/2), int((new_img.shape[0]-h)/2)   
        new_img[rand_y:rand_y+h, rand_x:rand_x+w] = bbox_img
        
        # to blur borders, we create a mask around the bbox and do inpainting
        mask = np.zeros(new_img.shape, dtype="uint8")
        border_size = 10
        mask[rand_y-border_size:rand_y+h+border_size, rand_x-border_size:rand_x+w+border_size] = 1
        mask[rand_y:rand_y+h, rand_x:rand_x+w] = 0
        new_img = cv2.inpaint(new_img, mask, 5, cv2.INPAINT_NS)
        
        # convert to PIL and resize image
        img = Image.fromarray(new_img)
        return img
        

    def __getitem__(self, index):
        """
        Get dataset example given an index.

        Args:
            index (int): index of the desired dataset example

        Returns:
            dict: Dataset example
        """
        current_sample = self.split_data[index]
        # Read image
        img_path = os.path.join(self.root_folder, current_sample["dataset_split"], current_sample["filename"])
        img = Image.open(img_path)

        if self.do_preprocessing == "crop":
            img = self.preprocess_crop(current_sample["boxes"], img)
        else:
            img = self.preprocess_expand(current_sample["boxes"], img, split=current_sample["dataset_split"])
            if self.do_preprocessing == 224 and not self.snapshot:
                # resnet will be used, so we need a 3-channel image
                img = img.convert('RGB')

        # Apply transforms (Resize + Normalize)
        if self.transforms is not None:
            img = self.transforms(img)

        # Get labels, we add an "end sequence" label (0) at the end
        labels = [int(box["label"]) for box in current_sample["boxes"]] + [self.end_class]

        # Max length is 5 (as in the paper)
        labels = labels[:5]

        return {"image": img, "labels": labels}