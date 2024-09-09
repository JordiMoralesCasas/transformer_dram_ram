import os
import pickle
from PIL import Image
import numpy as np
import cv2
import random
from tqdm import tqdm
import json
from multiprocessing import Pool
import argparse

from skimage.transform import resize
from shapely.geometry import box, Polygon

def create_sample(input):
        """
        Create a dataset sample.
        
        """
        id, numbers, img_folder, img_size, bbox_size = input
        current_sample = {
            "id": id,
            "numbers": []
            }
        
        # Initialize backgroud image
        new_img = np.zeros((img_size, img_size), dtype="uint8")

        # to blur borders, we create a mask around the bbox to do inpainting
        mask = np.ones(new_img.shape, dtype="uint8")
        
        # we need to keep track of where the numbers are placed to decide its order
        numbers_positions = []

        # Polygon object to keep track of numbers and avoid overlaping bboxes
        locs_union = Polygon()

        for number in numbers:            
            # open image
            img_path = os.path.join(data_dir, number["dataset_split"], number["filename"])
            img = Image.open(img_path).convert('L')
            
            # get digit bboxes
            bboxes = number["boxes"]
            
            current_sample["numbers"].append([digit["label"] for digit in bboxes])

            # Get smaller bounding box that contain all digit bounding boxes
            x1 = int(min([i["left"] for i in bboxes]))
            y1 = int(min([i["top"] for i in bboxes]))
            x2 = int(max([i["left"] + i["width"] for i in bboxes]))
            y2 = int(max([i["top"] + i["height"] for i in bboxes]))
            
            # get matrix containing the bbox
            bbox_img = np.array(img)[y1:y2, x1:x2]
            bbox_img = (resize(bbox_img, (bbox_size, bbox_size)) * 256).astype("uint8")
            
            # randomly paste the bbox on the background image.
            border = 5
            counter = 10
            while True:
                # candidate position
                rand_x, rand_y = np.random.randint(0, img_size-bbox_size), np.random.randint(0, img_size-bbox_size)
                
                # test if the current position doesnt intersect with previous numbers
                poly_bbox = box(rand_x-border, rand_y-border, rand_x+bbox_size+border, rand_y+bbox_size+border)
                if poly_bbox.intersection(locs_union).area == 0:
                    # if there is no intersection, the position is valid
                    locs_union = locs_union.union(poly_bbox)
                    break
                
                # Avoid loop getting stuck. If no valid positon can be found, reduce the border.
                # If even after removing the border no valid position is found, return error message
                counter -= 1
                if counter == 0:
                    if border != 0:
                        border, counter = 0, 30
                    else:
                        assert False, "No valid position can be found"
                
            new_img[rand_y:rand_y+bbox_size, rand_x:rand_x+bbox_size] = bbox_img
            
            # save bbox position for the current number
            numbers_positions.append(((rand_x, rand_y), (rand_x+bbox_size, rand_y+bbox_size)))
            
            # update inpainting mask
            mask[rand_y:rand_y+bbox_size, rand_x:rand_x+bbox_size] = 0
            
        new_img = cv2.inpaint(new_img, mask, 32, cv2.INPAINT_TELEA)

        # convert to PIL and save
        img = Image.fromarray(new_img[:, :, None].repeat(3, 2))
        img.save(os.path.join(img_folder, f"{current_sample['id']}.png"))
        
        # Decide label order
        # if both numbers have similar height...
        p1, p2 = numbers_positions[0], numbers_positions[1]
        if abs(p1[0][1] - p2[0][1]) < bbox_size:
            # the one more to the left is the first
            if p1[0][0] < p2[0][0]:
                current_sample["first"] = current_sample["numbers"][0]
                current_sample["second"] = current_sample["numbers"][1]
            else:
                current_sample["second"] = current_sample["numbers"][0]
                current_sample["first"] = current_sample["numbers"][1]
        else:
            if p1[0][1] < p2[0][1]:
                current_sample["first"] = current_sample["numbers"][0]
                current_sample["second"] = current_sample["numbers"][1]
            else:
                current_sample["second"] = current_sample["numbers"][0]
                current_sample["first"] = current_sample["numbers"][1]
        
        current_sample.pop("numbers")
        return current_sample
        
# Create dataset
def create_split(raw_data, split, save_dir, num_workers=8, img_size=224, bbox_size=64):
    """
    Create a split of the dataset.
    
    Args:
        raw_data (dict): 
            JSON file containing the raw data for the split.
        split (str):
            Which split to create.
        save_dir (str):
            Directory where the split data will be saved.
        num_workers (int):
            Size of the pool of workers to use when creating the data.
        img_size (int):
            Size of the new dataset images
        bbox_size (int):
            Size to which the cropped numbers will be reshaped.
    
    """
    assert split in ["train", "val", "test"], "Split must be 'train', 'val' or 'test'"
    
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
    img_folder = os.path.join(save_dir, split)
    if not os.path.exists(img_folder):
            os.makedirs(img_folder)
    
    # Start pool of workers and create dataset data
    pool_input = [(i, data, img_folder, img_size, bbox_size) for i, data in enumerate(raw_data)]
    with Pool(num_workers) as pool:
      split_data = list(tqdm(pool.imap(create_sample, pool_input), total=len(pool_input), desc=f"Creating {split} split: "))
    
    # save split data
    with open(os.path.join(save_dir, f"{split}_split.json"), "w") as f:
        json.dump(split_data, f)


if __name__ == "__main__":

    """
        Create the synthetic Multiple Number Dataset
    """
    
    # Parse arguments
    arg_lists = []
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default=None, help="Directory with the original SVHN dataset."
    )
    parser.add_argument(
        "--save_dir", type=str, default=None, help="Directory where the new dataset will be saved."
    )
    parser.add_argument(
        "--img_size", type=int, default=224, help="Size of the dataset samples (Square)."
    )
    parser.add_argument(
        "--bbox_size", type=int, default=64, help="Size of the resized number's bounding box (Square)."
    )
    parser.add_argument(
        "--train_split_size", type=float, default=0.8, help="Portion of the whole dataset that is used for training. The remainder is split evenly for validation and test"
    )
    parser.add_argument(
        "--dataset_length", type=int, default=500000, help="Length of the created dataset."
    )
    parser.add_argument(
        "--num_workers", type=int, default=32, help="Size of the pool of workers."
    )
    args = parser.parse_args()


    # Load splits data
    with open(args.data_dir + "train_data.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open(args.data_dir + "val_data.pkl", "rb") as f:
        val_data = pickle.load(f)
    with open(args.data_dir + "test_data.pkl", "rb") as f:
        test_data = pickle.load(f)
    
    print("Filter data by length...")
    all_data = train_data + val_data + test_data
    print("Size before filtering:", len(all_data))

    # filter data with numberes with up to N digits
    N = 2
    all_data = [i for i in all_data if len(i["boxes"]) <= N]
    print("Size after filtering:", len(all_data))

    dataset = []
    for i in range(args.dataset_length):
        # Get a random number from the dataset
        sample1 = random.sample(all_data, 1)[0]
        sample2 = random.sample(all_data, 1)[0]
        dataset.append((sample1, sample2))
        
    ### Create splits
    # Ensure reproducibility
    random.seed(0)

    total_length = len(dataset)
    print("Total dataset length:", total_length)

    train_length = int(0.8*total_length)
    val_length = int(0.1*total_length)

    train_data = dataset[:train_length]
    val_data = dataset[train_length:train_length+val_length]
    test_data = dataset[train_length+val_length:]

    print("Training set length:", len(train_data))
    print("Val set length:", len(val_data))
    print("Test set length:", len(test_data))

    # Create splits
    create_split(train_data, split="train", save_dir=args.save_dir, img_size=args.img_size, bbox_size=args.bbox_size, num_workers=args.num_workers)
    create_split(test_data, split="test", save_dir=args.save_dir, img_size=args.img_size, bbox_size=args.bbox_size, num_workers=args.num_workers)
    create_split(val_data, split="val", save_dir=args.save_dir, img_size=args.img_size, bbox_size=args.bbox_size, num_workers=args.num_workers)