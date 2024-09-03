from transformers import ViTImageProcessor, BatchEncoding
from typing import List
from dataclasses import dataclass


@dataclass
class DataCollatorForSVHN:
    """
    Data collator class for the multiple digit recognition
    task on the SVHN dataset.

    returns:
        mini-batch sample.
    """
    image_processor: ViTImageProcessor = None
    
    def __call__(self, batch: List[dict]) -> BatchEncoding:
        """
        Data collator call method. Processes the batches coming from the dataset to be model inputs.

        Args:
            batch (List[dict]):
                Batch containing dataset examples.
        Returns:
            BatchEncoding: Containins the processed batch. Each features is a Pytorch tensors.
        """
    
        data = {
            "pixel_values": [
                sample["image"] if self.image_processor else sample["image"].tolist() for sample in batch],
            "labels": []}
            
        if self.image_processor is not None:
            # Open image and process them (ViT)   
            pixel_values = self.image_processor(data["pixel_values"], return_tensors="pt").pixel_values
            data["pixel_values"] = pixel_values

        # Create labels and pad to have maximum length of the batch
        max_length = max([len(sample["labels"]) for sample in batch])
        for i, sample in enumerate(batch):
            number_of_labels = len(sample["labels"])
            remainder = (max_length - number_of_labels)
            data["labels"].append(sample["labels"] + [-100]*(remainder))
            
        # Convert to torch tensors
        batch = BatchEncoding(data, tensor_type="pt")

        return batch