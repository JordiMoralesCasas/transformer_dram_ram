import torch
from torchvision.transforms.v2 import (
    GaussianBlur,
    RandomChoice,
    Identity
)
from transformers import ViTImageProcessor, BatchEncoding
from collections import defaultdict
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class DataCollatorForSVHN:
    """
    Data collator class for the multiple digit recognition
    task on the SVHN dataset.

    returns:
        mini-batch sample.
    """
    
    def __call__(self, batch: List[dict]) -> BatchEncoding:
        """
        Data collator call method. Processes the batches coming from the dataset to be model inputs.

        Args:
            batch (List[dict]):
                Batch containing dataset examples.
        Returns:
            BatchEncoding: Containins the processed batch. Each features is a Pytorch tensors.
        """

        results = defaultdict(list)
        for sample in batch:
            results['pixel_values'].append(sample["image"].tolist())

        # Create labels and pad to have maximum length of the batch
        max_length = max([len(sample["labels"]) for sample in batch])
        for i, sample in enumerate(batch):
            number_of_labels = len(sample["labels"])
            remainder = (max_length - number_of_labels)
            results["labels"].append(sample["labels"] + [-100]*(remainder))
            
        # Convert to torch tensors
        batch = BatchEncoding(results, tensor_type="pt")

        return batch