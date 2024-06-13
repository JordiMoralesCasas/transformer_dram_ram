import torch
from transformers import ViTImageProcessor, BatchEncoding, T5TokenizerFast
from collections import defaultdict
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class DataCollator:
    """
    Data collator class for the Reading task on the DocILE's
    synthetic split dataset.

    Args:
        tokenizer (T5TokenizerFast): Tokenzier object
        return_img_path (bool): Wether to return the image paths.
        device (str): Current device

    Return:
        batch (torch.BatchEncoding): Batch containing dataset data.
            Data is paded across all the batch.
            keys: pixel_values, label_ids, decoder_attention_mask

        original_img_patches (list): list of length B of strings.
            Contains the input images' paths.
    """
    tokenizer: T5TokenizerFast
    return_img_path: bool = False
    device: str = None
    
    def __call__(self, batch: List[dict]) -> BatchEncoding:
        data = defaultdict(list)
        for sample in batch:
            for k, v in sample.items():
                if k == "image":
                    data["pixel_values"].append(v.tolist())
                else: 
                    data[k].append(v)

        # Tokenize answers + get attention mask
        output = self.tokenizer(data["answer"], return_attention_mask=True, return_tensors="pt", padding=True)
        label_ids = output.input_ids
        decoder_attention_mask = output.attention_mask
        data["label_ids"] = label_ids
        data["decoder_attention_mask"] = decoder_attention_mask

        # Drop useless features 
        data.pop("answer")
        original_img_paths = data.pop("img_path")

        # Create pytorch Batch Encoding
        batch = BatchEncoding(data, tensor_type="pt")

        if self.return_img_path:
            return (batch, original_img_paths)
        
        return batch