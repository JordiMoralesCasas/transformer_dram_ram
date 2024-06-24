import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn as nn
from anls import anls_score

from PIL import Image


def denormalize(T, coords):
    return 0.5 * ((coords + 1.0) * T)


def bounding_box(x, y, size, color="w"):
    x = int(x - (size / 2))
    y = int(y - (size / 2))
    rect = patches.Rectangle(
        (x, y), size, size, linewidth=1, edgecolor=color, fill=False
    )
    return rect


# https://github.com/pytorch/examples/blob/master/imagenet/main.py
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class NLSScorer:
    def __init__(self, tokenizer, threshold=0.5):
        self.tokenizer = tokenizer
        self.threshold = threshold

    def id_to_str(self, batch_ids):
        strings = [
            self.tokenizer.decode(ids, clean_up_tokenization_spaces=True, skip_special_tokens=True) 
            for ids in batch_ids
        ]
        return strings

    def compute_anls(self, pred, true):
        """
        Wrapper for the "anls score" function with a single answer.

        Args:
            pred (str): Predicted answer
            true (str): Ground truth answer

        Return: 
            NLS score
        """
        anls = anls_score(
            prediction=pred, 
            gold_labels=[true], 
            threshold=self.threshold
            )
        return anls
    
    def __call__(self, pred_ids, label_ids):
        pred_strs = self.id_to_str(pred_ids)
        label_strs = self.id_to_str(label_ids)
        nls_scores = [
            self.compute_anls(p, gt) for p, gt in zip(pred_strs, label_strs)
        ]
        
        return torch.tensor(nls_scores)


class AccScorer:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id
    
    def ensure_same_length(self, pred_ids, label_ids):
        # Ensure pred_ids and label_ids have the same length
        b, len_p = pred_ids.shape
        b, len_l = label_ids.shape

        if len_p > len_l:
            # If the prediction is longer, crop to the same length
            # as the labels
            pred_ids = pred_ids[:, :len_l]
        elif len_p < len_l:
            # If the prediction is shorter, add padding until the same
            # length as the labels
            pred_ids = torch.cat([
                pred_ids,
                torch.full((b, len_l - len_p), fill_value=self.pad_token_id, device=pred_ids.device)
            ], axis=1)
        return pred_ids, label_ids
    
    def __call__(self, pred_ids, label_ids):
        pred_ids, label_ids = self.ensure_same_length(pred_ids, label_ids)

        # Compute accuracy
        acc = (pred_ids.detach() == label_ids).float()
        pad_mask = label_ids != self.pad_token_id
        acc[~pad_mask] = 0
        acc = acc.sum(axis=1) / pad_mask.sum(axis=1)
        
        return acc
    

class AreaScorer: # TODO: Change name? 
    """
    For each glimpse, get the proportion of pixels that are not white
    """
    def __init__(self):
        pass
    
    def __call__(self, all_patches):
        # Conver patches to grayscale (avg across all channels)
        all_patches = all_patches.mean(dim=2)
        
        # get the proportion of pixels that are not white (defined by a threshold)
        thr = 0.05
        proportions = [(sample < 1 - thr).sum() / sample.numel() for sample in all_patches]
        return torch.tensor(proportions)


def resize_array(x, size):
    # 3D and 4D tensors allowed only
    assert x.ndim in [3, 4], "Only 3D and 4D Tensors allowed!"

    # 4D Tensor
    if x.ndim == 4:
        res = []
        for i in range(x.shape[0]):
            img = array2img(x[i])
            img = img.resize((size, size))
            img = np.asarray(img, dtype="float32")
            img = np.expand_dims(img, axis=0)
            img /= 255.0
            res.append(img)
        res = np.concatenate(res)
        res = np.expand_dims(res, axis=1)
        return res

    # 3D Tensor
    img = array2img(x)
    img = img.resize((size, size))
    res = np.asarray(img, dtype="float32")
    res = np.expand_dims(res, axis=0)
    res /= 255.0
    return res


def img2array(data_path, desired_size=None, expand=False, view=False):
    """
    Util function for loading RGB image into a numpy array.

    Returns array of shape (1, H, W, C).
    """
    img = Image.open(data_path)
    img = img.convert("RGB")
    if desired_size:
        img = img.resize((desired_size[1], desired_size[0]))
    if view:
        img.show()
    x = np.asarray(img, dtype="float32")
    if expand:
        x = np.expand_dims(x, axis=0)
    x /= 255.0
    return x


def array2img(x):
    """
    Util function for converting anumpy array to a PIL img.

    Returns PIL RGB img.
    """
    x = np.asarray(x)
    x = x + max(-np.min(x), 0)
    x_max = np.max(x)
    if x_max != 0:
        x /= x_max
    x *= 255
    return Image.fromarray(x.astype("uint8"), "RGB")


def plot_images(images, gd_truth):

    images = images.squeeze()
    assert len(images) == len(gd_truth) == 9

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    for i, ax in enumerate(axes.flat):
        # plot the image
        ax.imshow(images[i], cmap="Greys_r")

        xlabel = "{}".format(gd_truth[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def prepare_dirs(config):
    for path in [config.data_dir, config.ckpt_dir]:
        if not os.path.exists(path):
            os.makedirs(path)


def save_config(config):
    model_name = "ram_{}_{}x{}_{}".format(
        config.num_glimpses, config.patch_size, config.patch_size, config.glimpse_scale
    )
    filename = model_name + "_params.json"
    param_path = os.path.join(config.ckpt_dir, filename)

    print("[*] Model Checkpoint Dir: {}".format(config.ckpt_dir))
    print("[*] Param Path: {}".format(param_path))

    with open(param_path, "w") as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)


# Based on PositionalEncoding2D from https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/torch_encodings.py
class PositionalEncoding2D(nn.Module):
    def __init__(self, embedding_size):
        """
        :param embedding_size: Size of the embedding
        """
        super(PositionalEncoding2D, self).__init__()
        self.embedding_size = embedding_size
        
        # We required embedding_size to be multiple of 4
        assert embedding_size % 2 == 0
        self.num_channels = embedding_size // 2

        # Inverse frequence
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, self.num_channels, 2).float() / self.num_channels))

    def get_emb(self, pos):
        """
        Gets a base embedding for one dimension with sin and cos intertwined
        """
        sin_inp = torch.einsum("i,j->ij", pos, self.inv_freq.to(pos.device))
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return torch.flatten(emb, -2, -1)

    def forward(self, vis_embeddings, locs):
        """
        Apply positonal embeddings.

        Args:
            vis_embeddings: A 2d tensor of size (batch_size, hidden_size)
            locs: a 2d tensor of size (batch_size, 2). Contains the location
              of the glimpse (vis_embeddings), expressed as a cartesion 2D
              coordinate in the range [-1, 1].

        Returns:
            Input visual embeddings plus 2D positional embeddings
        """
        batch_size = vis_embeddings.shape[0]
        
        #Apply the positonal embeddings corresponding to the regions of each zoomed in image.
        vis_embeddings = vis_embeddings.view(-1, self.embedding_size)

        # Compute embeddings for each component
        emb_x = self.get_emb(locs[:, 0])
        emb_y = self.get_emb(locs[:, 1])

        # Join embeddings
        pos_emb = torch.zeros(
            (batch_size, self.embedding_size),
            device=vis_embeddings.device,
            dtype=vis_embeddings.dtype,
        )
        pos_emb[:, :self.num_channels] = emb_x
        pos_emb[:, self.num_channels:] = emb_y

        
        # Add positional embeddings to the input visual embeddings
        return vis_embeddings + pos_emb


# Based on "_shift_right()" from https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/models/t5/modeling_t5.py#L872
def shift_right(input_ids, decoder_start_token_id, pad_token_id):
    """# shift inputs to the right
    if is_torch_fx_proxy(input_ids):
        # Item assignment is not supported natively for proxies.
        shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
        shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
    else:"""
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = decoder_start_token_id

    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids