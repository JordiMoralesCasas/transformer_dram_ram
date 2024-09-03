import torch.nn as nn
import torch

import modelling.modules as modules
from torchvision.models import resnet18, resnet50

"""
 Based on https://github.com/kevinzakka/recurrent-visual-attention/blob/master/model.py
"""
class RecurrentAttention(nn.Module):
    """A Recurrent Model of Visual Attention (RAM) [1].

    RAM is a recurrent neural network that processes
    inputs sequentially, attending to different locations
    within the image one at a time, and incrementally
    combining information from these fixations to build
    up a dynamic internal representation of the image.
    
    We introduce a modified version using Transformer as 
    the core instead a RNN.

    References:
      [1]: Minh et. al., https://arxiv.org/abs/1406.6247
    """

    def __init__(
        self, g, k, s, c, h_g, h_l, std, hidden_size, num_classes, core_type, transformer_model
    ):
        """Constructor.

        Args:
          g (int): size of the square patches in the glimpses extracted by the retina.
          k (int): number of patches to extract per glimpse.
          s (int): scaling factor that controls the size of successive patches.
          c (int): number of channels in each image.
          h_g (int): hidden layer size of the fc layer for `phi`.
          h_l (int): hidden layer size of the fc layer for `l`.
          std (float): standard deviation of the Gaussian policy.
          hidden_size (int): hidden size of the rnn.
          num_classes (int): number of classes in the dataset.
          core_type (str): Type of core network to use (RNN or Transformer).
          transformer_model (str): Which transformer core to use (gpt2, trxl, gtrxl or DRAMLM)
        """
        super().__init__()

        self.std = std

        self.sensor = modules.GlimpseNetwork(h_g, h_l, g, k, s, c)
        self.hidden_size = hidden_size
        self.core_type = core_type
        if core_type == "rnn":
            self.core = modules.CoreNetworkRNN(hidden_size, hidden_size)
        else:
            self.core = modules.CoreNetworkTransformer(hidden_size, hidden_size, transformer_model=transformer_model)
            
        self.locator = modules.LocationNetwork(hidden_size, 2, std)
        self.classifier = modules.ActionNetwork(hidden_size, num_classes)
        self.baseliner = modules.BaselineNetwork(hidden_size, 1)

    def reset_glimpse_buffer(self, batch_size, device):
        """
            Initialize buffer where past glimpses will be stored.
        """
        self.core.past_glimpses = torch.empty((batch_size, 0, self.hidden_size), device=device)

    def forward(self, x, l_t_prev, h_t_prev, last=False):
        """Run RAM for one timestep on a minibatch of images.

        Args:
            x: a 4D Tensor of shape (B, H, W, C). The minibatch
                of images.
            l_t_prev: a 2D tensor of shape (B, 2). The location vector
                containing the glimpse coordinates [x, y] for the previous
                timestep `t-1`.
            h_t_prev: a 2D tensor of shape (B, hidden_size). The hidden
                state vector for the previous timestep `t-1`.
            last: a bool indicating whether this is the last timestep.
                If True, the action network returns an output probability
                vector over the classes and the baseline `b_t` for the
                current timestep `t`. Else, the core network returns the
                hidden state vector for the next timestep `t+1` and the
                location vector for the next timestep `t+1`.

        Returns:
            h_t: a 2D tensor of shape (B, hidden_size). The hidden
                state vector for the current timestep `t`.
            mu: a 2D tensor of shape (B, 2). The mean that parametrizes
                the Gaussian policy.
            l_t: a 2D tensor of shape (B, 2). The location vector
                containing the glimpse coordinates [x, y] for the
                current timestep `t`.
            b_t: a vector of length (B,). The baseline for the
                current time step `t`.
            log_probas: a 2D tensor of shape (B, num_classes). The
                output log probability vector over the classes.
            log_pi: a vector of length (B,).
        """
        g_t = self.sensor(x, l_t_prev)
        h_t = self.core(g_t, h_t_prev)

        log_pi, l_t = self.locator(h_t)
        b_t = self.baseliner(h_t).squeeze()

        if last:
            log_probas = self.classifier(h_t)
            return h_t, l_t, b_t, log_probas, log_pi

        return h_t, l_t, b_t, log_pi


class DeepRecurrentAttention(nn.Module):
    """
    DRAM is a recurrent neural network building on the ideas
    from RAM, that processes inputs sequentially, attending to
    different locations within the image one at a time, and 
    incrementally combining information from these fixations
    to build up a dynamic internal representation of the image.
    Differently from RAM, it is composed of two separate layers,
    one which can actually look at the whole image to provide
    interesting locations to look at, while the other layer
    makes digit predictions based solely on the extracted glimpses.
    
    We introduce a modified version using Transformer as the core
    instead of LSTMs.

    """

    def __init__(
        self, g, k, s, c, std, hidden_size, cell_size, inner_size, n_heads, 
        num_classes, core_type, device, transformer_model, use_encoder, image_size,
        snapshot, resnet=18
    ):
        """Constructor for the DRAM model.

        Args:
          g (int): size of the square patches in the glimpses extracted by the retina.
          k (int): number of patches to extract per glimpse.
          s (int): scaling factor that controls the size of successive patches.
          c (int): number of channels in each image.
          std (float): standard deviation of the Gaussian policy.
          hidden_size (int): Size of the hidden vectors.
          cell_size (int): Size of the LSTM cells.
          inner_size (int): Size of the inner projections of the Transformers.
          n_heads (int): Number of attention heads
          num_classes (int): number of classes in the dataset.
          core_type (str): Type of core network to use (RNN or Transformer).
          device (str): Current device
          transformer_model (str): Which transformer core to use (gpt2, trxl or gtrxl)
          use_encoder (bool): Wether a Transformer encoder is used to provide context.
          image_size (int): Size of the input images
          snapshot (bool): Wether we are in Snapshot mode.
          resnet (int): ResNet version to use (When applicable)
        """
        super().__init__()

        self.std = std
        self.device = device
        self.use_encoder = use_encoder
        self.hidden_size = hidden_size
        self.cell_size = cell_size
        
        image_size = 54 if snapshot else image_size

        assert image_size in [54, 110, 186, 224]
        if image_size == 224:
            if self.use_encoder and core_type != "rnn":
                #The encoder is part of the core network (transformer encoder)
                # Learnable query for starting the generation
                self.query_vector = nn.Embedding(1, self.hidden_size)
            else: 
                # Use resnet50 as our context network. Freeze all layers except the last,
                # which is replaced by a FC layer to project features to hidden size
                self.context = resnet18(pretrained=True) if resnet == 18 else resnet50(pretrained=True)
                for param in self.context.parameters():
                    param.requires_grad = False
                output_size = 512 if resnet == 18 else 2048
                self.context.fc = nn.Linear(output_size, hidden_size)
        else:
            strd = 2 if image_size > 54 and image_size < 224 else 1
            self.context = modules.ContextNetwork(hidden_size, stride=strd, img_size=image_size, snapshot=snapshot)
            
        self.sensor = modules.GlimpseNetworkDRAM(g, k, s, c, hidden_size, core_type=core_type)
        
        self.core_type = core_type
        if core_type == "rnn":
            self.core = modules.CoreNetworkLSTM(hidden_size, hidden_size)
        else:
            self.core = modules.CoreNetworkDoubleTransformer(hidden_size, hidden_size, inner_size, n_heads, transformer_model, use_encoder)
            
        self.locator = modules.LocationNetwork(hidden_size, 2, std)
        if num_classes:
            self.classifier = modules.ActionNetwork(hidden_size, num_classes)
        self.baseliner = modules.BaselineNetwork(hidden_size, 1)

    def reset_transformers_buffer(self, batch_size):
        """
            Reset buffers for past glimpses (transformer 1 inputs) and
            past states (transformer 2 inputs).
        """
        self.core.past_glimpses = torch.empty(
            (batch_size, 0, self.hidden_size), device=self.device)
        self.core.past_states = torch.empty(
            (batch_size, 0, self.hidden_size), device=self.device)

    def forward(self, x, l_t_prev, out_prev, first=False, last=False):
        """Run DRAM for one timestep on a minibatch of images.

        Args:
            x: a 4D Tensor of shape (B, H, W, C). The minibatch
                of images.
            l_t_prev: a 2D tensor of shape (B, 2). The location vector
                containing the glimpse coordinates [x, y] for the previous
                timestep `t-1`.
            out_prev: tuple containing the past step outputs. When using the 
                LSTM as core network, this tuple contain two tensor, 
                (2, B, hidden_size) and (2, B, cell_size), the hidden and cell
                states for the two layers of the network. When using the Transformer
                core, this tuple contain two tensors of size (B, hidden_size), the
                last hidden states of each Transformer, although they are not used
                because they are also stored in the past states/glimpses buffer.
            first: a bool indicating whether this is the first timestep.
                If True, the network has to create the context vector for the input
                image and use it to generate the location of the first glimpse. Also,
                store it in the second transformer's buffer.
            last: a bool indicating whether this is the last timestep.
                If True, the action network returns an output probability
                vector over the classes and the baseline `b_t` for the
                current timestep `t`. Else, the core network returns the
                hidden state vector for the next timestep `t+1` and the
                location vector for the next timestep `t+1`.

        Returns:
        
            h_t: a 2D tensor of shape (B, hidden_size). The hidden
                state vector for the current timestep `t`.
            h_t_1/h_t_2: a 2D tensor of shape (B, hidden_size). The hidden
                state vector for the current timestep `t` for the classification (1)
                or location (2) Transformers.
            c_t: a 2D tensor of shape (B, hidden_size). The LSTM cell
                state vector for the current timestep `t`.
            l_t: a 2D tensor of shape (B, 2). The location vector
                containing the glimpse coordinates [x, y] for the
                current timestep `t`.
            b_t: a vector of length (B,). The baseline for the
                current time step `t`.
            log_probas: a 2D tensor of shape (B, num_classes). The
                output log probability vector over the classes.
            log_pi: a vector of length (B,).
        """
        # LSTM as the core network
        if self.core_type == "rnn":
            if first:
                # In the first step we have to create the context image vector,
                # which will serve both as the first hidden state for the second layer
                # of the LSTM as well as input for the locator network to predict
                # where to look first.
                # The initial hidden state for the first layer of the RNNs is a
                # vector of zeros.
                h_t_1 = torch.zeros(
                    x.shape[0],
                    self.hidden_size,
                    dtype=torch.float,
                    device=self.device,
                    requires_grad=True,
                )
                h_t_2 = self.context(x)
                h_t = torch.stack([h_t_1, h_t_2], dim=0)

                # Cell states for the LSTM layers are also initialized as zeros
                c_t = torch.zeros(
                    (2, x.shape[0], self.hidden_size),
                    dtype=torch.float,
                    device=self.device,
                    requires_grad=True,
                )
            else:
                g_t = self.sensor(x, l_t_prev)
                (h_t, c_t) = self.core(g_t, out_prev)

            log_pi, l_t = self.locator(h_t[1, :, :])
            b_t = self.baseliner(h_t[1, :, :]).squeeze()

            if last:
                log_probas = self.classifier(h_t[0, :, :])
                return (h_t, c_t), l_t, b_t, log_probas, log_pi

            return (h_t, c_t), l_t, b_t, log_pi
        
        # Transformer as the core network
        else:
            if first:
                # Empty the glimpse and state buffers
                self.reset_transformers_buffer(batch_size=x.shape[0])
                
                
                if self.use_encoder:
                    # call encoder
                    self.core.compute_encoder_hidden_states(pixel_values=x)
                    query_embed = self.query_vector.weight.unsqueeze(0).repeat(x.shape[0], 1, 1).squeeze(1)
                    (h_t_1, h_t_2) = self.core(query_embed)
                else:
                    # Create the context vector and pass it through the 
                    # second transformer. The resulting hidden state
                    # will be used to obtain the location for the first
                    # glimpse.
                    context = self.context(x)
                    h_t_2 = self.core.process_context(context)
                    h_t_1 = None
                
            else:
                g_t = self.sensor(x, l_t_prev)
                (h_t_1, h_t_2) = self.core(g_t)

            log_pi, l_t = self.locator(h_t_2)
            b_t = self.baseliner(h_t_2).squeeze()

            if last:
                log_probas = self.classifier(h_t_1)
                return (h_t_1, h_t_2), l_t, b_t, log_probas, log_pi

            return (h_t_1, h_t_2), l_t, b_t, log_pi