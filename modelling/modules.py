import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal
from transformers import GPT2Model, AutoConfig, TransfoXLModel, AutoModel
from trainers.utils import PositionalEncoding2D
from modelling.gtrxl import GTrXL_wrapper as GTrXL


class Retina:
    """A visual retina.

    Extracts a foveated glimpse `phi` around location `l`
    from an image `x`.

    Concretely, encodes the region around `l` at a
    high-resolution but uses a progressively lower
    resolution for pixels further from `l`, resulting
    in a compressed representation of the original
    image `x`.
    """

    def __init__(self, g, k, s):
        """
        Contructor.
        
        Args:
        g (int): size of the first square patch.
        k (int): number of patches to extract in the glimpse.
        s (int): scaling factor that controls the size of
            successive patches.
        """
        self.g = g
        self.k = k
        self.s = s

    def foveate(self, x, l, flatten=True):
        """Extract `k` square patches of size `g`, centered
        at location `l`. The initial patch is a square of
        size `g`, and each subsequent patch is a square
        whose side is `s` times the size of the previous
        patch.

        The `k` patches are finally resized to (g, g) and
        concatenated into a tensor of shape (B, k, g, g, C).
        
        Args:
            x (torch.Tensor): a 4D Tensor of shape (B, H, W, C). The minibatch
                of images.
            l (torch.Tensor): a 2D Tensor of shape (B, 2). Contains normalized
                coordinates in the range [-1, 1].
        Returns:
            phi: a 5D tensor of shape (B, k, g, g, C). The
                foveated glimpse of the image.
        """
        phi = []
        size = self.g

        # extract k patches of increasing size
        for i in range(self.k):
            phi.append(self.extract_patch(x, l, size))
            size = int(self.s * size)

        # resize the patches to squares of size g
        for i in range(1, len(phi)):
            k = phi[i].shape[-1] // self.g
            phi[i] = F.avg_pool2d(phi[i], k)

        # concatenate into a single tensor and flatten
        phi = torch.cat(phi, 1)
        if flatten:
            phi = phi.view(phi.shape[0], -1)

        return phi

    def extract_patch(self, x, l, size):
        """Extract a single patch for each image in `x`.

        Args:
            x (torch.Tensor): a 4D Tensor of shape (B, H, W, C). The minibatch
                of images.
            l (torch.tensor): a 2D Tensor of shape (B, 2).
            size (int): a scalar defining the size of the extracted patch.

        Returns:
            patch: a 4D Tensor of shape (B, size, size, C)
        """
        B, C, H, W = x.shape

        start = self.denormalize(H, l)
        end = start + size

        # pad with zeros
        x = F.pad(x, (size // 2, size // 2, size // 2, size // 2))

        # loop through mini-batch and extract patches
        patch = []
        for i in range(B):
            patch.append(x[i, :, start[i, 1] : end[i, 1], start[i, 0] : end[i, 0]])
        return torch.stack(patch)

    def denormalize(self, T, coords):
        """Convert coordinates in the range [-1, 1] to
        coordinates in the range [0, T] where `T` is
        the size of the image.
        """
        return (0.5 * ((coords + 1.0) * T)).long()

    def exceeds(self, from_x, to_x, from_y, to_y, T):
        """Check whether the extracted patch will exceed
        the boundaries of the image of size `T`.
        """
        if (from_x < 0) or (from_y < 0) or (to_x > T) or (to_y > T):
            return True
        return False


class GlimpseNetworkDRAM(nn.Module):
    """The glimpse network.

    Uses the location predicted in the previous step to extract
    a glimpse around this position. The resulting glimpse is feed
    through a convlutional network to obtain a glimpse vector. 
    Positional information is then added following a sinusoidal positional
    encoding.
    """

    def __init__(self, g, k, s, c, hidden_size, kernel_sizes=(5, 3, 3), core_type="rnn"):
        """
        Contructor.
        
        Args:
            g (int): size of the square patches in the glimpses extracted
                by the retina.
            k (int): number of patches to extract per glimpse.
            s (int): scaling factor that controls the size of successive patches.
            c (int): number of channels in each image.
            hidden_size (int): Hidden size of the internal states.
            kernel_sizes (List[int]): Kernel sizes of the three convolutional layers
        """
        super().__init__()

        self.retina = Retina(g, k, s)

        assert g > 8, "Patch size must be greater than 8, and ideally even"

        # glimpse layer
        self.conv1 = nn.Conv2d(k, 64, kernel_size=kernel_sizes[0])
        self.conv2 = nn.Conv2d(64, 64, kernel_size=kernel_sizes[1])
        self.conv3 = nn.Conv2d(64, 128, kernel_size=kernel_sizes[2])
        self.fc1 = nn.Linear(128*(g-8)*(g-8), hidden_size)

        # location layer
        self.core_type = core_type
        if self.core_type == "rnn":
            # If we have an RNN core, use project location to obtain a "where"
            # vector and multiply it (point-wise) with the "what" vector
            self.fc2 = nn.Linear(2, hidden_size)
        else:
            # If we have a transformer core, use a sine positional encoding,
            # which is added to the "what" vector
            self.pos_encoding = PositionalEncoding2D(hidden_size)

    def forward(self, x, l_t_prev):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): a 4D Tensor of shape (B, H, W, C). The minibatch
                of images.
            l_t_prev (torch.Tensor): a 2D tensor of shape (B, 2). Contains the glimpse
                coordinates [x, y] for the previous timestep `t-1`.

        Returns:
            g_t: a 2D tensor of shape (B, hidden_size).
                The glimpse representation returned by
                the glimpse network for the current
                timestep `t`.
        """
        
        # generate glimpse phi from image x
        phi = self.retina.foveate(x, l_t_prev, flatten=False)

        # extract features -> "what" vector
        phi = self.conv1(phi)
        phi = self.conv2(phi)
        phi = self.conv3(phi)
        phi_out = self.fc1(phi.flatten(start_dim=1))

        if self.core_type == "rnn":
            # "where" vector
            l_out = F.relu(self.fc2(l_t_prev.flatten(start_dim=1)))
            g_t = torch.mul(phi_out, l_out)
        else:
            # use a 2D sine positional embeddings
            g_t = self.pos_encoding(phi_out, l_t_prev)

        return g_t
    
class GlimpseNetwork(nn.Module):
    """The glimpse network.

    Combines the "what" and the "where" into a glimpse
    feature vector `g_t`.

    - "what": glimpse extracted from the retina.
    - "where": location tuple where glimpse was extracted.

    Concretely, feeds the output of the retina `phi` to
    a fc layer and the glimpse location vector `l_t_prev`
    to a fc layer. Finally, these outputs are fed each
    through a fc layer and their sum is rectified.

    In other words:

        `g_t = relu( fc( fc(l) ) + fc( fc(phi) ) )`

    Args:
        h_g: hidden layer size of the fc layer for `phi`.
        h_l: hidden layer size of the fc layer for `l`.
        g: size of the square patches in the glimpses extracted
        by the retina.
        k: number of patches to extract per glimpse.
        s: scaling factor that controls the size of successive patches.
        c: number of channels in each image.
        x: a 4D Tensor of shape (B, H, W, C). The minibatch
            of images.
        l_t_prev: a 2D tensor of shape (B, 2). Contains the glimpse
            coordinates [x, y] for the previous timestep `t-1`.

    Returns:
        g_t: a 2D tensor of shape (B, hidden_size).
            The glimpse representation returned by
            the glimpse network for the current
            timestep `t`.
    """

    def __init__(self, h_g, h_l, g, k, s, c):
        super().__init__()

        self.retina = Retina(g, k, s)

        # glimpse layer
        D_in = k * g * g * c
        self.fc1 = nn.Linear(D_in, h_g)

        # location layer
        D_in = 2
        self.fc2 = nn.Linear(D_in, h_l)

        self.fc3 = nn.Linear(h_g, h_g + h_l)
        self.fc4 = nn.Linear(h_l, h_g + h_l)

    def forward(self, x, l_t_prev):
        # generate glimpse phi from image x
        phi = self.retina.foveate(x, l_t_prev)

        # flatten location vector
        l_t_prev = l_t_prev.view(l_t_prev.size(0), -1)

        # feed phi and l to respective fc layers
        phi_out = F.relu(self.fc1(phi))
        l_out = F.relu(self.fc2(l_t_prev))

        what = self.fc3(phi_out)
        where = self.fc4(l_out)

        # feed to fc layer
        g_t = F.relu(what + where)

        return g_t


class CoreNetworkRNN(nn.Module):
    """Core RNN network.

    An RNN that maintains an internal state by integrating
    information extracted from the history of past observations.
    It encodes the agent's knowledge of the environment through
    a state vector `h_t` that gets updated at every time step `t`.

    Concretely, it takes the glimpse representation `g_t` as input,
    and combines it with its internal state `h_t_prev` at the previous
    time step, to produce the new internal state `h_t` at the current
    time step.

    In other words:

        `h_t = relu( fc(h_t_prev) + fc(g_t) )`
    """

    def __init__(self, input_size, hidden_size):
        """
        Contructor.
        
        Args:
            input_size (int): input size of the rnn.
            hidden_size (int): hidden size of the rnn.
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def forward(self, g_t, h_t_prev):
        """
        Forward step of the core network.
        
        Args:
            g_t (torch.Tensor): a 2D tensor of shape (B, hidden_size). The glimpse
                representation returned by the glimpse network for the
                current timestep `t`.
            h_t_prev (torch.Tensor): a 2D tensor of shape (B, hidden_size). The
                hidden state vector for the previous timestep `t-1`.

        Returns:
            h_t: a 2D tensor of shape (B, hidden_size). The hidden
                state vector for the current timestep `t`.
        """
        h1 = self.i2h(g_t)
        h2 = self.h2h(h_t_prev)
        h_t = F.relu(h1 + h2)
        return h_t
    

class CoreNetworkTransformer(nn.Module):
    """Core Transformer network.

    Extends the idea of the core RNN network to the transformer architecture.
    The agent's knowledge is not encoded in a single state `h_t` that is 
    sequentially updated, but instead we keep the history of all past glimpses
    and states, and the agent can attend to them at any time.

    It takes the glimpse representation `g_t` as input, and self-attention is
    performed between all past glimpses to produce a new contextualized state
    `h_t`.
    """

    def __init__(self, input_size, hidden_size, transformer_model="gpt2"):
        """
        Constructor.
        
        Args:
            input_size (int): input size of the rnn.
            hidden_size (int): hidden size of the rnn.
            transformer_model (str): Which transformer core to use (gpt2, trxl or gtrxl)

        Returns:
            h_t: a 2D tensor of shape (B, hidden_size). The hidden
                state vector for the current timestep `t`.
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize transformer
        if transformer_model == "gpt2":
            config = AutoConfig.from_pretrained("gpt2")
            config.hidden_size = self.hidden_size
            config.vocab_size = 0
            config.max_position_embeddings = 6
            config.num_hidden_layers = 1
            config.n_head = 1
            config.n_inner = 64
            self.transformer = GPT2Model(config)
        elif transformer_model == "trxl":
            config = AutoConfig.from_pretrained("Transfo-xl-wt103")
            config.d_model = self.hidden_size
            config.d_embed  = self.hidden_size
            config.n_head = 1
            config.d_head = 32
            config.d_inner = self.hidden_size
            config.n_layer = 2
            config.mem_len = 6
            config.vocab_size = 0
            config.cutoffs = [0, 0] # Related to vocab length.. We do not use this
            self.transformer = TransfoXLModel(config)
        elif transformer_model == "gtrxl":
            config = {
                "input_dim": self.hidden_size,
                "head_dim": self.hidden_size,
                "embedding_dim": self.hidden_size,
                "memory_len": 0, # We already implement our own memory
                "head_num": 1,
                "mlp_num": 1,
                "layer_num": 1
                }
            self.transformer = GTrXL(**config)

        # We will store the past glimpses in this variable
        self.past_glimpses = None
    
    def forward(self, g_t, _):
        """
        Forward step of the Transformer core network.
        
        Args:
            g_t (torch.Tensor): a 2D tensor of shape (B, hidden_size). The glimpse
                representation returned by the glimpse network for the
                current timestep `t`.

        Returns:
            h_t: a 2D tensor of shape (B, hidden_size). The hidden
                state vector for the current timestep `t`.
        """
        # We need to keep track of all the transformer's inputs.
        # Using the generated past_key_values does not seem to work well
        # when backpropagating the loss (inplace operations)
        self.past_glimpses = torch.cat([
            self.past_glimpses, 
            g_t[:, None, :]
            ], axis=1)
        
        # Compute next hidden state
        output = self.transformer(inputs_embeds=self.past_glimpses
        ) 

        h_t = output.last_hidden_state[:, -1, :]
        return h_t 


class CoreNetworkLSTM(nn.Module):
    """Core LSTM network.

    Core network for the DRAM model. A two-layers LSTM where the first one
    encodes the knowledge about past glimpses, which is then used for predicting
    the digits; and the second one that takes the knowledge from the 
    context vector (whole input image) and the hidden states from the first layer
    to produce hidden states used for choosing the next glimpse locations and
    the baselines (expected total Reward).
    """

    def __init__(self, input_size, hidden_size):
        """
        Constructor.
        
        Args:
            input_size (int): input size of the LSTM.
            hidden_size (int): hidden size of the LSTM.        
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)

    def forward(self, g_t, out_prev):
        """
        Forward step of the LSTM core network.
        
        Args:
            g_t (torch.Tensor): a 2D tensor of shape (B, hidden_size). The glimpse
                representation returned by the glimpse network for the
                current timestep `t`.
            out_prev (torch.Tensor): a 2D tensor of shape (B, hidden_size). The
                hidden state vector for the previous timestep `t-1`.

        Returns:
            h_t: a 2D tensor of shape (B, hidden_size). The hidden
                state vector for the current timestep `t`.
        
        """
        _, (h_t, c_t) = self.lstm(g_t[:, None, :], out_prev)
        return (h_t, c_t)
    
class CoreNetworkDoubleTransformer(nn.Module):
    """Core Double Transformer network.

    Expands on the idea of having a LSTM with two layer to separate
    the context information from the digit predictions, but using 
    two single-layer transformers, the second one taking the output 
    hidden states of the first as inputs.

    The context vector is fed to the second Transformer (location 
    Transformer) at the beginning of the generation to include this
    information. It is also possible to use a pretrained Transformer
    encoder to provide the context.

    Args:
        input_size: input size of the rnn.
        hidden_size: hidden size of the rnn.
        g_t: a 2D tensor of shape (B, hidden_size). The glimpse
            representation returned by the glimpse network for the
            current timestep `t`.

    Returns:
        h_t: a 2D tensor of shape (B, hidden_size). The hidden
            state vector for the current timestep `t`.
    """

    def __init__(self, input_size, hidden_size, inner_size, n_heads, transformer_model="gpt2", use_encoder=False):
        """
        Constructor.
        Args:
            input_size (int): input size of the transformers.
            hidden_size (int): hidden size of the transformers.
            inner_size (int): inner size of the transformers (inner projections).
            n_heads (int): Number of attention heads
            transformer_model (str): Which transformer core to use (gpt2, trxl or gtrxl)
          use_encoder (bool): Wether a Transformer encoder is used to provide context.

        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize transformer
        if transformer_model == "gpt2":
            config = AutoConfig.from_pretrained("gpt2")
            config.hidden_size = self.hidden_size
            config.vocab_size = 0
            config.max_position_embeddings = 48 #TODO: This depends on the number of glimpses
            config.num_hidden_layers = 1
            config.n_head = n_heads
            config.n_inner = inner_size
            self.tr1 = GPT2Model(config)
            self.tr2 = GPT2Model(config)
        elif transformer_model == "trxl":
            config = AutoConfig.from_pretrained("Transfo-xl-wt103")
            config.d_model = self.hidden_size
            config.d_embed  = self.hidden_size
            config.n_head = n_heads
            config.d_inner = inner_size
            config.n_layer = 1
            config.mem_len = 48
            config.vocab_size = 0
            config.cutoffs = [0, 0] # Related to vocab length.. We do not use this
            self.tr1 = TransfoXLModel(config)
            self.tr2 = TransfoXLModel(config)
        elif transformer_model == "gtrxl":
            config = {
                "input_dim": self.hidden_size,
                "head_dim": self.hidden_size,
                "embedding_dim": self.hidden_size,
                "memory_len": 0, # We already implement our own memory
                "head_num": n_heads,
                "mlp_num": 1,
                "layer_num": 1
                }
            self.tr1 = GTrXL(**config)
            self.tr2 = GTrXL(**config, do_crossattention=use_encoder)

        self.encoder_hidden_states = None
        if use_encoder:
            # Instantitate image encoder
            self.image_encoder = AutoModel.from_pretrained("google/vit-base-patch16-224")
            
            # Freeze encoder
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        
        # We will store the past glimpses in this variable
        self.past_states = None
        self.past_glimpses = None
        
    def compute_encoder_hidden_states(self, pixel_values):
        """
        Takes the pixel values of the snapshot image and
        computes the encoder hidden states in order for the top decoder
        to do cross-attention.

        Args:
            pixel_values (torch.tensor): tensor containing a batch of image
                tensors.
        """
        outputs_top = self.image_encoder(pixel_values=pixel_values)
        self.encoder_hidden_states = outputs_top.last_hidden_state
    
    def process_context(self, context):
        """
        Given the context image, a context vector is produced and then 
        fed through the location transformer to provide the first
        glimpse location.
        """
        # In the first step, the second transformer has to produce
        # a hidden state for the context vector, which will be used for
        # producing the location for the first glimpse
        self.past_states = torch.cat([
            self.past_states, 
            context[:, None, :]
            ], axis=1)
        
        output = self.tr2(inputs_embeds=self.past_states)
        h_t_2 = output.last_hidden_state[:, -1, :]
        return h_t_2

    def forward(self, g_t):
        """
        Forward step.
        Args:
            g_t: a 2D tensor of shape (B, hidden_size). The glimpse
                representation returned by the glimpse network for the
                current timestep `t`.

        Returns:
            h_t: a 2D tensor of shape (B, hidden_size). The hidden
                state vector for the current timestep `t`.
        """
        # We need to keep track of all the transformer's inputs.
        # Using the generated past_key_values does not seem to work well
        # when backpropagating the loss (inplace operations)
        
        self.past_glimpses = torch.cat([
            self.past_glimpses, 
            g_t[:, None, :]
            ], axis=1)
        
        # Compute next hidden state
        output = self.tr1(inputs_embeds=self.past_glimpses)
        h_t_1 = output.last_hidden_state[:, -1, :]

        self.past_states = torch.cat([
            self.past_states, 
            h_t_1[:, None, :]
            ], axis=1)
            
        output = self.tr2(
            inputs_embeds=self.past_states,
            encoder_hidden_states=self.encoder_hidden_states)
        """# for TrXL
        output = self.tr2(
            inputs_embeds=self.past_states)"""
        h_t_2 = output.last_hidden_state[:, -1, :]

        return (h_t_1, h_t_2)         

class ActionNetwork(nn.Module):
    """The action network.

    Uses the internal state `h_t` of the core network to
    produce the final output classification.

    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a softmax to create a vector of
    output probabilities over the possible classes.

    Hence, the environment action `a_t` is drawn from a
    distribution conditioned on an affine transformation
    of the hidden state vector `h_t`, or in other words,
    the action network is simply a linear softmax classifier.
    """

    def __init__(self, input_size, output_size):
        """
        Constructor.
        
        Args:
            input_size (int): input size of the fc layer.
            output_size (int): output size of the fc layer.
        """
        super().__init__()

        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        """
        Forward step.
        
        Args:
            h_t: the hidden state vector of the core network
                for the current time step `t`.

        Returns:
            a_t: output probability vector over the classes.
        """
        a_t = F.log_softmax(self.fc(h_t), dim=1)
        return a_t


class LocationNetwork(nn.Module):
    """The location network.

    Uses the internal state `h_t` of the core network to
    produce the location coordinates `l_t` for the next
    time step.

    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a tanh to clamp the output beween
    [-1, 1]. This produces a 2D vector of means used to
    parametrize a two-component Gaussian with a fixed
    variance from which the location coordinates `l_t`
    for the next time step are sampled.

    Hence, the location `l_t` is chosen stochastically
    from a distribution conditioned on an affine
    transformation of the hidden state vector `h_t`.
    """

    def __init__(self, input_size, output_size, std):
        """
        Constructor.
        
        Args:
            input_size (int): input size of the fc layer.
            output_size (int): output size of the fc layer.
            std (float): standard deviation of the normal distribution.
        """
        super().__init__()

        self.std = std

        hiden_size = input_size // 2
        self.fc = nn.Linear(input_size, hiden_size)
        self.fc_lt = nn.Linear(hiden_size, output_size)

    def forward(self, h_t):
        """
        Forward step.
        
        Args:
            h_t: the hidden state vector of the core network for
                the current time step `t`.

        Returns:
            log_pi: a vector of length (B,).
            l_t: a 2D vector of shape (B, 2).
        """
        # compute mean
        feat = F.relu(self.fc(h_t.detach()))
        mu = torch.tanh(self.fc_lt(feat))

        # reparametrization trick
        l_t = Normal(mu, self.std).rsample()
        l_t = l_t.detach()
        log_pi = Normal(mu, self.std).log_prob(l_t)

        # we assume both dimensions are independent
        # 1. pdf of the joint is the product of the pdfs
        # 2. log of the product is the sum of the logs
        log_pi = torch.sum(log_pi, dim=1)

        # bound between [-1, 1]
        l_t = torch.clamp(l_t, -1, 1)

        return log_pi, l_t


class BaselineNetwork(nn.Module):
    """The baseline network.

    This network regresses the baseline in the
    reward function to reduce the variance of
    the gradient update.
    """

    def __init__(self, input_size, output_size):
        """
        Constructor.
        
        Args:
            input_size (int): input size of the fc layer.
            output_size (int): output size of the fc layer.
        """
        super().__init__()

        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        """
        Forward step.
        
        Args:
            h_t: the hidden state vector of the core network
                for the current time step `t`.

        Returns:
            b_t: a 2D vector of shape (B, 1). The baseline
                for the current time step `t`.
        """
        b_t = self.fc(h_t.detach())
        return b_t


class ContextNetwork(nn.Module):
    """The context network.

    Provides a coarse representation of the input image

    When using the LSTM core network, it will be used as the initial hidden
    state for the seconda layer of the network. When working with an (decoder)
    Transformer, it will be the first input of the second transformer.
    """
    
    def __init__(self, hidden_size, stride, kernel_sizes=(5,3,3), img_size=54, snapshot=False):
        """
        Constructor.
        
        Args:
            hidden_size: output size of the coarse vector.
            stride (int): Convolution's stride.
            kernel_sizes (List[int]): Kernel sizes of the three convolutional layers
            img_size (int): Size of the context images
            snapshot (bool): Wether we are in snapshot mode.
        """
        super().__init__()
        
        # wether we will be using snapshots of the original image as context
        self.snapshot = snapshot
        
        channels = (64, 64, 128)

        self.conv1 = nn.Conv2d(1, channels[0], stride=stride, kernel_size=kernel_sizes[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=kernel_sizes[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=kernel_sizes[2])
        self.pool = nn.MaxPool2d((2,2))
        
        if img_size == 54:
            final_feat_map_size = 4
        elif img_size == 64:
            final_feat_map_size = 4
        elif img_size == 110:
            final_feat_map_size = 5
        elif img_size == 186:
            final_feat_map_size = 9
        
        
        self.fc = nn.Linear(channels[-1]*final_feat_map_size*final_feat_map_size, hidden_size)

    def forward(self, x):
        """
        Forward step.
        
        Args:
            x: a 4D tensor of shape (B, 1, width, height). The grayscale
                input image.
        
        Return:
            i_c: a 2D vector of shape (B, hidden_size). Coarse representation of 
                the input image.
        """
        if self.snapshot:
            x = F.interpolate(x, size=(54, 54))
            
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        i_c = self.fc(x.flatten(start_dim=1))
        return i_c
