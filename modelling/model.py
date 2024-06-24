import torch.nn as nn
import torch

import modelling.modules as modules

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

    References:
      [1]: Minh et. al., https://arxiv.org/abs/1406.6247
    """

    def __init__(
        self, g, k, s, c, h_g, h_l, std, hidden_size, num_classes, core_type, transformer_model
    ):
        """Constructor.

        Args:
          g: size of the square patches in the glimpses extracted by the retina.
          k: number of patches to extract per glimpse.
          s: scaling factor that controls the size of successive patches.
          c: number of channels in each image.
          h_g: hidden layer size of the fc layer for `phi`.
          h_l: hidden layer size of the fc layer for `l`.
          std: standard deviation of the Gaussian policy.
          hidden_size: hidden size of the rnn.
          num_classes: number of classes in the dataset.
          num_glimpses: number of glimpses to take per image,
            i.e. number of BPTT steps.
          core_type: Type of core network to use (RNN or Transformer).
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
    """A Recurrent Model of Visual Attention (RAM) [1].

    RAM is a recurrent neural network that processes
    inputs sequentially, attending to different locations
    within the image one at a time, and incrementally
    combining information from these fixations to build
    up a dynamic internal representation of the image.

    References:
      [1]: Minh et. al., https://arxiv.org/abs/1406.6247
    """

    def __init__(
        self, g, k, s, c, h_g, h_l, std, hidden_size, cell_size, inner_size, n_heads, num_classes, core_type, device, transformer_model, max_length
    ):
        """Constructor.

        Args:
          g: size of the square patches in the glimpses extracted by the retina.
          k: number of patches to extract per glimpse.
          s: scaling factor that controls the size of successive patches.
          c: number of channels in each image.
          h_g: hidden layer size of the fc layer for `phi`.
          h_l: hidden layer size of the fc layer for `l`.
          std: standard deviation of the Gaussian policy.
          hidden_size: hidden size of the rnn.
          num_classes: number of classes in the dataset.
          num_glimpses: number of glimpses to take per image,
            i.e. number of BPTT steps.
          core_type: Type of core network to use (RNN or Transformer).
          device: Current device
          transformer_model: Which transformer core to use (gpt2, trxl, gtrxl or DRAMLM)
        """
        super().__init__()

        self.std = std
        self.device = device
        self.context = modules.ContextNetwork(hidden_size, stride=2 if transformer_model == "DRAMLM" else 1)
        self.sensor = modules.GlimpseNetworkDRAM(g, k, s, c, hidden_size, core_type=core_type)
        self.hidden_size = hidden_size
        self.cell_size = cell_size
        
        # Max length of generated answer (used only for the DRAM LM model)
        self.max_length = max_length 
        self.core_type = core_type
        if core_type == "rnn":
            self.core = modules.CoreNetworkLSTM(hidden_size, hidden_size)
        else:
            self.core = modules.CoreNetworkDoubleTransformer(hidden_size, hidden_size, inner_size, n_heads, transformer_model)
            
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
        
    def lm_answer(self, label_ids, decoder_attention_mask, pad_token_id, decoder_start_token_id):
        lm_output = self.core.lm_answer(
                label_ids=label_ids,
                decoder_attention_mask=decoder_attention_mask,
                pad_token_id=pad_token_id,
                decoder_start_token_id=decoder_start_token_id)
        lm_log_probas = self.core.lm_head(lm_output)
        return lm_log_probas
    
    def infer_answer(self, batch_size: int, start_token_id, eos_token_id, pad_token_id):
        """
        Autoregressively generate a sequence of tokens (answer token ids)

        Args:
            batch_size: size of the mini batches.

        Returns:
            2d tensor (batch_size, seq_len) containing the ids
                for the generated answers.

        """
        predicted_tokens = []
        predicted_logits = []
        running_sequences = torch.ones((batch_size, ), device=self.device, dtype=torch.bool)
        
        # start generation token
        next_token = torch.full((batch_size, ), fill_value=start_token_id, device=self.device)
        for i in range(self.max_length):
            # get output for current token
            output = self.core.get_token_output(next_token)

            # compute logits
            next_logits = self.core.lm_head(output)

            # get next token
            next_token = torch.argmax(next_logits, dim=1)

            # mask tokens predicted by finished sequences
            next_token[~running_sequences] = pad_token_id

            # save token
            predicted_tokens.append(next_token)
            predicted_logits.append(next_logits)

            # if a sequence predicts the "end token", stop that sequence
            running_sequences = torch.mul(running_sequences, next_token != eos_token_id)

        # Get sequence of predicted tokens
        pred_ids = torch.stack(predicted_tokens).transpose(0, 1)
        # Also get sequence of predicted tokens
        pred_embeds = torch.stack(predicted_logits).transpose(0, 1)
        
        return pred_ids, pred_embeds

    def forward(self, x, l_t_prev, out_prev, first=False, last=False):
        """Run RAM for one timestep on a minibatch of images.

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
                last hidden states of each transformers, although they are not used.
                state vector for the previous timestep `t-1`.
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
            output: tuple with two elements. Refer to the `out_prev` argument for
                more information.
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