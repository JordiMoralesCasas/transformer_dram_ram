"""
COPIED FROM https://github.com/opendilab/PPOxFamily/blob/main/chapter5_time/gtrxl.py
Modified to allow for crossattention with standard Transformer's encoder hidden states (CrossAttentionGTrXL class).
Also, a wrapper (GTrXL_wrapper) is created to better integrate the GTrXL with our existing code.


Gated Transformer XL (GTrXL) <link https://arxiv.org/abs/1910.06764 link> is a stabilized transformer architecture for reinforcement learning.
This document mainly includes:
- Pytorch implementation for GTrXL.
- An example to test GTrXL.
"""
from typing import Optional, Dict
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


import treetensor
from ding.torch_utils import GRUGatingUnit, build_normalization
from ding.torch_utils.network.nn_module import fc_block
from ding.torch_utils.network.gtrxl import PositionalEmbedding, Memory, AttentionXL


class CrossAttentionGTrXL(torch.nn.Module):
    """
    Overview:
         An implementation of the Attention mechanism used in the TransformerXL model.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, input_dim: int, head_dim: int, head_num: int, dropout: nn.Module) -> None:
        """
        Overview:
            Initialize the AttentionXL module.
        Arguments:
            - input_dim (:obj:`int`): The dimensionality of the input features.
            - head_dim (:obj:`int`): The dimensionality of each attention head.
            - head_num (:obj:`int`): The number of attention heads.
            - dropout (:obj:`nn.Module`): The dropout layer to use
        """

        super(CrossAttentionGTrXL, self).__init__()
        self.head_num = head_num
        self.head_dim = head_dim
        self.dropout = dropout
        self.attention_kv = fc_block(input_dim, head_dim * head_num * 2)  # key, value
        self.attention_q = fc_block(input_dim, head_dim * head_num)  # query (not computed with past hidden states)
        self.project = fc_block(head_dim * head_num, input_dim)  # project attention output back to input_dim
        #self.project_pos = fc_block(input_dim, head_dim * head_num)  # project the positional embedding
        self.scale = 1 / (head_dim ** 0.5)  # for scaled dot product attention

    def _rel_shift(self, x: torch.Tensor, zero_upper: bool = False) -> torch.Tensor:
        """
        Overview:
            Perform a relative shift operation on the attention score matrix.
            Example:
                a00 a01 a02      0 a00 a01 a02       0  a00 a01      a02  0  a10     a02  0   0
                a10 a11 a12  =>  0 a10 a11 a12  =>  a02  0  a10  =>  a11 a12  0  =>  a11 a12  0
                a20 a21 a22      0 a20 a21 a22      a11 a12  0       a20 a21 a22     a20 a21 a22
                                                    a20 a21 a22
                1) Append one "column" of zeros to the left
                2) Reshape the matrix from [3 x 4] into [4 x 3]
                3) Remove the first "row"
                4) Mask out the upper triangle (optional)

        .. note::
            See the following material for better understanding:
                https://github.com/kimiyoung/transformer-xl/issues/8
                https://arxiv.org/pdf/1901.02860.pdf (Appendix B)
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor with shape (cur_seq, full_seq, bs, head_num).
            - zero_upper (:obj:`bool`): If True, the upper-right triangle of the matrix is set to zero.
        Returns:
            - x (:obj:`torch.Tensor`): The input tensor after the relative shift operation, \
                with shape (cur_seq, full_seq, bs, head_num).
        """

        x_padded = F.pad(x, [1, 0])  # step 1
        x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))  # step 2
        x = x_padded[:, :, 1:].view_as(x)  # step 3
        if zero_upper:
            ones = torch.ones((x.size(2), x.size(3))).unsqueeze(0).unsqueeze(0)
            x = x * torch.tril(ones.to(x.device), x.size(3) - x.size(2))  # step 4
        return x

    def forward(
            self,
            inputs: torch.Tensor,
            pos_embedding: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            u: torch.nn.Parameter,
            v: torch.nn.Parameter,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Overview:
            Compute the forward pass for the AttentionXL module.
        Arguments:
            - inputs (:obj:`torch.Tensor`): The attention input with shape (cur_seq, bs, input_dim).
            - pos_embedding (:obj:`torch.Tensor`): The positional embedding with shape (full_seq, 1, full_seq).
            - full_input (:obj:`torch.Tensor`): The concatenated memory and input tensor with shape \
                (full_seq, bs, input_dim).
            - u (:obj:`torch.nn.Parameter`): The content parameter with shape (head_num, head_dim).
            - v (:obj:`torch.nn.Parameter`): The position parameter with shape (head_num, head_dim).
            - mask (:obj:`Optional[torch.Tensor]`): The attention mask with shape (cur_seq, full_seq, 1). \
                If None, no masking is applied.
        Returns:
            - output (:obj:`torch.Tensor`): The output of the attention mechanism with shape (cur_seq, bs, input_dim).
        """

        bs, dec_seq, enc_seq = inputs.shape[1], inputs.shape[0], encoder_hidden_states.shape[0]

        kv = self.attention_kv(encoder_hidden_states)
        key, value = torch.chunk(kv, 2, dim=-1)  # k and v from the encoder hidden states
        query = self.attention_q(inputs)  # q from the decoder inputs
        #r = self.project_pos(pos_embedding)  # full_seq x 1 x num_head*dim_head

        key = key.view(enc_seq, bs, self.head_num, self.head_dim)
        value = value.view(enc_seq, bs, self.head_num, self.head_dim)
        query = query.view(dec_seq, bs, self.head_num, self.head_dim)
        #r = r.view(dec_seq, self.head_num, self.head_dim)

        # (query + u) * key^T
        q_u = query + u
        content_attn = q_u.permute(1, 2, 0, 3) @ key.permute(1, 2, 3, 0)  # bs x head_num x cur_seq x full_seq

        # (query + v) * R^T
        #q_v = query + v
        #position_attn = q_v.permute(1, 2, 0, 3) @ r.permute(1, 2, 0)  # bs x head_num x cur_seq x full_seq
        #position_attn = self._rel_shift(position_attn)

        attn = content_attn# + position_attn  # bs x head_num x cur_seq x full_seq
        attn.mul_(self.scale)

        # fills float('-inf') where mask is True to let softmax ignore those positions.
        if mask is not None and mask.any().item():
            mask = mask.permute(2, 0, 1).unsqueeze(1)  # 1 x 1 x cur_seq x full_seq
            assert mask.shape[2:] == attn.shape[2:]  # check shape of mask
            attn = attn.masked_fill(mask, -float("inf")).type_as(attn)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # multiply softmax output by value
        attn_vec = attn @ value.permute(1, 2, 0, 3)
        attn_vec = attn_vec.permute(2, 0, 1, 3)

        attn_vec = attn_vec.contiguous().view(dec_seq, bs, self.head_num * self.head_dim)
        # cur_seq x bs x head_num * head_dim
        output = self.dropout(self.project(attn_vec))  # cur_seq x bs x input_dim
        return output


class GatedTransformerXLLayer(torch.nn.Module):
    """
    **Overview**:
        The basic layer design of Gated Transformer-XL. This module mainly includes AttentionXL,
        Feed-Forward-Network, layer normalization, and GRU-gating.
    """
    def __init__(
            self,
            input_dim: int,
            head_dim: int,
            hidden_dim: int,
            head_num: int,
            mlp_num: int,
            dropout: nn.Module,
            activation: nn.Module,
            gru_gating: bool = True,
            gru_bias: float = 2.,
            do_crossattention: bool = False
    ) -> None:
        super(GatedTransformerXLLayer, self).__init__()
        self.do_crossattention = do_crossattention
        self.dropout = dropout
        # Decide whether to use GRU-gating.
        self.gating = gru_gating
        if self.gating is True:
            self.gate1 = GRUGatingUnit(input_dim, gru_bias)
            self.gate2 = GRUGatingUnit(input_dim, gru_bias)
            if do_crossattention:
                self.gate_ca = GRUGatingUnit(input_dim, gru_bias)
        # Build attention block using the AttentionXL class,
        # a feed-forward network with optional dropout, and two layer normalization layers.
        self.attention = AttentionXL(
            input_dim,
            head_dim,
            head_num,
            dropout,
        )
        if self.do_crossattention:
            self.crossattention = CrossAttentionGTrXL(
                input_dim,
                head_dim,
                head_num,
                dropout,
            )

        # Build Feed-Forward-Network.
        layers = []
        dims = [input_dim] + [hidden_dim] * (mlp_num - 1) + [input_dim]
        for i in range(mlp_num):
            layers.append(fc_block(dims[i], dims[i + 1], activation=activation))
            if i != mlp_num - 1:
                layers.append(self.dropout)
        layers.append(self.dropout)
        self.mlp = nn.Sequential(*layers)
        # Build layer norm.
        self.layernorm1 = build_normalization('LN')(input_dim)
        self.layernorm2 = build_normalization('LN')(input_dim)
        if self.do_crossattention:
            self.layernorm_ca = build_normalization('LN')(input_dim)
        self.activation = activation

    # delimiter
    def forward(
            self,
            inputs: torch.Tensor,
            pos_embedding: torch.Tensor,
            u: torch.nn.Parameter,
            v: torch.nn.Parameter,
            memory: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        **Overview**:
            The forward computation graph of GTrXL layer.
        """
        # Concat memory with input across sequence dimension. The shape is: [full_sequence, batch_size, input_dim]
        full_input = torch.cat([memory, inputs], dim=0)
        # Forward calculation for GTrXL layer.
        # In GTrXL, the layer normalization is put before the attention layer.
        x1 = self.layernorm1(full_input)
        # Attention module.
        a1 = self.dropout(self.attention(inputs, pos_embedding, x1, u, v, mask=mask))
        a1 = self.activation(a1)
        # In GTrXL, gating layer replace the resnet layer in TrXL.
        o1 = self.gate1(inputs, a1) if self.gating else inputs + a1

        if self.do_crossattention:
            xca = self.layernorm_ca(o1)
            # Attention module.
            aca = self.dropout(self.crossattention(xca, pos_embedding, encoder_hidden_states, u, v))
            aca = self.activation(aca)
            # In GTrXL, gating layer replace the resnet layer in TrXL.
            oca = self.gate_ca(o1, aca) if self.gating else inputs + aca
        else:
            oca = o1

        x2 = self.layernorm2(oca)
        # Feed Forward Network.
        m2 = self.dropout(self.mlp(x2))
        o2 = self.gate2(oca, m2) if self.gating else oca + m2
        return o2


# delimiter
class GTrXL(nn.Module):
    """
    **Overview**:
        PyTorch implementation for GTrXL, which is used to model the long-term time dependency in reinforcement learning.
    """
    def __init__(
        self,
        input_dim: int,
        head_dim: int = 128,
        embedding_dim: int = 256,
        head_num: int = 2,
        mlp_num: int = 2,
        layer_num: int = 3,
        memory_len: int = 64,
        dropout_ratio: float = 0.,
        activation: nn.Module = nn.ReLU(),
        gru_gating: bool = True,
        gru_bias: float = 2.,
        use_embedding_layer: bool = True,
        do_crossattention: bool = True
    ) -> None:
        super(GTrXL, self).__init__()
        assert embedding_dim % 2 == 0, 'embedding_dim={} should be even'.format(input_dim)
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.embedding_dim = embedding_dim
        if isinstance(input_dim, list):
            input_dim = np.prod(input_dim)
        # Initialize embedding layer.
        self.use_embedding_layer = use_embedding_layer
        if self.use_embedding_layer:
            self.embedding = fc_block(input_dim, embedding_dim, activation=activation)
        # Initialize activate function.
        self.activation = activation
        # Initialize position embedding.
        self.pos_embedding = PositionalEmbedding(embedding_dim)
        # Memory to save hidden states of past segments. It will be initialized in the forward method to get its size dynamically.
        self.memory = None
        self.memory_len = memory_len
        # Initialize GTrXL layers.
        layers = []
        # Put all the embedding_dims into a list.
        # For the i-th layer, the input embedding is dims[i], while the output embedding is dims[i+1]
        dims = [embedding_dim] + [embedding_dim] * layer_num
        self.dropout = nn.Dropout(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        for i in range(layer_num):
            layers.append(
                GatedTransformerXLLayer(
                    dims[i], head_dim, dims[i+1], head_num, mlp_num, self.dropout, self.activation, gru_gating,
                    gru_bias, do_crossattention
                )
            )
        self.layers = nn.Sequential(*layers)
        # u and v are the parameters to compute global content bias and global positional bias.
        self.u, self.v = (
            torch.nn.Parameter(torch.zeros(self.head_num, self.head_dim)),
            torch.nn.Parameter(torch.zeros(self.head_num, self.head_dim)),
        )
        # Create an attention mask for each different seq_len. In this way we don't need to create a new one each time we call the forward method.
        self.att_mask = {}
        # Create a pos embedding for each different seq_len. In this way we don't need to create a new one each time we call the forward method.
        self.pos_embedding_dict = {}

    # delimiter
    def reset_memory(self, batch_size: Optional[int] = None, state: Optional[torch.Tensor] = None):
        """
        **Overview**:
            Reset the memory of GTrXL, which is called at the beginning of each episode.
            Memory is used to save hidden states of past segments.
        """
        # Reset the memory of GTrXL.
        self.memory = Memory(memory_len=self.memory_len, layer_num=self.layer_num, embedding_dim=self.embedding_dim)
        # If batch_size is not None, specify the batch_size when initializing the memory.
        if batch_size is not None:
            self.memory = Memory(self.memory_len, batch_size, self.embedding_dim, self.layer_num)
        # If state is not None, add state into the memory.
        elif state is not None:
            self.memory.init(state)

    # delimiter
    def get_memory(self):
        """
        **Overview**:
            Access the memory of GTrXL.
        """
        # Get the memory of GTrXL.
        if self.memory is None:
            return None
        else:
            return self.memory.get()

    # delimiter
    def forward(self, x: torch.Tensor, encoder_hidden_states: torch.Tensor = None, batch_first: bool = False, return_mem: bool = True) -> Dict[str, torch.Tensor]:
        """
        **Overview**:
            The forward computation graph of GTrXL.
        """
        # If the first dimension of input x is batch_size,
        # then reshape x from  [batch_size ,sequence_length ,input_dim] to [sequence_length, batch_size, input_dim]
        if batch_first:
            x = torch.transpose(x, 1, 0)
            if encoder_hidden_states is not None:
                encoder_hidden_states = torch.transpose(encoder_hidden_states, 1, 0)
        cur_seq, bs = x.shape[:2]
        # Get back memory.
        memory = None if self.memory is None else self.memory.get()
        # Abnormal case: no memory or memory shape mismatch.
        if memory is None:
            self.reset_memory(bs)
        elif memory.shape[-2] != bs or memory.shape[-1] != self.embedding_dim:
            warnings.warn(
                "Memory {} and Input {} dimensions don't match,"
                " this will cause the memory to be initialized to fit your input!".format(
                    list(memory.shape[-2:]), [x.shape[-2]] + [self.embedding_dim]
                )
            )
            self.reset_memory(bs)
        self.memory.to(x.device)
        memory = self.memory.get()
        # Pass through embedding layer.
        if self.use_embedding_layer:
            x = self.dropout(self.embedding(x))
        # Get full sequence length: memory length + current length
        prev_seq = self.memory_len
        full_seq = cur_seq + prev_seq
        # If the attention mask for current sequence length is already created, reuse the mask stored in ``self.att_mask`` .
        if cur_seq in self.att_mask.keys():
            attn_mask = self.att_mask[cur_seq]
        # Otherwise, create a new attention mask and store it into ``self.att_mask`` .
        else:
            # For example, if cur_seq = 3, full_seq = 7, then the mask is:
            # $$ \begin{matrix} 0 & 0 & 0 & 0 & 0 & 1 & 1 \\ 0 & 0 & 0 & 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 \end{matrix}$$
            # This forces that the hidden state of current token is only associated with previous tokens.
            attn_mask = (
                torch.triu(
                    torch.ones((cur_seq, full_seq)),
                    diagonal=1 + prev_seq,
                ).bool().unsqueeze(-1).to(x.device)
            )
            self.att_mask[cur_seq] = attn_mask
        # If the position encoding for current sequence length is already created, reuse it stored in ``self.pos_embedding_dict`` .
        if cur_seq in self.pos_embedding_dict.keys():
            pos_embedding = self.pos_embedding_dict[cur_seq]
        # Otherwise, create a new position encoding and store it into ``self.pos_embedding_dict`` .
        else:
            pos_ips = torch.arange(full_seq - 1, -1, -1.0, dtype=torch.float)  # full_seq
            pos_embedding = self.pos_embedding(pos_ips.to(x.device))
            self.pos_embedding_dict[cur_seq] = pos_embedding
        pos_embedding = self.dropout(pos_embedding)  # full_seq x 1 x embedding_dim

        hidden_state = [x]
        out = x
        # Calculate results for each GTrXL layer.
        for i in range(self.layer_num):
            layer = self.layers[i]
            out = layer(
                out,
                pos_embedding,
                self.u,
                self.v,
                mask=attn_mask,
                memory=memory[i],
                encoder_hidden_states=encoder_hidden_states
            )
            hidden_state.append(out.clone())
        out = self.dropout(out)
        # Update the GTrXL memory.
        self.memory.update(hidden_state)
        # If the first dimension of output is required to be batch_size, then reshape x from  [sequence_length, batch_size, input_dim] to [batch_size ,sequence_length ,input_dim].
        if batch_first:
            out = torch.transpose(out, 1, 0)
        # Return memory is needed.
        if return_mem:
            output = treetensor.Object({"logit": out, "memory": memory})
        else:
            output = treetensor.Object({"logit": out})
        return output


"""
    class to fit GTrXL into the existing code
"""
class GTrXL_wrapper(nn.Module):
    def __init__(
        self,
        input_dim: int,
        head_dim: int = 128,
        embedding_dim: int = 256,
        head_num: int = 2,
        mlp_num: int = 2,
        layer_num: int = 3,
        memory_len: int = 64,
        dropout_ratio: float = 0.,
        activation: nn.Module = nn.ReLU(),
        gru_gating: bool = True,
        gru_bias: float = 2.,
        use_embedding_layer: bool = True,
        do_crossattention: bool = False
        ):
        super(GTrXL_wrapper, self).__init__()

        self.gtrxl = GTrXL(
            input_dim,
            head_dim = head_dim,
            embedding_dim = embedding_dim,
            head_num = head_num,
            mlp_num = mlp_num,
            layer_num = layer_num,
            memory_len = memory_len,
            dropout_ratio = dropout_ratio,
            activation = activation,
            gru_gating = gru_gating,
            gru_bias = gru_bias,
            use_embedding_layer = use_embedding_layer,
            do_crossattention = do_crossattention
        )

    def forward(self, inputs_embeds=None, encoder_hidden_states=None):
        outputs = self.gtrxl(
            x=inputs_embeds, batch_first=True, return_mem=False, encoder_hidden_states=encoder_hidden_states)
        return treetensor.Object({"last_hidden_state": outputs.logit})


if __name__ == '__main__':
    # delimiter
    def test_gtrxl() -> None:
        """
        **Overview**:
            Test function of GTrXL.
        """
        # Generate data for testing.
        input_dim = 128
        seq_len = 64
        bs = 32
        embedding_dim = 256
        layer_num = 5
        mem_len = 40
        memory = [None, torch.rand(layer_num + 1, mem_len, bs, embedding_dim)]

        # Test GTrXL under different situations.
        for i in range(2):
            m = memory[i]
            model = GTrXL(
                input_dim=input_dim,
                head_dim=2,
                embedding_dim=embedding_dim,
                memory_len=mem_len,
                head_num=2,
                mlp_num=2,
                layer_num=layer_num,
            )
            # Input shape: [sequence_length, batch_size, input_dim]
            input = torch.rand(seq_len, bs, input_dim, requires_grad=True)
            # Reset the model memory.
            if m is None:
                model.reset_memory(batch_size=bs)
            else:
                model.reset_memory(state=m)
            output = model(input)
            # Check the shape of output.
            assert output['logit'].shape == (seq_len, bs, embedding_dim)
            assert output['memory'].shape == (layer_num + 1, mem_len, bs, embedding_dim)
            torch.sum(output['logit']).backward()
            # Check the gradient.
            assert isinstance(input.grad, torch.Tensor)
            # Check memory.
            memory_out = output['memory']
            if m is not None:
                assert torch.all(torch.eq(memory_out, m))

    test_gtrxl()