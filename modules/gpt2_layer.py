from torch import nn
import torch
import torch.nn.functional as F

from modules.attention import CausalSelfAttention

class ReFTBlock(nn.Module):
    def __init__(self, hidden_size, rank=8, note="none"):
        super().__init__()
        self.note = note
        self.rank = rank
        self.reft_A = nn.Parameter(torch.randn(hidden_size, rank))
        self.reft_B = nn.Parameter(torch.zeros(rank, hidden_size))
        self.scaling = 1.0 / rank ** 0.5

    def forward(self, x):
        delta = torch.matmul(torch.matmul(x, self.reft_A), self.reft_B) * self.scaling
        return x + delta

class SwiGLU(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.swiglu_intermediate_size = 2048
        self.swiglu_intermediate_size = intermediate_size * 2 // 3  # 3072 -> 2048
        # print(f'hidden_size: {hidden_size}, intermediate_size: {intermediate_size}, self.swiglu_intermediate_size: {self.swiglu_intermediate_size}')
        self.gate_proj = nn.Linear(hidden_size, 2 * self.swiglu_intermediate_size)
        self.out_proj = nn.Linear(self.swiglu_intermediate_size, hidden_size)

    def forward(self, x):
        projected = self.gate_proj(x)  # [bs, seq_len, 2*swiglu_intermediate_size]
        gate, value = torch.chunk(projected, 2, dim=-1)
        swish_gate = F.silu(gate)
        activated = swish_gate * value  # [bs, seq_len, swiglu_intermediate_size]
        # print(f'projected.shape: {projected.shape}, gate.shape: {gate.shape}, x.shape: {x.shape}, activated.shape: {activated.shape}')
        return self.out_proj(activated)  # [bs, seq_len, hidden_size]


class GPT2Layer(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    # Multi-head attention.
    self.self_attention = CausalSelfAttention(config)
    # Add-norm for multi-head attention.
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Feed forward.
    if config.use_swiglu:
        self.swiglu = SwiGLU(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size
        )
    else:
        self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.interm_af = F.gelu
        # Add-norm for feed forward.
        self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)
    # reft layers
    if config.use_reft:
      self.reft_attn = ReFTBlock(config.hidden_size, note="attn_out")
      self.reft_ffn = ReFTBlock(config.hidden_size, note="ffn_out")

  def add(self, input, output, dense_layer, dropout):
    """
    TODO: Implement this helper method for the forward function.
      - This function is applied after the multi-head attention layer as well as after the feed forward layer.
      - GPT-2 layer applies dropout to the transformed output of each sub-layer,
        before it is added to the sub-layer input. WE DO NOT APPLY THE LAYER NORM
        IN THIS FUNCTION.
    """
    projected = dense_layer(output)
    dropped = dropout(projected)
    return input + dropped

  def forward(self, hidden_states, attention_mask):
    """
    TODO: Implement the forward pass. Some key points to consider:
           - A multi-head attention layer (CausalSelfAttention) that computes self-attention based on masked inputs.
           - Layer normalization applied *before* the attention layer and feed-forward layer.
           - Apply dropout, residual connection, and layer normalization according to the plot in the assignment. (Use self.add)
           - A feed-forward layer that applies transformations to further refine the hidden states.
    """

    # Self-attention layer
    # Pre-LN
    attn_norm = self.attention_layer_norm(hidden_states)
    # Self attention compuation
    attn_output = self.self_attention(attn_norm, attention_mask)
    if self.config.use_reft:
      attn_output = self.reft_attn(attn_output)
    # Apply add function
    hidden_states = self.add(
        input = hidden_states,
        output = attn_output,
        dense_layer = self.attention_dense,
        dropout = self.attention_dropout
    )
    
    # Pre-LN
    ffn_norm = self.out_layer_norm(hidden_states)
    if self.config.use_swiglu:
        ffn_output = self.swiglu(ffn_norm)
    else:
        ffn_output = self.interm_af(self.interm_dense(ffn_norm))  # GELU
        ffn_output = self.out_dense(ffn_output)
    if self.config.use_reft:
      ffn_output = self.reft_ffn(ffn_output)
    # direct connect
    hidden_states = self.add(
        input=hidden_states,
        output=ffn_output,
        dense_layer=lambda x: x,
        dropout=self.out_dropout
    )

    return hidden_states

