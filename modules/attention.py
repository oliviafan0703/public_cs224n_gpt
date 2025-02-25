import torch

from einops import rearrange
from torch import nn
import torch.nn.functional as F

class LoRALayer(nn.Module):
  def __init__(self, in_dim, out_dim, rank=8, alpha=16):
    super().__init__()
    self.rank = rank
    self.alpha = alpha
    # FROZEN: Original weight and bias
    self.weight = nn.Parameter(torch.zeros(out_dim, in_dim), requires_grad=False)
    self.bias = nn.Parameter(torch.zeros(out_dim), requires_grad=False)
    # TRAINABLE: LoRA parameters
    self.A = nn.Parameter(torch.randn(out_dim, rank))  # Low-rank matrix A
    self.B = nn.Parameter(torch.zeros(rank, in_dim))  # Low-rank matrix B
    # Scaling factor
    self.scaling = alpha / rank

  def forward(self, x):
    # Output = Wx + (B*A)x * scaling
    return F.linear(x, self.weight + (self.A @ self.B).T * self.scaling, self.bias)

class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Check if ReFT should be used
    self.use_reft = config.use_reft
    
    # Initialize the linear transformation layers for key, value, query.
    if config.use_lora:
      self.query = LoRALayer(config.hidden_size, self.all_head_size)
      self.key = LoRALayer(config.hidden_size, self.all_head_size)
      self.value = LoRALayer(config.hidden_size, self.all_head_size)
    else:
      self.query = nn.Linear(config.hidden_size, self.all_head_size)
      self.key = nn.Linear(config.hidden_size, self.all_head_size)
      self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def apply_reft(self, context):
        # Apply task-specific projection for ReFT if enabled
        if self.use_reft:
            task_specific_projection = nn.Linear(context.size(-1), context.size(-1))  
            fine_tuned_context = task_specific_projection(context)
            return fine_tuned_context
        return context  

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):
    # Compute attention score [b, h, t, t], input dimention is [b, h, t, d]
    attention_scores = torch.matmul(query, key.transpose(-1, -2))
    
    # Scale attention score
    attention_scores = attention_scores / (self.attention_head_size ** 0.5)
    
    # Apply musk
    seq_len = query.size(-2)
    causal_mask = torch.triu(torch.ones((1, 1, seq_len, seq_len), device=attention_mask.device), diagonal=1).bool()
    causal_mask = causal_mask.to(dtype=attention_scores.dtype)
    causal_mask = causal_mask * -1e4
    combined_mask = attention_mask + causal_mask
    attention_scores = attention_scores + combined_mask
    
    # Compute attention prob and apply dropout
    attention_scores = self.dropout(attention_scores)
    attention_probs = torch.softmax(attention_scores, dim=-1)
    
    # Use attention prob to with value
    context = torch.matmul(attention_probs, value)
    
    # rearrange to get the target output dim
    context = rearrange(context, "b h t d -> b t (h d)")

    # Apply ReFT here if enabled
    context = self.apply_reft(context)
    
    return context

  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
