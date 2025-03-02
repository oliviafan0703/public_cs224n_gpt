import torch
import math
from torch import nn
from transformers import GPT2Model as OpenAIGPT2Model

from config import GPT2Config
from models.base_gpt import GPTPreTrainedModel
from modules.gpt2_layer import GPT2Layer
from utils import get_extended_attention_mask


class GPT2Model(GPTPreTrainedModel):
  """
  The GPT model returns the final embeddings for each token in a sentence.

  The model consists of:
  1. Embedding layers (used in self.embed).
  2. A stack of n GPT layers (used in self.encode).
  3. A linear transformation layer for the [CLS] token (used in self.forward, as given).
  """

  def __init__(self, config):
    super().__init__(config)
    self.config = config
    self.use_lora = self.config.use_lora
    self.use_reft = self.config.use_reft

    # Embedding layers.
    self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)

    # Register position_ids (1, len position emb) to buffer because it is a constant.
    position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
    self.register_buffer('position_ids', position_ids)

    # GPT-2 layers.
    self.gpt_layers = nn.ModuleList([GPT2Layer(config) for _ in range(config.num_hidden_layers)])

    # [CLS] token transformations.
    self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.pooler_af = nn.Tanh()

    # Final layer norm.
    self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    self.init_weights()

  def embed(self, input_ids):
    input_shape = input_ids.size()
    seq_length = input_shape[1]

    inputs_embeds = None

    ### YOUR CODE HERE
    input_shape = input_ids.size()
    seq_length = input_shape[1]

    # Get the word embeddings for each token
    word_embeds = self.word_embedding(input_ids)

    pos_ids = self.position_ids[:, :seq_length]
    pos_embeds = None

    ### TODO: Use pos_ids to get position embedding from self.pos_embedding into pos_embeds.
    ###       Then, add two embeddings together; then apply dropout and return.
    ### YOUR CODE HERE
    pos_embeds = self.pos_embedding(pos_ids)
    combined_embeds = word_embeds + pos_embeds
    embedding_output = self.embed_dropout(combined_embeds)

    return embedding_output

  def encode(self, hidden_states, attention_mask):
    """
    hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
    attention_mask: [batch_size, seq_len]
    """
    # Get the extended attention mask for self-attention.
    # Returns extended_attention_mask of size [batch_size, 1, 1, seq_len].
    # Distinguishes between non-padding tokens (with a value of 0) and padding tokens
    # (with a value of a large negative number).
    extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)

    # Pass the hidden states through the encoder layers.
    for i, layer_module in enumerate(self.gpt_layers):
      # Feed the encoding from the last bert_layer to the next.
      hidden_states = layer_module(hidden_states, extended_attention_mask)

    return hidden_states

  def forward(self, input_ids, attention_mask):
    """
    input_ids: [batch_size, seq_len], seq_len is the max length of the batch
    attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
    """
    # Get the embedding for each input token.
    embedding_output = self.embed(input_ids=input_ids)

    # Feed to a transformer (a stack of GPTLayers).
    sequence_output = self.encode(embedding_output, attention_mask=attention_mask)
    sequence_output = self.final_layer_norm(sequence_output)

    # Get the hidden state of the final token.
    last_non_pad_idx = attention_mask.sum(dim=1) - 1  # Subtract 1 to get last index
    last_token = sequence_output[torch.arange(sequence_output.shape[0]), last_non_pad_idx]

    return {'last_hidden_state': sequence_output, 'last_token': last_token}

  def hidden_state_to_token(self, hidden_state):
    """
    GPT-2 uses weight tying with the input word embeddings. The logits are the dot product between output hidden states
    and the word embedding weights:

      return hidden_state(s) * E^T
    """
    ### YOUR CODE HERE
    embedding_weights = self.word_embedding.weight.transpose(0, 1)
    logits = torch.matmul(hidden_state, embedding_weights)

    return logits


  @classmethod
  def from_pretrained(cls, model='gpt2', d=768, l=12, num_heads=12, use_lora=False, use_reft=False, use_swiglu=False):
    gpt_model = OpenAIGPT2Model.from_pretrained(model).eval()
    our_model = GPT2Model(GPT2Config(hidden_size=d, num_hidden_layers=l, num_attention_heads=num_heads,
                                     intermediate_size=d*3,
                                     use_lora=use_lora, use_reft=use_reft, use_swiglu=use_swiglu)).eval()

    # Load word and positional embeddings.
    our_model.word_embedding.load_state_dict(gpt_model.wte.state_dict())
    our_model.pos_embedding.load_state_dict(gpt_model.wpe.state_dict())

    for i in range(l):
      l = our_model.gpt_layers[i]
      # Remap the Q,K,V weights from a conv1d to 3 linear projections
      l.self_attention.query.weight.data = gpt_model.state_dict()[f'h.{i}.attn.c_attn.weight'][:, :d].T
      l.self_attention.query.bias.data = gpt_model.state_dict()[f'h.{i}.attn.c_attn.bias'][:d]
      l.self_attention.key.weight.data = gpt_model.state_dict()[f'h.{i}.attn.c_attn.weight'][:, d:d*2].T
      l.self_attention.key.bias.data = gpt_model.state_dict()[f'h.{i}.attn.c_attn.bias'][d:d*2]
      l.self_attention.value.weight.data = gpt_model.state_dict()[f'h.{i}.attn.c_attn.weight'][:, d*2:].T
      l.self_attention.value.bias.data = gpt_model.state_dict()[f'h.{i}.attn.c_attn.bias'][d*2:]

      # Remap final dense layer in MHA.
      l.attention_dense.weight.data = gpt_model.state_dict()[f'h.{i}.attn.c_proj.weight'].T
      l.attention_dense.bias.data = gpt_model.state_dict()[f'h.{i}.attn.c_proj.bias']

      # Remap attention layer norm.
      l.attention_layer_norm.weight.data = gpt_model.state_dict()[f'h.{i}.ln_1.weight']
      l.attention_layer_norm.bias.data = gpt_model.state_dict()[f'h.{i}.ln_1.bias']

      # When we change to swiglu, we need to change to a different initialization method
      if use_swiglu:
        print(f'initializing swiglu parameters with original weights for layer {i}')
        # Get the original weight from the pre-trained model
        c_fc_weight = gpt_model.state_dict()[f'h.{i}.mlp.c_fc.weight']  # [768, 3072]
        c_proj_weight = gpt_model.state_dict()[f'h.{i}.mlp.c_proj.weight']  # [3072, 768]
        # Use the same calculation as in your SwiGLU class (intermediate_size = d*3 = 2304)
        swiglu_intermediate_size = (d * 3) * 2 // 3  # 2304 * 2 // 3 = 1536
        # Make sure there's enough data in the original weights to extract what we need
        if c_fc_weight.shape[1] < swiglu_intermediate_size * 2:
          # You may need to handle this case - perhaps with padding or truncation
          print(
            f"Warning: Original weight dimension {c_fc_weight.shape[1]} is smaller than required {swiglu_intermediate_size * 2}")
        # split the initial value (may need to adjust slicing if dimensions don't match exactly)
        gate_weight = c_fc_weight[:, :swiglu_intermediate_size]  # [768, 1536]
        value_weight = c_fc_weight[:, swiglu_intermediate_size: 2 * swiglu_intermediate_size]  # [768, 1536]
        if True:
          nn.init.normal_(gate_weight, mean=0.0, std=0.02 / math.sqrt(swiglu_intermediate_size))
          nn.init.kaiming_normal_(value_weight, mode='fan_in', nonlinearity='linear')
        # combine weights
        swiglu_gate_weight = torch.cat([gate_weight, value_weight], dim=1)  # [768, 3072]
        l.swiglu.gate_proj.weight.data = swiglu_gate_weight.T  # [3072, 768]
        # change output layer
        l.swiglu.out_proj.weight.data = c_proj_weight[:swiglu_intermediate_size, :].T  # [768, 1536]
        # Don't forget to update biases if needed
        if 'c_fc.bias' in gpt_model.state_dict():
          c_fc_bias = gpt_model.state_dict()[f'h.{i}.mlp.c_fc.bias']
          gate_bias = c_fc_bias[:swiglu_intermediate_size]
          value_bias = c_fc_bias[swiglu_intermediate_size:2 * swiglu_intermediate_size]
          swiglu_gate_bias = torch.cat([gate_bias, value_bias])
          l.swiglu.gate_proj.bias.data = swiglu_gate_bias
        if 'c_proj.bias' in gpt_model.state_dict():
          l.swiglu.out_proj.bias.data = gpt_model.state_dict()[f'h.{i}.mlp.c_proj.bias']
      else:
        # Remap post-attention MLP layers.
        l.interm_dense.weight.data = gpt_model.state_dict()[f'h.{i}.mlp.c_fc.weight'].T
        l.interm_dense.bias.data = gpt_model.state_dict()[f'h.{i}.mlp.c_fc.bias']
        l.out_dense.weight.data = gpt_model.state_dict()[f'h.{i}.mlp.c_proj.weight'].T
        l.out_dense.bias.data = gpt_model.state_dict()[f'h.{i}.mlp.c_proj.bias']

      # Remap second layer norm weights.
      l.out_layer_norm.weight.data = gpt_model.state_dict()[f'h.{i}.ln_2.weight']
      l.out_layer_norm.bias.data = gpt_model.state_dict()[f'h.{i}.ln_2.bias']

      # Init the values for ReFT if defined
      if use_reft:
        nn.init.kaiming_normal_(l.reft_attn.reft_A, mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_normal_(l.reft_ffn.reft_A, mode='fan_in', nonlinearity='linear')
        # nn.init.normal_(l.reft_attn.reft_A, mean=0, std=0.02)
        nn.init.zeros_(l.reft_attn.reft_B)
        # nn.init.normal_(l.reft_ffn.reft_A, mean=0, std=0.02)
        nn.init.zeros_(l.reft_ffn.reft_B)

    # Remap the final layer norm values.
    our_model.final_layer_norm.weight.data = gpt_model.state_dict()['ln_f.weight']
    our_model.final_layer_norm.bias.data = gpt_model.state_dict()['ln_f.bias']

    return our_model
