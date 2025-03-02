'''
Paraphrase detection for GPT starter code.

Consider:
 - ParaphraseGPT: Your implementation of the GPT-2 classification model.
 - train: Training procedure for ParaphraseGPT on the Quora paraphrase detection dataset.
 - test: Test procedure. This function generates the required files for your submission.

Running:
  `python paraphrase_detection.py --use_gpu`
trains and evaluates your ParaphraseGPT model and writes the required submission files.
'''

import argparse
import random
import torch
import math

import numpy as np
import torch.nn.functional as F
from modules.attention import LoRALayer

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import (
  ParaphraseDetectionDataset,
  ParaphraseDetectionTestDataset,
  load_paraphrase_data
)
from evaluation import model_eval_paraphrase, model_test_paraphrase
from models.gpt2 import GPT2Model

from optimizer import AdamW

TQDM_DISABLE = False

# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


# Compute the parameters trainable
def count_trainable_params(model):
  lora_params = 0
  reft_params = 0
  non_trainable_params = 0

  for name, param in model.named_parameters():
    if param.requires_grad:
      # 统计 LoRA 参数
      if 'lora_A' in name or 'lora_B' in name:
        lora_params += param.numel()
      # 统计 ReFT 参数
      elif 'reft_A' in name or 'reft_B' in name:
        reft_params += param.numel()
    else:
      non_trainable_params += param.numel()

  print(f"LoRA Trainable Params: {lora_params}")
  print(f"ReFT Trainable Params: {reft_params}")
  print(f"Total Trainable Params: {lora_params + reft_params}")
  print(f"Total Non-Trainable Params: {non_trainable_params}")
  print(f"Total Trainable Ratio: {(lora_params + reft_params)/(1 + lora_params + reft_params + non_trainable_params)}")
  return lora_params, reft_params, non_trainable_params

class ParaphraseGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, args):
    super().__init__()
    if args.use_lora and not args.use_reft:
      self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads, use_lora=True)
      # Freeze pretained parameters
      for param in self.gpt.parameters():
        param.requires_grad = False
      # Unfreeze Lora parameters
      for name, param in self.gpt.named_parameters():
        if "lora_A" in name or "lora_B" in name:
          print(f'require grad for {name}')
          param.requires_grad = True
      # Init parameters
      for module in self.gpt.modules():
        if isinstance(module, LoRALayer):
          nn.init.kaiming_uniform_(module.lora_A, a=math.sqrt(5))
          nn.init.zeros_(module.lora_B)
    elif args.use_reft and not args.use_lora:
      self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads, use_reft=True)
      # Freeze all parameters
      for param in self.gpt.parameters():
        param.requires_grad = False
      # Train reft parameteres
      for name, param in self.gpt.named_parameters():
        if "reft" in name:
          print(f'require grad for {name}')
          param.requires_grad = True
    elif args.use_reft and args.use_lora:
      self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads, use_lora=True, use_reft=True)
      # Freeze all parameters
      for param in self.gpt.parameters():
        param.requires_grad = False
      # Train reft parameteres
      for name, param in self.gpt.named_parameters():
        print(name, param.shape)
        if "reft" in name:
          print(f'require grad for {name}')
          param.requires_grad = True
      # Unfreeze Lora parameters
      for name, param in self.gpt.named_parameters():
        if "lora_A" in name or "lora_B" in name:
          print(f'require grad for {name}')
          param.requires_grad = True
      # Init parameters
      for module in self.gpt.modules():
        if isinstance(module, LoRALayer):
          nn.init.normal_(module.lora_A, mean=0, std=0.02)
          nn.init.zeros_(module.lora_B)
    else:
        self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads, use_swiglu=args.use_swiglu)
        # By default, fine-tune the full model.
        for param in self.gpt.parameters():
          param.requires_grad = True
    self.paraphrase_detection_head = nn.Linear(args.d, 2)  # Paraphrase detection has two outputs: 1 (yes) or 0 (no).
    count_trainable_params(self.gpt)

  def forward(self, input_ids, attention_mask):
    """
    TODO: Predict the label of the token using the paraphrase_detection_head Linear layer.

    We structure the input as:

      'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '

    So you want to find the prediction for the next token at the end of this sentence. Optimistically, it will be the
    token "yes" (byte pair encoding index of 8505) for examples that are paraphrases or "no" (byte pair encoding index
    of 3919) for examples that are not paraphrases.
    """

    'Takes a batch of sentences and produces embeddings for them.'
    ### YOUR CODE HERE
    gpt_output = self.gpt(input_ids = input_ids, attention_mask = attention_mask)
        
    last_hidden_state = gpt_output['last_hidden_state']
        
    logits = self.paraphrase_detection_head(last_hidden_state[:, -1, :]) 
        
    return logits


def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")

# The function that we use to set the dropout path
def set_dropout_rate(model, p):
  print(f'setting the dropout rate to {p} for the following training')
  for module in model.modules():
    if isinstance(module, nn.Dropout):
      module.p = p

def train(args):
  """Train GPT-2 for paraphrase detection on the Quora dataset."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # Create the data and its corresponding datasets and dataloader.
  para_train_data = load_paraphrase_data(args.para_train)
  para_dev_data = load_paraphrase_data(args.para_dev)

  para_train_data = ParaphraseDetectionDataset(para_train_data, args)
  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)

  para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=para_train_data.collate_fn)
  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)

  args = add_arguments(args)
  model = ParaphraseGPT(args)
  model = model.to(device)

  lr = args.lr
  # optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.)
  from torch import optim
  if args.use_muon_optimizer:
    from muon import Muon
    # Find ≥2D parameters in the body of the network -- these should be optimized by Muon
    muon_params = [p for p in model.parameters() if p.ndim >= 2]
    # Find everything else -- these should be optimized by AdamW
    adamw_params = [p for p in model.parameters() if p.ndim < 2]
    # Create the optimizers from both muon and adamw
    optimizers = [Muon(muon_params, lr=0.02, momentum=0.95),
                  optim.AdamW(adamw_params, lr=lr, weight_decay=1e-6)]
  else:
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
  best_dev_acc = 0

  correct_predictions = 0 
  total_predictions = 0

  # Run for the specified number of epochs.
  print(f'num epochs to train: {args.epochs}')
  for epoch in range(args.epochs):
    print(f'current epoch: {epoch}')
    if args.use_early_dropout:
      stop_dropout_rate_epoch = int(args.stop_dropout_rate_epoch_ratio * args.epochs)
      current_dropout_rate_to_use = max(0, args.end_dropout_rate + (1 - epoch / stop_dropout_rate_epoch) * (args.early_dropout_rate - args.end_dropout_rate))
      # Print debug info
      print(f'use_early_dropout, stop_dropout_rate_epoch_ratio: {args.stop_dropout_rate_epoch_ratio}, '
            f'stop_dropout_rate_epoch: {stop_dropout_rate_epoch}, epoch: {epoch}, early_dropout_rate: {args.early_dropout_rate}, '
            f'end_dropout_rate: {args.end_dropout_rate}, current_dropout_rate_to_use: {current_dropout_rate_to_use}')
      if epoch < stop_dropout_rate_epoch:
        print(f'setting dropout rate {current_dropout_rate_to_use}')
        set_dropout_rate(model, current_dropout_rate_to_use)  # set dropout value
      else:
        print(f'setting dropout rate {args.end_dropout_rate}')
        set_dropout_rate(model, args.end_dropout_rate)        # end dropout value
    model.train()
    train_loss = 0
    num_batches = 0
    for i, batch in enumerate(tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)):
      # Get the input and move it to the gpu (I do not recommend training this model on CPU).
      b_ids, b_mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels'].flatten()
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)
      labels = labels.to(device)

      # The labels are 1 for paraphrases and 0 for non-paraphrases.
      labels = torch.where(labels == 8505, torch.tensor(1, device=labels.device), torch.tensor(0, device=labels.device))
      mapped_labels = labels.long()

      # Compute the loss, gradients, and update the model's parameters.
      if args.use_muon_optimizer:
        for optimizer in optimizers:
          optimizer.zero_grad()
        logits = model(b_ids, b_mask)
        preds = torch.argmax(logits, dim=1)
        loss = F.cross_entropy(logits, mapped_labels, reduction='mean')
        loss.backward()
        for optimizer in optimizers:
          optimizer.step()
      else:
        optimizer.zero_grad()
        logits = model(b_ids, b_mask)
        preds = torch.argmax(logits, dim=1)
        loss = F.cross_entropy(logits, mapped_labels, reduction='mean')
        loss.backward()
        optimizer.step()

      train_loss += loss.item()
      num_batches += 1

      correct_predictions += (preds == mapped_labels).sum().item()
      total_predictions += mapped_labels.size(0)

      # Print accuracy every 1000 batches
      if (i + 1) % 1000 == 0:
        acc = correct_predictions / total_predictions
        print(f"Epoch {epoch}, Batch {i+1}: Train Accuracy: {acc:.4f}")


    train_loss = train_loss / num_batches

    dev_acc, dev_f1, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)

    if dev_acc > best_dev_acc:
      best_dev_acc = dev_acc
      save_model(model, optimizer, args, args.filepath)

    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}")


@torch.no_grad()
def test(args):
  """Evaluate your model on the dev and test datasets; save the predictions to disk."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(args.filepath)

  model = ParaphraseGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()
  print(f"Loaded model to test from {args.filepath}")

  para_dev_data = load_paraphrase_data(args.para_dev)
  para_test_data = load_paraphrase_data(args.para_test, split='test')

  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
  para_test_data = ParaphraseDetectionTestDataset(para_test_data, args)

  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)
  para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                    collate_fn=para_test_data.collate_fn)

  dev_para_acc, _, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase(para_dev_dataloader, model, device)
  print(f"dev paraphrase acc :: {dev_para_acc :.3f}")
  test_para_y_pred, test_para_sent_ids = model_test_paraphrase(para_test_dataloader, model, device)

  with open(args.para_dev_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
      f.write(f"{p}, {s} \n")

  with open(args.para_test_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(test_para_sent_ids, test_para_y_pred):
      f.write(f"{p}, {s} \n")


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
  parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
  parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_lora", action='store_true')
  parser.add_argument("--use_reft", action='store_true')
  parser.add_argument("--use_swiglu", action='store_true')
  parser.add_argument("--use_gpu", action='store_true')
  parser.add_argument("--use_early_dropout", action='store_true')
  parser.add_argument("--early_dropout_rate", type=float, help="early dropout rate", default=0.3)
  parser.add_argument("--end_dropout_rate", type=float, help="end dropout rate", default=0.1)
  parser.add_argument("--stop_dropout_rate_epoch_ratio", type=float, help="end dropout rate", default=0.8)
  parser.add_argument("--use_muon_optimizer", action='store_true')

  parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str,
                      help="The model size as specified on hugging face. DO NOT use the xl model.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')

  args = parser.parse_args()
  print(f'use_lora: {args.use_lora}')
  print(f'use_reft: {args.use_reft}')
  print(f'use_swiglu: {args.use_swiglu}')
  print(f'use_muon_optimizer: {args.use_muon_optimizer}')
  print(f'dropout schedule: use_early_dropout {args.use_early_dropout}, early_dropout_rate {args.early_dropout_rate}, '
        f'end_dropout_rate {args.end_dropout_rate},stop_dropout_rate_epoch_ratio {args.stop_dropout_rate_epoch_ratio}')
  if args.use_early_dropout:
    assert args.early_dropout_rate >= 0
    assert args.end_dropout_rate >= 0
    assert args.stop_dropout_rate_epoch_ratio >= 0
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-paraphrase.pt'  # Save path.
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  train(args)
  test(args)
