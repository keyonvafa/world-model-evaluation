from model import SimpleTokenizer, collate_fn
from utils import load_model, load_heldout_data
from torch.utils.data import DataLoader
import torch
import numpy as np
import argparse
from tqdm import tqdm
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='shortest-paths')
parser.add_argument('--use-untrained-model', action='store_true')
args = parser.parse_args()
data = args.data
use_untrained_model = args.use_untrained_model

model = load_model(data, use_untrained_model)
tokenizer = model.tokenizer
valid_turns = tokenizer.valid_turns
node_and_direction_to_neighbor = tokenizer.node_and_direction_to_neighbor

dataset = load_heldout_data(data, tokenizer)

batch_size = 128
num_success = 0
num_total = 0
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
bar = tqdm(dataloader)
for batch in bar:
  bsz, _ = batch['input_ids'].shape
  with torch.no_grad():
    input_ids = batch['input_ids'].to(model.device)
    mask = batch['attention_mask'].to(model.device)
    outputs = model.model(input_ids, attention_mask=mask, labels=input_ids)
    logits = outputs.logits
    top_preds = torch.argmax(logits, dim=-1)
  
  for i in range(bsz):
    sequence_str = tokenizer.decode(batch['input_ids'][i])
    sequence_list = sequence_str.split(" ")
    start_node, end_node = int(sequence_list[0]), int(sequence_list[1])
    current_state = start_node
    for length_of_partial_sequence in range(2, len(sequence_list)):
      top_pred = top_preds[i, length_of_partial_sequence-1]
      top_pred_str = tokenizer.decode(top_pred)
      num_total += 1
      next_str = sequence_list[length_of_partial_sequence]
      if top_pred_str in valid_turns[current_state]:
        num_success += 1
      elif top_pred_str == 'end' and current_state == end_node:
        num_success += 1
      if next_str != 'end':
        current_state = node_and_direction_to_neighbor[(current_state, next_str)]
  p = num_success / num_total
  std = np.sqrt(p * (1 - p) / num_total)
  bar.set_description(f"Fraction successful: {p:.3f} ({std:.3f}) {num_success}/{num_total}")

