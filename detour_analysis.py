from model import SimpleTokenizer, collate_fn
from utils import load_model, load_heldout_data, is_valid_sequence
from torch.utils.data import DataLoader
import torch
import numpy as np
import argparse
from tqdm import tqdm
import pdb
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='shortest-paths')
parser.add_argument('--use-untrained-model', action='store_true')
parser.add_argument('--detour-prob', type=float, default=0.01)
parser.add_argument('--num-trials', type=int, default=100)
parser.add_argument('--detour-type', type=str, default='random_valid', help='Options: least_likely, random_valid, second_most_likely')

args = parser.parse_args()
data = args.data
use_untrained_model = args.use_untrained_model
detour_prob = args.detour_prob
num_trials = args.num_trials
detour_type = args.detour_type

model = load_model(data, use_untrained_model)
tokenizer = model.tokenizer
valid_turns = tokenizer.valid_turns
node_and_direction_to_neighbor = tokenizer.node_and_direction_to_neighbor
eos_token_id = tokenizer.word_to_id['end']
num_special_tokens = 3 # <start node>, <end node>, ...,  <end>

dataset = load_heldout_data(data, tokenizer)
heldout_sequences = dataset.tokenized_sentences

with open(f'data/{data}/shortest_paths.pkl', "rb") as f:
  shortest_paths = pickle.load(f)

rs = np.random.RandomState(0)
bar = tqdm(rs.choice(len(heldout_sequences), num_trials, replace=False))
total_nodes = 0
success_nodes = 0
for ind in bar:
  input_ids = torch.tensor(heldout_sequences[ind][:2]).unsqueeze(0).to(model.device)
  origin, destination = tokenizer.decode(input_ids[0]).split(" ")
  origin = int(origin)
  state = int(origin)
  destination = int(destination)
  for i in range(2, 100): 
    turn_options = valid_turns[state]
    turn_states = [node_and_direction_to_neighbor[(state, turn)] for turn in turn_options]
    turn_dists = [shortest_paths[(turn_state,destination)] if (turn_state, destination) in shortest_paths else np.inf for turn_state in turn_states]
    # Only include turns that can get us to the destination in less than 100 - i moves 
    turn_options = [turn_options[turn_ind] for turn_ind in range(len(turn_options)) if turn_dists[turn_ind] < 100 - num_special_tokens - i]
    with torch.no_grad():
      probs = model.model(input_ids).logits.softmax(-1)
    relevant_probs = probs[0, -1, [tokenizer.word_to_id[turn] for turn in turn_options]]
    transformer_pred = torch.argmax(probs[0, -1, :])
    # Insert detour with probability detour_prob
    if np.random.rand() < detour_prob and state != destination and len(turn_options) > 0:
      if detour_type == 'least_likely':
        # Force model to take least likely one.
        # next_token = transformer_pred[None, None]
        next_token = torch.tensor(tokenizer.encode(turn_options[torch.argmin(relevant_probs)])).unsqueeze(0).to(model.device)
      elif detour_type == 'random_valid':
          # Sample a valid one that's different from the transformer pred
        turn_options_except_pred = [turn for turn in turn_options if turn != tokenizer.decode(transformer_pred)]
        if len(turn_options_except_pred) == 0:
          next_token = torch.tensor(tokenizer.encode(turn_options[0])).unsqueeze(0).to(model.device)
        else:
          next_token = torch.tensor(tokenizer.encode(np.random.choice(turn_options_except_pred))).unsqueeze(0).to(model.device)
      elif detour_type == 'second_most_likely':
        # Force model to take highest rated turn option except pred
        turn_options_except_pred = [turn for turn in turn_options if turn != tokenizer.decode(transformer_pred)]
        if len(turn_options_except_pred) == 0:
          next_token = torch.tensor(tokenizer.encode(turn_options[0])).unsqueeze(0).to(model.device)
        else:
          relevant_probs = probs[0, -1, [tokenizer.word_to_id[turn] for turn in turn_options_except_pred]]
          next_token = torch.tensor(tokenizer.encode(turn_options_except_pred[torch.argmax(relevant_probs)])).unsqueeze(0).to(model.device)
      else:
        raise ValueError(f"Invalid detour type: {detour_type}")
    else:
      next_token = transformer_pred[None, None]
    input_ids = torch.cat((input_ids, next_token), dim=-1)
    if next_token == eos_token_id:
      break
    try:
      state = node_and_direction_to_neighbor[(state, tokenizer.decode(next_token[0]))]
    except:
      # Getting here means the model has suggested a token that's not in the valid_turns dict.
      pass
  #
  path = tokenizer.decode(input_ids[0])
  is_valid = is_valid_sequence(path, valid_turns, node_and_direction_to_neighbor)
  total_nodes += 1
  if is_valid:
    success_nodes += 1
  success_rate = success_nodes/total_nodes
  std = np.sqrt(success_rate*(1-success_rate)/total_nodes)
  bar.set_description(f"Fraction successful {data} ({detour_type} detours, p={detour_prob}): {success_rate:.3f} ({std:.3f})")


