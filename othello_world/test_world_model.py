from collections import defaultdict
from itertools import product
import numpy as np
import argparse
from tqdm import tqdm
from data import get_othello
from mingpt.dataset import CharDataset
from mingpt.model import GPT, GPTConfig
from data.othello import OthelloBoardState
from mingpt.utils import sample

import utils
import torch
import pdb
import random

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='championship', help='championship, synthetic, or untrained')
parser.add_argument('--num-game-samples', type=int, default=10000)
parser.add_argument('--epsilon', type=float, default=0.01)
parser.add_argument('--num-suffix-samples', type=int, default=30)
parser.add_argument('--num-trials', type=int, default=1000)

args = parser.parse_args()
model_name = args.model
num_game_samples = args.num_game_samples
epsilon = args.epsilon
num_suffix_samples = args.num_suffix_samples
num_trials = args.num_trials

synthetic_othello = get_othello(ood_num=-1, data_root=None, wthor=True)
dataset = CharDataset(synthetic_othello)

mconf = GPTConfig(dataset.vocab_size, dataset.block_size, n_layer=8, n_head=8, n_embd=512)
model = GPT(mconf)

if model_name == 'championship':
  model.load_state_dict(torch.load("ckpts/gpt_championship.ckpt", map_location="cpu"))
elif model_name == 'synthetic':
  model.load_state_dict(torch.load("ckpts/gpt_synthetic.ckpt", map_location="cpu"))
else:
  assert model_name == 'untrained'

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

vocabulary = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29, 30, 31, 32, 33, 34, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]

all_states, game_length_to_state, state_to_prefixes = utils.get_state_samples(
  num_samples=num_game_samples, all_sequences=synthetic_othello.sequences)

states_with_multiple_prefixes = [x for x in all_states if len(state_to_prefixes[x]) > 1]
num_states_with_multiple_prefixes = len(states_with_multiple_prefixes)


def is_suffix_valid(board, suffix):
  for move in suffix:
    if move not in board.get_valid_moves():
      return False
    board.update([move])
  return True


def sample_model_suffixes_from_prefix(prefix, model, num_suffix_samples, epsilon, device):
  x = torch.tensor([dataset.stoi[s] for s in prefix], dtype=torch.long)[None, ...].to(device)
  # use the sample function which we imported
  samples= []
  for _ in range(num_suffix_samples):
    sampled_game = sample(model, x, 60 - len(prefix), temperature=1.0, sample=True, top_k=None, epsilon_cutoff=epsilon)
    sampled_game = sampled_game[0]
    sampled_game = [dataset.itos[int(i)] for i in sampled_game[len(prefix):]]
    samples.append(sampled_game)
  return samples


def get_conditional_probability_of_suffixes(prefix, suffixes, model, batch_size=512):
  len_prefix = len(prefix)
  input_ids = []
  too_long_inds = []
  not_too_long_inds = []
  for suffix in suffixes:
    input_ids.append([dataset.stoi[s] for s in prefix + suffix])
    if len(input_ids[-1]) > dataset.block_size + 1:
      too_long_inds.append(len(input_ids) - 1)
    else:
      not_too_long_inds.append(len(input_ids) - 1)
  # Pad input IDs to the same length
  input_ids = [x for i, x in enumerate(input_ids) if i in not_too_long_inds]
  pad_token = 0  
  padded_input_ids = []
  for id in input_ids:
    padded_input_ids.append(id + [pad_token] * (max([len(x) for x in input_ids]) - len(id)))
  #
  padded_input_ids = torch.tensor(padded_input_ids).to(device)
  num_batches = (len(padded_input_ids) - 1) // batch_size + 1
  logits_list = []
  for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size
    with torch.no_grad():
      logits, _ = model(padded_input_ids[start_idx:end_idx][:, :-1])
    logits_list.append(logits)
  if len(logits_list) > 0:
    logits = torch.cat(logits_list, dim=0)
    probs = torch.softmax(logits, -1)
    next_token_probs = torch.gather(probs, dim=-1, index=padded_input_ids[:, 1:].unsqueeze(-1))[:, :, 0]
  # NOTE: Ignoring <eos>
  suffix_probs = [[] for _ in suffixes]
  for i, j in enumerate(not_too_long_inds):
    suffix_probs[j] = next_token_probs[i, (len_prefix-1):(len_prefix + len(suffixes[j]) - 1)].cpu().numpy()
  for i in too_long_inds:
    suffix_probs[i] = np.zeros(len(suffixes[i]))
  return suffix_probs


def get_mn_boundary_from_samples(samples, suffix_probs, epsilon):
  num_samples = len(suffix_probs)
  above_threshold = [suffix_probs[i] > epsilon for i in range(num_samples)]
  longest_suffixes = []
  for i in range(num_samples):
    for j in range(len(suffix_probs[i])):
      if not above_threshold[i][j]:
        longest_suffixes.append(samples[i][:j+1])
        break
  if len(longest_suffixes) == 0 or all(len(x) == 0 for x in longest_suffixes):
    mn_boundary = []
  else:
    unique_suffixes = set([tuple(x) for x in longest_suffixes])
    unique_suffixes = [list(x) for x in unique_suffixes]
    # If a suffix is the beginning subseqeunce of another, remove it
    mn_boundary = []
    for suffix in unique_suffixes:
      for other_suffix in unique_suffixes:
        if suffix != other_suffix:
          if other_suffix[:len(suffix)] == suffix:
            break
      else:
        mn_boundary.append(suffix)
  return mn_boundary 


def perform_single_compression_test():
  state_ind = np.random.choice(num_states_with_multiple_prefixes)
  state = states_with_multiple_prefixes[state_ind]
  num_prefixes = len(state_to_prefixes[state])
  ind1, ind2 = np.random.choice(num_prefixes, size=2, replace=False)
  prefix1 = list(list(state_to_prefixes[state])[ind1])
  prefix2 = list(list(state_to_prefixes[state])[ind2])
  assert prefix1 != prefix2
  suffixes1 = sample_model_suffixes_from_prefix(prefix1, model, num_suffix_samples, epsilon, device)
  suffix1_probs_prefix2 = get_conditional_probability_of_suffixes(prefix2, suffixes1, model)
  precision = all([all(suffix1_probs_prefix2[i] > epsilon) for i in range(num_suffix_samples)])
  return float(precision), tuple(prefix1), tuple(prefix2), state


def get_distinction_precision(prefix1, prefix2):
  suffixes1 = sample_model_suffixes_from_prefix(prefix1, model, num_suffix_samples, epsilon, device)
  suffix1_probs_prefix2 = get_conditional_probability_of_suffixes(prefix2, suffixes1, model)
  mn_boundary_model = get_mn_boundary_from_samples(suffixes1, suffix1_probs_prefix2, epsilon)
  if len(mn_boundary_model) == 0 or all(len(x) == 0 for x in mn_boundary_model):
    precision = 1.
  else:
    intersection = 0
    for suffix in mn_boundary_model:
      board1 = OthelloBoardState()
      board1.update(prefix1)
      board2 = OthelloBoardState()
      board2.update(prefix2)
      if is_suffix_valid(board1, suffix) and not is_suffix_valid(board2, suffix):
        intersection += 1
    precision = intersection / len(mn_boundary_model)
  return precision


def get_distinction_recall(prefix1, prefix2):
  real_samples = []
  real_suffix_probs = []
  for _ in range(num_suffix_samples):
    board1 = OthelloBoardState()
    board1.update(prefix1)
    board2 = OthelloBoardState()
    board2.update(prefix2)
    suffix = utils.sample_game_from_board(board1)
    real_samples.append(suffix)
    sequence_probs = []
    for move in suffix:
      if move not in board2.get_valid_moves():
        sequence_probs.append(0.)
        break
      sequence_probs.append(1 / len(board2.get_valid_moves()))
      board2.update([move])
    real_suffix_probs.append(np.array(sequence_probs))
  mn_boundary_real = get_mn_boundary_from_samples(real_samples, real_suffix_probs, epsilon)
  if len(mn_boundary_real) == 0:
    recall = 1.
  else:
    model_suffix_probs1 = get_conditional_probability_of_suffixes(prefix1, mn_boundary_real, model)
    model_suffix_probs2 = get_conditional_probability_of_suffixes(prefix2, mn_boundary_real, model)
    model_accepts1 = [suffix for k, suffix in enumerate(mn_boundary_real) if all(model_suffix_probs1[k] > epsilon)]
    model_accepts2 = [suffix for k, suffix in enumerate(mn_boundary_real) if all(model_suffix_probs2[k] > epsilon)]
    model_accepts1 = set([tuple(x) for x in model_accepts1])
    model_accepts2 = set([tuple(x) for x in model_accepts2])
    diference = model_accepts1.difference(model_accepts2)
    recall = len(diference) / len(mn_boundary_real)
  return recall


def perform_single_distinction_test():
  # Sample two games with the same length.
  game_length = np.random.choice(np.arange(4, 60))
  num_games = len(game_length_to_state[game_length])
  ind1, ind2 = np.random.choice(num_games, 2, replace=True)
  tuple_state1 = list(game_length_to_state[game_length])[ind1]
  tuple_state2 = list(game_length_to_state[game_length])[ind2]
  num_prefixes1 = len(state_to_prefixes[tuple_state1])
  num_prefixes2 = len(state_to_prefixes[tuple_state2])
  prefix1 = list(list(state_to_prefixes[tuple_state1])[np.random.choice(num_prefixes1)])
  prefix2 = list(list(state_to_prefixes[tuple_state2])[np.random.choice(num_prefixes2)])
  # First, sample from the transformer
  precision = get_distinction_precision(prefix1, prefix2)
  recall = get_distinction_recall(prefix1, prefix2)
  return precision, recall, tuple(prefix1), tuple(prefix2), tuple_state1, tuple_state2



### FOR COMPRESSION TEST ###
state_to_prefixes_to_score = defaultdict(lambda: defaultdict(list))
bar = tqdm(range(num_trials))
for trial in bar:
  try:
    precision, prefix1, prefix2, state = perform_single_compression_test()
    state_to_prefixes_to_score[(state)][(prefix1, prefix2)].append(precision)
    average_precisions = [[np.mean(v) for k, v in inner_dict.items()] for k1, inner_dict in state_to_prefixes_to_score.items()]
    mean_precision = np.mean(average_precisions)
    std = np.std(average_precisions) / np.sqrt(len(average_precisions))
    bar.set_description(f"Mean compression precision for {model_name}: {mean_precision:.3f} ({std:.3f})")
  except:
    pass

### FOR DISTINCTION TEST ###
state_pair_to_prefixes_to_precision = defaultdict(lambda: defaultdict(list))
state_pair_to_prefixes_to_recall = defaultdict(lambda: defaultdict(list))
bar = tqdm(range(num_trials))
for trial in bar:
  try:
    precision, recall, prefix1, prefix2, state1, state2 = perform_single_distinction_test()
    state_pair_to_prefixes_to_precision[(state1, state2)][(prefix1, prefix2)].append(precision)
    state_pair_to_prefixes_to_recall[(state1, state2)][(prefix1, prefix2)].append(recall)
    average_precisions = [[np.mean(v) for k, v in inner_dict.items()] for k1, inner_dict in state_pair_to_prefixes_to_precision.items()]
    average_recalls = [[np.mean(v) for k, v in inner_dict.items()] for k1, inner_dict in state_pair_to_prefixes_to_recall.items()]
    mean_precision = np.mean(average_precisions)
    mean_recall = np.mean(average_recalls)
    std_precision = np.std(average_precisions) / np.sqrt(len(average_precisions))
    std_recall = np.std(average_recalls) / np.sqrt(len(average_recalls))
    bar.set_description(f"Mean distinction precision for {model_name}: {mean_precision:.3f} ({std_precision:.3f}) | Mean distinction recall: {mean_recall:.3f} ({std_recall:.3f})")
  except:
    pass
