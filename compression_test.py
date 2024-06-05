from model import SimpleTokenizer, collate_fn
import utils
from torch.utils.data import DataLoader
import pickle
import torch
import numpy as np
import argparse
from tqdm import tqdm
from collections import defaultdict
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='shortest-paths')
parser.add_argument('--use-untrained-model', action='store_true')
parser.add_argument('--num-suffix-samples', type=int, default=30)
parser.add_argument('--epsilon', type=float, default=0.01)
parser.add_argument('--num-trials', type=int, default=100)

args = parser.parse_args()
data = args.data
use_untrained_model = args.use_untrained_model
num_suffix_samples = args.num_suffix_samples
epsilon = args.epsilon
num_trials = args.num_trials

# Load model and tokenizer
model = utils.load_model(data, use_untrained_model)
tokenizer = model.tokenizer
valid_turns = tokenizer.valid_turns
node_and_direction_to_neighbor = tokenizer.node_and_direction_to_neighbor
num_special_tokens = 3 # <start node>, <end node>, ...,  <end>

# Load information about true map
with open(f'data/{data}/all_pairs.pkl', "rb") as f:
  all_pairs = pickle.load(f)

with open(f'data/{data}/shortest_paths.pkl', "rb") as f:
  shortest_paths = pickle.load(f)

all_start_nodes = set([pair[0] for pair in all_pairs])
all_end_nodes = set([pair[1] for pair in all_pairs])
all_nodes = all_start_nodes.union(all_end_nodes)
all_nodes = np.array([node for node in all_nodes if len(valid_turns[node]) > 0])

for node in all_nodes:
  node_and_direction_to_neighbor[(node, 'end')] = 'end'
node_and_direction_to_neighbor[('end', 'end')] = 'end'
node_and_neighbor_to_direction = {(node, neighbor): direction for (node, direction), neighbor in node_and_direction_to_neighbor.items()}

valid_previous_turns, node_and_previous_direction_to_neighbors = utils.create_reverse_maps(valid_turns, node_and_direction_to_neighbor)

def perform_single_compression_test():
  state_ind = np.random.choice(len(all_pairs))
  start_node, end_node = all_pairs[state_ind]
  shortest_path = shortest_paths[(start_node, end_node)]
  # Make sure we can get to the end node in 100 moves.
  prefix_len = np.random.choice(range(1, 100 - shortest_path - num_special_tokens))
  prefix1 = utils.sample_length_k_prefix_from_state(start_node, end_node, prefix_len, valid_previous_turns, node_and_previous_direction_to_neighbors)
  prefix2 = utils.sample_length_k_prefix_from_state(start_node, end_node, prefix_len, valid_previous_turns, node_and_previous_direction_to_neighbors)
  assert prefix1 != prefix2
  suffixes1 = utils.sample_model_suffixes_from_prefix(prefix1, model, num_suffix_samples, epsilon)
  suffix1_probs_prefix2 = utils.get_conditional_probability_of_suffixes_after_prefix(prefix2, suffixes1, model)
  precision = all([all(suffix1_probs_prefix2[i] > epsilon) for i in range(num_suffix_samples)])
  return float(precision), tuple(prefix1), tuple(prefix2), start_node, end_node


state_pair_to_prefixes_to_score = defaultdict(lambda: defaultdict(list))
bar = tqdm(range(num_trials))
for trial in bar:
  try:
    precision, prefix1, prefix2, start_node, end_node = perform_single_compression_test()
    state_pair_to_prefixes_to_score[(start_node, end_node)][(prefix1, prefix2)].append(precision)
    average_precisions = [[np.mean(v) for k, v in inner_dict.items()] for k1, inner_dict in state_pair_to_prefixes_to_score.items()]
    mean_precision = np.mean(average_precisions)
    std = np.std(average_precisions) / np.sqrt(len(average_precisions))
    bar.set_description(f"Mean compression precision: {mean_precision:.3f} ({std:.3f})")
  except:
    # Reasons for failure: sampled prefix gets stuck in sink, prefixes are the same, etc.
    pass

