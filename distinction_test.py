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
parser.add_argument('--max-suffix-length', type=int, default=5)
parser.add_argument('--num-suffix-samples', type=int, default=5)
parser.add_argument('--epsilon', type=float, default=0.01)
parser.add_argument('--num-trials', type=int, default=100)

args = parser.parse_args()
data = args.data
use_untrained_model = args.use_untrained_model
max_suffix_length = args.max_suffix_length
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


def get_distinction_precision(prefix1, prefix2, start_node1, end_node1, start_node2, end_node2):
  suffixes1 = utils.sample_model_suffixes_from_prefix(prefix1, model, num_suffix_samples, epsilon)
  suffix1_probs_prefix2 = utils.get_conditional_probability_of_suffixes_after_prefix(prefix2, suffixes1, model)
  above_threshold = [suffix1_probs_prefix2[i] > epsilon for i in range(num_suffix_samples)]
  # Get shortest differentiating suffixes for model's MN boundary.
  shortest_differentiating_suffixes = []
  for i in range(num_suffix_samples):
    for j in range(len(suffix1_probs_prefix2[i])):
      if not above_threshold[i][j]:
        shortest_differentiating_suffixes.append(suffixes1[i][:j+1])
        break
  mn_boundary_model = set([tuple(x) for x in shortest_differentiating_suffixes])
  mn_boundary_model = [list(x) for x in mn_boundary_model]
  # If model's MN boundary is empty, precision is 1. 
  if all(len(x) == 0 for x in mn_boundary_model):
    precision = 1.
  else:
    intersection = 0
    for suffix in mn_boundary_model:
      if (utils.is_suffix_valid(suffix, start_node1, end_node1, valid_turns, node_and_direction_to_neighbor) 
          and not utils.is_suffix_valid(suffix, start_node2, end_node2, valid_turns, node_and_direction_to_neighbor)):
        intersection += 1
    precision = intersection / len(mn_boundary_model)
  return precision


def get_distinction_recall(prefix1, prefix2, start_node1, end_node1, start_node2, end_node2):
  valid_suffixes1 = utils.get_all_suffixes_from_state(start_node1, end_node1, max_suffix_length, valid_turns, node_and_direction_to_neighbor)
  valid_suffixes2 = utils.get_all_suffixes_from_state(start_node2, end_node2, max_suffix_length, valid_turns, node_and_direction_to_neighbor)
  mn_boundary_world = utils.get_true_mn_boundary(valid_suffixes1, valid_suffixes2, start_node2, end_node2, valid_turns, node_and_direction_to_neighbor)
  if len(mn_boundary_world) == 0:
    recall = 1.
  else:
    model_suffix_probs1 = utils.get_conditional_probability_of_suffixes_after_prefix(prefix1, mn_boundary_world, model)
    model_suffix_probs2 = utils.get_conditional_probability_of_suffixes_after_prefix(prefix2, mn_boundary_world, model)
    model_accepts1 = set([tuple(suffix) for k, suffix in enumerate(mn_boundary_world) if all(model_suffix_probs1[k] > epsilon)])
    model_accepts2 = set([tuple(suffix) for k, suffix in enumerate(mn_boundary_world) if all(model_suffix_probs2[k] > epsilon)])
    model_difference = model_accepts1.difference(model_accepts2)
    recall = len(model_difference) / len(mn_boundary_world)
  return recall


def perform_single_distinction_test():
  state_inds = np.random.choice(len(all_pairs), 2, replace=False)
  (start_node1, end_node1), (start_node2, end_node2) = all_pairs[state_inds[0]], all_pairs[state_inds[1]]
  shortest_path = shortest_paths[(start_node1, end_node1)]
  # Make sure we can get to the end node in 100 moves.
  prefix_len = np.random.choice(range(1, 100 - shortest_path - num_special_tokens))
  prefix1 = utils.sample_length_k_prefix_from_state(start_node1, end_node1, prefix_len, valid_previous_turns, node_and_previous_direction_to_neighbors)
  prefix2 = utils.sample_length_k_prefix_from_state(start_node2, end_node2, prefix_len, valid_previous_turns, node_and_previous_direction_to_neighbors)
  precision = get_distinction_precision(prefix1, prefix2, start_node1, end_node1, start_node2, end_node2)
  recall = get_distinction_recall(prefix1, prefix2, start_node1, end_node1, start_node2, end_node2)
  return precision, recall, tuple(prefix1), tuple(prefix2), start_node1, end_node1, start_node2, end_node2

state_pair_to_prefixes_to_precision = defaultdict(lambda: defaultdict(list))
state_pair_to_prefixes_to_recall = defaultdict(lambda: defaultdict(list))
bar = tqdm(range(num_trials))
for trial in bar:
  try:
    precision, recall, prefix1, prefix2, start_node1, end_node1, start_node2, end_node2 = perform_single_distinction_test()
    state_pair_to_prefixes_to_precision[(start_node1, end_node1, start_node2, end_node2)][(prefix1, prefix2)].append(precision)
    state_pair_to_prefixes_to_recall[(start_node1, end_node1, start_node2, end_node2)][(prefix1, prefix2)].append(recall)
    average_precisions = [[np.mean(v) for k, v in inner_dict.items()] for k1, inner_dict in state_pair_to_prefixes_to_precision.items()]
    average_recalls = [[np.mean(v) for k, v in inner_dict.items()] for k1, inner_dict in state_pair_to_prefixes_to_recall.items()]
    mean_precision = np.mean(average_precisions)
    mean_recall = np.mean(average_recalls)
    std_precision = np.std(average_precisions) / np.sqrt(len(average_precisions))
    std_recall = np.std(average_recalls) / np.sqrt(len(average_recalls))
    bar.set_description(f"Mean distinction precision: {mean_precision:.3f} ({std_precision:.3f}) | Mean distinction recall: {mean_recall:.3f} ({std_recall:.3f})")
  except:
    # Reasons for failure: sampled prefix gets stuck in sink, etc.
    pass

