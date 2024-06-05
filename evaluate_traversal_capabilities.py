from model import SimpleTokenizer, collate_fn
from utils import load_model, is_valid_sequence, load_heldout_data, get_state_sequence, weighted_distance
from torch.utils.data import DataLoader
import torch
import numpy as np
import argparse
from tqdm import tqdm
import osmnx as ox
import pdb
from collections import defaultdict
from datetime import datetime
historical_date = datetime(2024, 5, 5, 0, 0, 0)
ox.settings.overpass_settings = f'[out:json][timeout:180][date:"{historical_date.isoformat()}Z"]'

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='shortest-paths')
parser.add_argument('--use-untrained-model', action='store_true')
args = parser.parse_args()
data = args.data
use_untrained_model = args.use_untrained_model

model = load_model(data, use_untrained_model=use_untrained_model)
tokenizer = model.tokenizer
valid_turns = tokenizer.valid_turns
node_and_direction_to_neighbor = tokenizer.node_and_direction_to_neighbor
eos_token_id = tokenizer.word_to_id['end']

dataset = load_heldout_data(data, tokenizer)

batch_size = 128
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

samples = []
num_successful = 0
total_nodes = 0
bar = tqdm(dataloader)
for batch in bar:
  with torch.no_grad():
    input_ids = batch['input_ids'].to(model.device)
    prefix = input_ids[:, :2]
    generated_ids = model.model.generate(
       prefix, 
       max_length=128, 
       num_return_sequences=1, 
       eos_token_id=eos_token_id,
       do_sample=False)
    batch_samples = [tokenizer.decode(generated_id) for generated_id in generated_ids]
    for sample in batch_samples:
      total_nodes += 1
      if is_valid_sequence(sample, valid_turns, node_and_direction_to_neighbor):
        num_successful += 1
        samples.append(sample)
      else:
        start, end = sample.split(" ")[:2]
  bar.set_description(f"Fraction successful: {num_successful/total_nodes:.2f} ({num_successful}/{total_nodes})")

percent_valid_traversal = num_successful/total_nodes
valid_traversal_std = np.sqrt(percent_valid_traversal * (1-percent_valid_traversal) / total_nodes)

print(f"Results for {data}")
print(f"Percent valid traversals: {percent_valid_traversal:.3f} ({valid_traversal_std:.3f})")

if data in ['shortest-paths', 'noisy-shortest-paths']:
  # See what percent recover shortest- or near-shortest paths.
  # For shortest, the true validation data has the shortest path.
  with open(f"data/{data}/heldout_sequences.txt", "r") as f:
    all_heldout_sequences = f.read().split("\n")

  place_name = "Manhattan, New York City, New York, USA"
  G = ox.graph_from_place(place_name, network_type="drive")

  pair_to_dists = defaultdict(list)
  for sequence in tqdm(all_heldout_sequences):
    pair = int(sequence.split()[0]), int(sequence.split()[1])
    seq = get_state_sequence(sequence, node_and_direction_to_neighbor)
    if not 42438066 in seq:
      dist = weighted_distance(seq, G)
      pair_to_dists[pair].append(dist)

  pair_to_min_dist = {pair: min(dists) for pair, dists in pair_to_dists.items()}

  # Get state sequences from model.
  recovered_dists = []
  true_min_dists = []
  for j, sample in enumerate(samples):
    pair = int(sample.split()[0]), int(sample.split()[1])
    seq = get_state_sequence(sample, node_and_direction_to_neighbor)
    if not 42438066 in seq:
      dist = weighted_distance(seq, G)
      recovered_dists.append(dist)
      true_min_dists.append(pair_to_min_dist[pair])

  recovered_dists = np.array(recovered_dists)
  true_min_dists = np.array(true_min_dists)
  percent_recover_shortest = np.mean(recovered_dists == true_min_dists)
  percent_within_1 = np.mean((recovered_dists / true_min_dists) <= 1.01)
  percent_within_5 = np.mean((recovered_dists / true_min_dists) <= 1.05)
  percent_within_10 = np.mean((recovered_dists / true_min_dists) <= 1.10)
  percent_within_50 = np.mean((recovered_dists / true_min_dists) <= 1.50)

  # Normalize by invalid
  percent_recover_shortest = percent_recover_shortest * percent_valid_traversal
  percent_within_1 = percent_within_1 * percent_valid_traversal
  percent_within_5 = percent_within_5 * percent_valid_traversal
  percent_within_10 = percent_within_10 * percent_valid_traversal
  percent_within_50 = percent_within_50 * percent_valid_traversal

  std_shortest = np.sqrt(percent_recover_shortest * (1-percent_recover_shortest) / total_nodes)
  std_within_1 = np.sqrt(percent_within_1 * (1-percent_within_1) / total_nodes)
  std_within_5 = np.sqrt(percent_within_5 * (1-percent_within_5) / total_nodes)
  std_within_10 = np.sqrt(percent_within_10 * (1-percent_within_10) / total_nodes)
  std_within_50 = np.sqrt(percent_within_50 * (1-percent_within_50) / total_nodes)

  print("------------------------------------------------------------------------")
  print("   NOTE: The percent that recover shortest paths include all sequences in the denominator "
        "not just valid ones recovered by the model. So if all valid traversals by the model "
        "are the shortest ones, it doesn't mean the score will be 100\%.")
  print("------------------------------------------------------------------------")
  print(f"Percent recover shortest: {percent_recover_shortest:.3f} ({std_shortest:.3f})")
  print(f"Percent within 1%: {percent_within_1:.3f} ({std_within_1:.3f})")
  print(f"Percent within 5%: {percent_within_5:.3f} ({std_within_5:.3f})")
  print(f"Percent within 10%: {percent_within_10:.3f} ({std_within_10:.3f})")
  print(f"Percent within 50%: {percent_within_50:.3f} ({std_within_50:.3f})")

