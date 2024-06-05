"""Get all possible pairs of nodes that appear in training and fit in context length."""
from model import SimpleTokenizer, collate_fn
from utils import load_model
from torch.utils.data import DataLoader
import torch
import numpy as np
import argparse
from tqdm import tqdm
import pickle
import osmnx as ox
import networkx as nx
from datetime import datetime
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='shortest-paths')
args = parser.parse_args()
data = args.data

valid_turns = pickle.load(open(f'data/{data}/valid_turns.pkl', 'rb'))
node_and_direction_to_neighbor = pickle.load(open(f'data/{data}/node_and_direction_to_neighbor.pkl', 'rb'))

with open(f"data/{data}/train_sequences.txt", "r") as f:
  train_sequences = f.read().split("\n")

train_nodes = set()
bar = tqdm(train_sequences)
for seq in bar:
  if len(seq.split(" ")) + 3 < 100:
    start, finish = seq.split(" ")[:2]
    train_nodes.update([int(start), int(finish)])

# Only include train_nodes in valid_turns.keys()
train_nodes = train_nodes.intersection(set(valid_turns.keys()))

# train_nodes.difference(set(list(valid_turns.keys())))
# intersections = ox.graph_to_gdfs(G, nodes=True, edges=False)
# intersections[intersections.index == 42438066][['y', 'x']].values

historical_date = datetime(2024, 5, 5, 0, 0, 0)
ox.settings.overpass_settings = f'[out:json][timeout:180][date:"{historical_date.isoformat()}Z"]'

place_name = "Manhattan, New York City, New York, USA"
G = ox.graph_from_place(place_name, network_type="drive")

# Only include train nodes in G.nodes
train_nodes = set(train_nodes).intersection(set(G.nodes))

# Create a custom graph that contains only the nodes that we use.
custom_G = nx.DiGraph()
for node in valid_turns.keys():
    if node in G.nodes:
        custom_G.add_node(node)

for (node, direction), neighbor in node_and_direction_to_neighbor.items():
    if node in custom_G.nodes and neighbor in custom_G.nodes:
        custom_G.add_edge(node, neighbor, length=1)

# Get shortest paths
all_shortest = nx.all_pairs_dijkstra_path(custom_G, weight='length')
all_shortest_dict = dict(all_shortest)

all_pairs = []
for node1 in train_nodes:
  if len(valid_turns[node1]) > 0:
    for node2 in train_nodes:
      if node1 != node2:
        if node2 in all_shortest_dict[node1]:
          seq_list = all_shortest_dict[node1][node2]
          # Make sure sequence is short enough to be considered during training.
          # Add 3 tokens for <node1>, <node2>, <eos>
          if len(all_shortest_dict[node1][node2]) + 3 < 100:
            all_pairs.append((node1, node2))

shortest_paths = {}
for key1, value1 in all_shortest_dict.items():
  for key2, value2 in value1.items():
    shortest_paths[(key1, key2)] = len(value2)

with open(f"data/{data}/all_pairs.pkl", "wb") as f:
  pickle.dump(all_pairs, f)

with open(f"data/{data}/shortest_paths.pkl", "wb") as f:
  pickle.dump(shortest_paths, f)

with open(f"data/{data}/all_nodes.pkl", "wb") as f:
  pickle.dump(list(train_nodes), f)

print(f"Saved to data/{data}/all_pairs.pkl")

