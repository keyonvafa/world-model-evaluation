from model import SimpleTokenizer, collate_fn
from utils import load_model, is_valid_sequence
from torch.utils.data import DataLoader
import torch
import numpy as np
import argparse
from tqdm import tqdm
import pickle
import osmnx as ox
import networkx as nx
import pdb
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='shortest-paths')
args = parser.parse_args()
data = args.data

model = load_model(data, use_untrained_model=False)
tokenizer = model.tokenizer
valid_turns = tokenizer.valid_turns
node_and_direction_to_neighbor = tokenizer.node_and_direction_to_neighbor
eos_token_id = tokenizer.word_to_id['end']


# These are pairs that are (a) seen in the training data and (b) have 
# legal traversals up to the max length of the training data
with open(f'data/{data}/all_pairs.pkl', "rb") as f:
  all_pairs = pickle.load(f)

all_pairs = np.array(all_pairs)
device = model.device

sample_pairs = all_pairs

num_samples = 50 
batch_size = 128
samples = []
num_successful = 0
total_nodes = 0
bar = tqdm(range(num_samples))
for _ in bar:
  pairs = sample_pairs[np.random.choice(len(sample_pairs), size=batch_size)]
  prefix = torch.tensor(
    [tokenizer.encode(" ".join([str(x) for x in list(pair)])) for pair in pairs]).to(device)
  generated_ids = model.model.generate(
     prefix, 
     max_length=128, 
     num_return_sequences=1, 
     eos_token_id=eos_token_id,
     temperature=1.0, 
     do_sample=True)
  batch_samples = [tokenizer.decode(generated_id) for generated_id in generated_ids]
  samples.extend(batch_samples)
  for sample in batch_samples:
    total_nodes += 1
    if is_valid_sequence(sample, valid_turns, node_and_direction_to_neighbor):
      num_successful += 1
    else:
      start, end = sample.split(" ")[:2]
  bar.set_description(f"Fraction successful: {num_successful/total_nodes:.2f} ({num_successful}/{total_nodes})")


# Save samples to `samples/{data}/`
os.makedirs(f"samples/{data}", exist_ok=True)
with open(f"samples/{data}/samples.txt", "w") as f:
  for sample in samples:
    f.write(sample)
    f.write("\n")

