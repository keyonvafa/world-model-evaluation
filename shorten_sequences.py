import os
import pdb
import numpy as np
import pickle

# data = 'shortest-paths'
# data = 'noisy-shortest-paths'
data = 'random-walks'

valid = True

# For random-walks, data is already constructed to have a max length. 
max_len = 100 if data != 'random-walks' else np.inf

if valid:
  data_dir = f"data/{data}/valid_sequences_full_length.txt"
  save_dir = f"data/{data}/heldout_sequences.txt"
else:
  data_dir = f"data/{data}/train_sequences_full_length.txt"
  save_dir = f"data/{data}/train_sequences.txt"

with open(data_dir, "r") as f:
  sequences = f.read().split("\n")

# pdb.set_trace()
# # Code to make sure no leakage.
# valid_sequence_pairs = set([tuple(sequence.split(" ")[:2]) for sequence in sequences])
# with open(f"{data_dir}/train_sequences.txt", "r") as f:
#   train_sequences = f.read().split("\n")
# train_sequence_pairs = set([tuple(sequence.split(" ")[:2]) for sequence in train_sequences])
# assert len(train_sequence_pairs.intersection(valid_sequence_pairs)) == 0

for i in range(len(sequences)):
  sequences[i] += " end"

num_tokens_valid = [len(sequence.split(" ")) for sequence in sequences]

sequences = [sequences[i] for i in range(len(sequences)) if num_tokens_valid[i] < max_len]

with open(save_dir, "w") as f:
  f.write("\n".join(sequences))

print(f"Saved to {save_dir}")