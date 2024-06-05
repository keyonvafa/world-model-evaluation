from collections import defaultdict
from model import GPT2Model, SimpleTokenizer, TextDataset, collate_fn
import numpy as np
import random
import torch


def create_reverse_maps(valid_turns, node_and_direction_to_neighbor):
  valid_previous_turns = defaultdict(list)
  node_and_previous_direction_to_neighbors = defaultdict(list)
  for node, moves in valid_turns.items():
    for move in moves:
      next_move = node_and_direction_to_neighbor[(node, move)]
      valid_previous_turns[next_move].append(move)
      node_and_previous_direction_to_neighbors[(next_move, move)].append(node)
  return valid_previous_turns, node_and_previous_direction_to_neighbors


def get_all_suffixes_from_state(start_state, end_state, seq_len, valid_turns, node_and_direction_to_neighbor):
  def dfs(current_state, move_list, suffixes):
    if len(move_list) == seq_len:
      suffixes.append(move_list[:])
      return
    if current_state == 'end':
      valid_moves = ['end']
    else:
      valid_moves = valid_turns[current_state]
      if current_state == end_state:
        valid_moves += ['end']
    for next_move in valid_moves:
      next_state = node_and_direction_to_neighbor[(current_state, next_move)]
      if next_state in valid_turns:
        move_list.append(next_move)
        dfs(next_state, move_list, suffixes)
        move_list.pop()
      else:
        return
  suffixes = []
  dfs(start_state, [], suffixes)
  return suffixes


def get_conditional_probability_of_suffixes_after_prefix(prefix, suffixes, model, batch_size=32):
  prefix_len =len(prefix)
  max_suffix_len = max(len(suffix) for suffix in suffixes)
  input_ids = []
  for suffix in suffixes:
    input_ids.append(model.tokenizer.encode(" ".join(prefix + suffix)))
  # Pad input_ids to the same length
  padded_input_ids = []
  attention_masks = []
  for ids in input_ids:
    padding_length = prefix_len + max_suffix_len - len(ids)
    padded_ids = ids + [model.tokenizer.pad_token_id] * padding_length
    attention_mask = [1] * len(ids) + [0] * padding_length
    padded_input_ids.append(padded_ids)
    attention_masks.append(attention_mask)
  padded_input_ids = torch.tensor(padded_input_ids).to(model.device)
  attention_masks = torch.tensor(attention_masks).to(model.device)
  num_batches = (len(padded_input_ids) - 1) // batch_size + 1
  logits_list = []
  for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = start_idx + batch_size
    with torch.no_grad():
      logits = model.model(padded_input_ids[start_idx:end_idx], attention_mask=attention_masks[start_idx:end_idx]).logits
    logits_list.append(logits)
  logits = torch.cat(logits_list, dim=0)
  probs = torch.softmax(logits, -1)
  next_token_probs = torch.gather(probs[:, :-1], dim=-1, index=padded_input_ids[:, 1:].unsqueeze(-1))[:, :, 0]
  # Modify probs after <end>
  next_token_probs = modify_probs_after_eos(padded_input_ids, next_token_probs, model.tokenizer.word_to_id['end'], 1, 0)
  num_suffixes = len(suffixes)
  # Adjust the indexing to account for different suffix lengths
  suffix_probs = []
  for j in range(num_suffixes):
    suffix_len = len(suffixes[j])
    suffix_probs.append(next_token_probs[j, (prefix_len -1):(prefix_len + suffix_len -1)].cpu().numpy())
  return suffix_probs


def get_state_sequence(sequence_str, node_and_direction_to_neighbor):
  split_seq = sequence_str.split(" ")
  start_state = int(split_seq[0])
  moves = split_seq[2:]
  current_state = start_state
  state_seq = [current_state]
  for move in moves:
    if move != 'end':
      current_state = node_and_direction_to_neighbor[(current_state, move)]
      state_seq.append(current_state)
  return state_seq


def get_true_mn_boundary(valid_suffixes1, valid_suffixes2, current_state2, end_state2, valid_turns, node_and_direction_to_neighbor):
  myhill_nerode_set = set()
  set_difference = [x for x in valid_suffixes1 if x not in valid_suffixes2]
  for example in set_difference:
    for i in range(1, len(example) + 1):
      if not is_suffix_valid(example[:i], current_state2, end_state2, valid_turns, node_and_direction_to_neighbor):
        myhill_nerode_set.add(tuple(example[:i]))
        break
  mn_boundary = [list(x) for x in myhill_nerode_set]
  return mn_boundary


def is_suffix_valid(suffix, current_state, end_state, valid_turns, node_and_direction_to_neighbor):
  for turn in suffix:
    if turn == 'end':
      if current_state != end_state:
        return False
    else:
      if turn not in valid_turns[current_state]:
        return False
      current_state = node_and_direction_to_neighbor[(current_state, turn)]
  return True


def is_valid_sequence(sample, valid_turns, node_and_direction_to_neighbor):
  generated_list = sample.split(" ")
  start_node, end_node = int(generated_list[0]), int(generated_list[1])
  directions = generated_list[2:]
  current_state = start_node
  state_seq = [current_state]
  for _, direction in enumerate(directions):
    if direction != 'end':
      if direction in valid_turns[current_state]:
        current_state = node_and_direction_to_neighbor[(current_state, direction)]
        state_seq.append(current_state)
      else:
        return False
    else:
      if current_state == end_node:
        return True
      else:
        return False
  return False


def load_model(data, use_untrained_model=False):
  data_dir = f'data/{data}'
  model_dir = f'ckpts/{data}'

  if data == 'shortest-paths':
    num_layers, n_embd, n_head = 12, 768, 12
  elif data in ['noisy-shortest-paths', 'random-walks']:
    num_layers, n_embd, n_head = 48, 1600, 25
  else:
    raise ValueError(f"Invalid data: {data}")

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  tokenizer = torch.load(f"{data_dir}/tokenizer.pt")

  # Set seed
  torch.manual_seed(42)
  model = GPT2Model(tokenizer, 
                    vocab_size=len(tokenizer.word_to_id),
                    n_embd=n_embd,
                    n_layer=num_layers,
                    n_head=n_head,)

  if not use_untrained_model:
    checkpoint_path = f"{model_dir}/model.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    del checkpoint

  model.to(device)
  model.eval()
  return model


def load_train_data(data, tokenizer, num_samples=None):
  data_dir = f'data/{data}'
  with open(f"{data_dir}/train_sequences.txt", "r") as f:
    train_sequences = f.read().split("\n")

  if num_samples is not None:
    rs = np.random.RandomState(42)
    train_sequences = [train_sequences[i] for i in rs.choice(len(train_sequences), num_samples, replace=False)]

  tokenized_sequences = [tokenizer.encode(sentence) for sentence in train_sequences]
  dataset = TextDataset(tokenized_sequences)
  return dataset


def load_heldout_data(data, tokenizer):
  data_dir = f'data/{data}'
  with open(f"{data_dir}/heldout_sequences.txt", "r") as f:
    heldout_sequences = f.read().split("\n")

  num_samples = 1000
  rs = np.random.RandomState(42)
  heldout_sequences = [heldout_sequences[i] for i in rs.choice(len(heldout_sequences), num_samples, replace=False)]
  tokenized_sequences = [tokenizer.encode(sentence) for sentence in heldout_sequences]
  dataset = TextDataset(tokenized_sequences)
  return dataset


def modify_probs_after_eos(input_ids, next_token_probs, eos_token_id, valid_score, invalid_score):
  """Modify the probabilities after the first <eos> token to be valid_score if the rest of the sequence is <eos>."""
  _, seq_len = input_ids.shape
  eos_mask = input_ids == eos_token_id
  eos_indices = eos_mask.nonzero()
  for batch_idx, eos_idx in eos_indices:
    if eos_idx < seq_len - 1:
      after_eos = input_ids[batch_idx, eos_idx + 1:]
      if (after_eos == eos_token_id).all():
        next_token_probs[batch_idx, eos_idx:] = valid_score
      else:
        next_token_probs[batch_idx, eos_idx:] = invalid_score
  return next_token_probs


def sample_length_k_prefix_from_state(current_state, end_state, k, valid_previous_turns, node_and_previous_direction_to_neighbors):
  # Perform random walk
  state = current_state
  direction_list = []
  for _ in range(k):
    valid_directions = valid_previous_turns[state]
    direction = random.choice(valid_directions)
    state = random.choice(node_and_previous_direction_to_neighbors[(state, direction)])
    direction_list.append(direction)
  direction_list.append(str(end_state))
  direction_list.append(str(state))
  direction_list = direction_list[::-1]
  return direction_list


def sample_model_suffixes_from_prefix(prefix, model, num_suffix_samples, epsilon):
  prefix_ids = torch.tensor([model.tokenizer.encode(" ".join(prefix))]).to(model.device)
  model_suffixes = model.model.generate(
    prefix_ids, max_length=128, num_return_sequences=num_suffix_samples, 
    eos_token_id=model.tokenizer.word_to_id['end'], do_sample=True, temperature=1., epsilon_cutoff=epsilon)
  tokenized_suffixes = [model.tokenizer.decode(output[len(prefix_ids[0]):]).split(" ") for output in model_suffixes]
  return tokenized_suffixes


def weighted_distance(traversal, G):
    total_distance = 0
    for i in range(len(traversal) - 1):
      node1 = traversal[i]
      node2 = traversal[i + 1]
      weight = G[node1][node2][0]['length']
      total_distance += weight
    return total_distance