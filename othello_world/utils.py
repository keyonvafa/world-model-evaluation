import numpy as np
from tqdm import tqdm
from collections import defaultdict
from data.othello import OthelloBoardState



def sample_game_from_board(board):
  seq = []
  while len(board.get_valid_moves()) != 0:
    valid_moves = board.get_valid_moves()
    move = np.random.choice(valid_moves)
    board.update([move])
    seq.append(move)
  return seq


def get_state_samples(num_samples, all_sequences):
  rs = np.random.RandomState(0)
  sample_inds= rs.choice(len(all_sequences), num_samples, replace=False)
  smaple_sequences = [all_sequences[i] for i in sample_inds]
  bar = tqdm(smaple_sequences)
  state_to_prefixes = defaultdict(set)
  for whole_game in bar:
    for length_of_partial_game in range(1, len(whole_game)):
      prefix = whole_game[:length_of_partial_game]
      state = OthelloBoardState()
      state.update(prefix, prt=False)
      tuple_state = tuple(state.state.flatten())
      state_to_prefixes[tuple_state].add(tuple(prefix))

  all_states = list(state_to_prefixes.keys())
  game_length_to_state = defaultdict(set)
  for state, prefixes in state_to_prefixes.items():
    game_length_to_state[len(list(prefixes)[0])].add(state)
  return all_states, game_length_to_state, state_to_prefixes

