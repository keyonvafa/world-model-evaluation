import itertools
import numpy as np
import pdb
import random
import together
from openai import OpenAI
from tqdm import tqdm
import requests
import os

# TODO: Set these to your API keys
# together_api_key = ...
# open_ai_api_key = ...
client = OpenAI(api_key=open_ai_api_key)

num_seats = 3
full_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
assert num_seats < len(full_alphabet)

model_name = "meta-llama/Llama-2-70b-chat-hf" 
# model_name = "meta-llama/Llama-3-8b-chat-hf" 
# model_name = "meta-llama/Llama-3-70b-chat-hf" 
# model_name = "mistralai/Mixtral-8x22B-Instruct-v0.1"
# model_name = 'Qwen/Qwen1.5-72B-Chat'
# model_name = 'Qwen/Qwen1.5-110B-Chat'
# model_name = 'gpt-3.5-turbo'
# model_name = 'gpt-4'


class SeatingDFA:
    def __init__(self):
        self.people = full_alphabet[:num_seats]
        self.initial_state = list(itertools.permutations(self.people))
    #
    def get_all_possible_statements(self):
        statements = []
        for arrangement in self.initial_state:
            for i, person in enumerate(arrangement):
                statements.append(f"{person} is in seat {i+1}")
            for i in range(len(arrangement)):
                for j in range(len(arrangement)):
                    distance = abs(j - i)
                    if distance != 0:
                        statements.append(f"{arrangement[i]} is {distance} away from {arrangement[j]}")
        statements = list(set(statements))
        return statements
    #
    def get_all_possible_length_k_statements(self, k):
        all_statements = self.get_all_possible_statements()
        all_suffixes = list(itertools.product(all_statements, repeat=k))
        return all_suffixes
    #
    def simulate_moves(self, state, moves):
        for move in moves:
            state = self.apply_move(state, move)
            if not state:
                return None
        return state
    #
    def apply_move(self, state, move):
        new_state = []
        for arrangement in state:
            if self.is_valid_move(arrangement, move):
                new_state.append(arrangement)
        # Treat as set
        new_state = set(new_state)
        return new_state
    #
    def is_valid_move(self, arrangement, move):
        if ' is in seat ' in move:
            person, seat = move.split(' is in seat ')
            seat = int(seat) - 1
            return arrangement[seat] == person
        elif ' is ' in move and ' away from ' in move:
            person1, rest = move.split(' is ')
            distance, person2 = rest.split(' away from ')
            distance = int(distance)
            idx1 = arrangement.index(person1)
            idx2 = arrangement.index(person2)
            return abs(idx1 - idx2) == distance
        return False
    # 
    def is_valid_sequence(self, state, sequence):
        for statement in sequence:
            if not any(self.is_valid_move(arrangement, statement) for arrangement in state):
                return False
            state = self.apply_move(state, statement)
        return True
    #
    def get_valid_statements(self, state):
        valid_statements = []
        for arrangement in state:
            for i, person in enumerate(arrangement):
                valid_statements.append(f"{person} is in seat {i+1}")
            for i in range(len(arrangement)):
                for j in range(len(arrangement)):
                    distance = abs(j - i)
                    if distance != 0:
                      valid_statements.append(f"{arrangement[i]} is {distance} away from {arrangement[j]}")
        # Get unique valid statements
        valid_statements = list(set(valid_statements))
        return valid_statements
    #
    def is_valid_sequence(self, state, sequence):
        for statement in sequence:
            if not any(self.is_valid_move(arrangement, statement) for arrangement in state):
                return False
            state = self.apply_move(state, statement)
        return True
    # 
    def get_all_valid_length_k_suffixes(self, state, k):
        valid_statements = self.get_valid_statements(state)
        all_suffixes = list(itertools.product(valid_statements, repeat=k))
        valid_suffixes = [suffix for suffix in all_suffixes if self.is_valid_sequence(state, suffix)]
        return valid_suffixes
    #
    def does_statement_reduce_state_space(self, state, statement):
        new_state = self.apply_move(state, statement)
        return len(new_state) < len(state)
    #
    def simulate_random_moves_from_start(self, k):
        state = self.initial_state
        moves = []
        for _ in range(k):
            valid_statements = self.get_valid_statements(state)
            if not valid_statements:
                break
            move = random.choice(valid_statements)
            moves.append(move)
            state = self.apply_move(state, move)
        return moves, state
    #
    #
    def simulate_random_moves_until_one_state(self):
        state = self.initial_state
        moves = []
        while True:
            valid_statements = self.get_valid_statements(state)
            if not valid_statements:
                break
            move = random.choice(valid_statements)
            moves.append(move)
            state = self.apply_move(state, move)
            if len(state) == 1:
                break
        assert len(list(state)) == 1
        return moves, state
    #
    def sample_prefix_leading_to_state(self, current_state):
      valid_statements = self.get_valid_statements(current_state)
      non_reducing_statements = [statement for statement in valid_statements if not self.does_statement_reduce_state_space(current_state, statement)]
      sampled_statements = []
      sampled_statement = np.random.choice(non_reducing_statements)
      sampled_statements.append(sampled_statement)
      while self.simulate_moves(self.initial_state, sampled_statements) != current_state:
          sampled_statement = np.random.choice(non_reducing_statements)
          sampled_statements.append(sampled_statement)
      return sampled_statements
    #
    def sample_two_prefixes_leading_to_same_state(self, prefix_len):
      moves, current_state = self.simulate_random_moves_from_start(prefix_len)
      prefix1 = self.sample_prefix_leading_to_state(current_state)
      prefix2 = self.sample_prefix_leading_to_state(current_state)
      while prefix1 == prefix2:
        moves, current_state = self.simulate_random_moves_from_start(prefix_len)
        prefix1 = self.sample_prefix_leading_to_state(current_state)
        prefix2 = self.sample_prefix_leading_to_state(current_state)
      return prefix1, prefix2, current_state
    # 
    def sample_two_prefixes_leading_to_different_states(self, prefix_len):
      moves, current_state1 = self.simulate_random_moves_from_start(prefix_len)
      prefix1 = self.sample_prefix_leading_to_state(current_state1)
      moves, current_state2 = self.simulate_random_moves_from_start(prefix_len)
      prefix2 = self.sample_prefix_leading_to_state(current_state2)
      while current_state1 == current_state2:
        moves, current_state1 = self.simulate_random_moves_from_start(prefix_len)
        prefix1 = self.sample_prefix_leading_to_state(current_state1)
        moves, current_state2 = self.simulate_random_moves_from_start(prefix_len)
        prefix2 = self.sample_prefix_leading_to_state(current_state2)
      return prefix1, prefix2, current_state1, current_state2


def get_normal_evaluation_prompt(moves, query):
  prompt = f"There are {num_seats} individuals named {', '.join(full_alphabet[:num_seats-1])}, and {full_alphabet[num_seats-1]}, and there are {num_seats} seats, positioned {1}-{num_seats}. We have the following statements:\n"
  for i, move in enumerate(moves):
    prompt += f"{i+1}. {move}\n"
  prompt += f"Based on this information, where is {query} seated? You can use chain-of-thought reasoning, but make sure your response ends with 'ANSWER: ' followed by a single number between 1 and {num_seats}."
  return prompt


def get_single_query_prompt(moves, suffix):
  prompt = f"There are {num_seats} individuals named {', '.join(full_alphabet[:num_seats-1])}, and {full_alphabet[num_seats-1]}, and there are {num_seats} seats, positioned {1}-{num_seats}. We have the following statements:\n"
  for i, move in enumerate(moves):
    prompt += f"{i+1}. {move}\n"
  prompt += f"\nBased on this information, consider the proposed continuation:\n"
  for i, move in enumerate(suffix):
    prompt += f"{i+1}. {move}\n"
  prompt += "\nIs this a valid continuation? You can use chain-of-thought reasoning, but make sure your response ends with 'ANSWER: ' followed by one of the following statements without quotes: 'yes', 'no'."
  return prompt

def get_myhill_nerode_list(state1, state2, k):
  dfa = SeatingDFA()
  valid_suffixes1 = dfa.get_all_valid_length_k_suffixes(state1, k=k)
  valid_suffixes2 = dfa.get_all_valid_length_k_suffixes(state2, k=k)
  true_set_difference = set(valid_suffixes1).difference(set(valid_suffixes2))
  true_set_difference = list(true_set_difference)
  myhill_nerode_set = set()
  for example in true_set_difference:
    for i in range(1, len(example) + 1):
      if not dfa.is_valid_sequence(state2, example[:i]):
        myhill_nerode_set.add(example[:i])
        break
  myhill_nerode_list = list(myhill_nerode_set)
  return myhill_nerode_list


def query_model(model_name, prompt, max_tokens=20):
  while True:
    try:
      if 'gpt' in model_name:
        output = client.chat.completions.create(
          model=model_name,
          temperature=0.0,
          max_tokens=max_tokens,
          messages=[
            {"role": "user", "content": prompt}
          ]
        )
        full_output = output.choices[0].message.content.strip()
      else:
        endpoint = 'https://api.together.xyz/v1/chat/completions'
        res = requests.post(endpoint, json={
            "model":model_name,
            "max_tokens":max_tokens,
            "temperature": 0.0,
            "top_p": 0.7,
            "top_k": 50,
            "repetition_penalty": 1,
            "stop": [
                "<|eot_id|>"
            ],
            "messages": [
                {
                    "content": prompt,
                    "role": "user"
                }
            ]
        }, headers={
            "Authorization": f"Bearer {together_api_key}",
        })
        full_output = res.json()['choices'][0]['message']['content'].strip()
      break
    except:
      pass
  return full_output


### REGULAR EVALUATION   
dfa = SeatingDFA()
total_nodes = 0
valid_nodes = 0
bar = tqdm(range(60)) 
for _ in bar:
  moves, state = dfa.simulate_random_moves_until_one_state()
  state = list(state)[0]
  query = np.random.choice(state)
  prompt = get_normal_evaluation_prompt(moves, query)
  correct_answer = int([i+1 for i, person in enumerate(state) if person == query][0])
  full_output = query_model(model_name, prompt, max_tokens=500)
  try:
    clean_output = int(full_output.split("ANSWER: ")[-1].strip())
    total_nodes += 1
    if clean_output == correct_answer:
      valid_nodes += 1
    std = ((valid_nodes/total_nodes) * (1 - valid_nodes/total_nodes)) / np.sqrt(total_nodes)
    bar.set_description(f"Accuracy: {valid_nodes/total_nodes:.3f} ({std:.3f})")
  except:
    pass

## COMPRESSION TEST
# Give transformer prefix1, prefix2, see what the difference is
dfa = SeatingDFA()
num_trials = 100
k = 1
num_samples = 5
accepted_responses = ['yes', 'no']
all_length_k_statements = dfa.get_all_possible_length_k_statements(k)
denominator = 0
numerator = 0
bar1 = tqdm(range(num_trials))
for _ in bar1:
  prefix_len = np.random.choice(np.arange(1, 3))
  prefix1, prefix2, state = dfa.sample_two_prefixes_leading_to_same_state(prefix_len=prefix_len)
  all_valid_length_k_statements = dfa.get_all_valid_length_k_suffixes(state, k)
  any_invalid = False
  any_real_response = False
  for _ in range(num_samples):
    # randomly sample a suffix from either the valid set or the invalid set
    if np.random.rand() < 0.5:
      suffix = all_length_k_statements[np.random.choice(len(all_length_k_statements))]
    else:
      suffix = all_valid_length_k_statements[np.random.choice(len(all_valid_length_k_statements))]
    prompt1 = get_single_query_prompt(prefix1, suffix)
    prompt2 = get_single_query_prompt(prefix2, suffix)
    full_output1 = query_model(model_name, prompt1, max_tokens=1000)
    clean_output1 = full_output1.split("ANSWER: ")[-1].strip().replace("\n", "").replace(".", "")
    full_output2 = query_model(model_name, prompt2, max_tokens=1000)
    clean_output2 = full_output2.split("ANSWER: ")[-1].strip().replace("\n", "").replace(".", "")
    if clean_output1 in accepted_responses and clean_output2 in accepted_responses:
      any_real_response = True
      if clean_output1 != clean_output2:
        any_invalid = True
        break
  if any_real_response:
    denominator += 1
    if not any_invalid:
      numerator += 1
  if denominator > 0:
    p = numerator / denominator
    std = np.sqrt(p * (1-p)) / np.sqrt(denominator)
    bar1.set_description(f"Success rate: {numerator/denominator:.3f} ({std:.3f})")


## Distinction test    
dfa = SeatingDFA()
k = 1
accepted_responses = ['yes', 'no']
num_trials = 100
num_samples = 5
recalls = []
bar1 = tqdm(range(num_trials))
for _ in bar1:
  prefix_len = np.random.choice(np.arange(1, 3))
  prefix1, prefix2, state1, state2 = dfa.sample_two_prefixes_leading_to_different_states(prefix_len=prefix_len)
  myhill_nerode_list = get_myhill_nerode_list(state1, state2, k=k)
  if len(myhill_nerode_list) > 0:
    suffixes_to_sample = np.random.choice(len(myhill_nerode_list), size=min(num_samples, len(myhill_nerode_list)), replace=False)
    bar2 = tqdm(range(len(suffixes_to_sample)))
    num_examples_for_state = 0
    num_correct_for_state = 0
    for i in bar2:
      nerode_suffix = myhill_nerode_list[suffixes_to_sample[i]]
      prompt1 = get_single_query_prompt(prefix1, nerode_suffix)
      prompt2 = get_single_query_prompt(prefix2, nerode_suffix)
      full_output1 = query_model(model_name, prompt1, max_tokens=1000)
      clean_output1 = full_output1.split("ANSWER: ")[-1].strip().replace("\n", "").replace(".", "")
      full_output2 = query_model(model_name, prompt2, max_tokens=1000)
      clean_output2 = full_output2.split("ANSWER: ")[-1].strip().replace("\n", "").replace(".", "")
      if clean_output1 in accepted_responses and clean_output2 in accepted_responses:
        num_examples_for_state += 1
        if clean_output1 == 'yes' and clean_output2 == 'no':
          num_correct_for_state += 1
    if num_examples_for_state > 0:
      recall = num_correct_for_state / num_examples_for_state
      bar1.set_description(f"Average recall: {np.mean(recalls):.3f} ({np.std(recalls) / np.sqrt(len(recalls)):.3f})")
      recalls.append(recall)


