from model import SimpleTokenizer, collate_fn
from utils import get_state_sequence, load_model, load_train_data, load_heldout_data
from torch.utils.data import DataLoader
import torch
import numpy as np
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='shortest-paths')
parser.add_argument('--use-untrained-model', action='store_true')
args = parser.parse_args()
data = args.data
use_untrained_model = args.use_untrained_model


class MultinomialLogisticRegression(nn.Module):
  def __init__(self, input_dim, num_classes):
    super(MultinomialLogisticRegression, self).__init__()
    self.linear = nn.Linear(input_dim, num_classes)

  def forward(self, x):
    out = self.linear(x)
    return out


model = load_model(data, use_untrained_model)
tokenizer = model.tokenizer
valid_turns = tokenizer.valid_turns
node_and_direction_to_neighbor = tokenizer.node_and_direction_to_neighbor

# num_samples = 100000
num_samples = 25000
dataset = load_train_data(data, tokenizer, num_samples=num_samples)
# dataset = load_heldout_data(data, tokenizer)

# Iterate through dataset and get representations.
batch_size = 128
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
representations = []
labels = []
print("Getting representations...")
bar = tqdm(dataloader)
for batch in bar:
  bsz, seq_len = batch['input_ids'].shape
  with torch.no_grad():
    input_ids = batch['input_ids'].to(model.device)
    mask = batch['attention_mask'].to(model.device)
    outputs = model.model(input_ids, attention_mask=mask, labels=input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    hidden_states = torch.stack(hidden_states, dim=1)
  for i in range(bsz):
    sequence_str = tokenizer.decode(batch['input_ids'][i])
    sequence_states = get_state_sequence(sequence_str, node_and_direction_to_neighbor)
    # First state is first token
    labels.append(sequence_states[0])
    representations.append(hidden_states[i, -1, 0, :].cpu().numpy())
    for j in range(1, len(sequence_states)):
      labels.append(sequence_states[j])
      representations.append(hidden_states[i, -1, j+1, :].cpu().numpy())

label_array = np.array(labels)
representations_array = np.array(representations)

# Convert labels to {0, 1, ..., num_classes-1}
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(label_array)

# Split train and test.
train_inds = np.random.choice(len(representations_array), size=int(0.8 * len(representations_array)), replace=False)
test_inds = np.setdiff1d(np.arange(len(representations_array)), train_inds)
X_train, X_test = representations_array[train_inds], representations_array[test_inds]
y_train, y_test = encoded_labels[train_inds], encoded_labels[test_inds]
X_train, X_test, y_train, y_test = torch.tensor(X_train).float(), torch.tensor(X_test).float(), torch.tensor(y_train), torch.tensor(y_test)

# Initialize probe.
input_dim = X_train.shape[1]
num_classes = len(label_encoder.classes_)
device = model.device
net = MultinomialLogisticRegression(input_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

# Train probe.
num_epochs = 100
batch_size = 2048
bar = tqdm(range(num_epochs))
for epoch in bar:
  for i in range(0, len(X_train), batch_size):
    batch_X = X_train[i:i+batch_size].to(device)
    batch_y = y_train[i:i+batch_size].to(device)
    outputs = net(batch_X)
    loss = criterion(outputs, batch_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  if (epoch + 1) % 1 == 0:
    with torch.no_grad():
      X_test = X_test.to(device)
      y_test = y_test.to(device)
      outputs = net(X_test)
      _, predicted = torch.max(outputs.data, 1)
      accuracy = accuracy_score(y_test.cpu().numpy(), predicted.cpu().numpy())
      num_accurate = sum(y_test.cpu().numpy() == predicted.cpu().numpy())
      p = num_accurate / len(y_test)
      std = np.sqrt(p * (1 - p) / len(y_test))
      bar.set_description(f"Probe accuracy: {accuracy:.3f} ({std:.3f})")

