import torch
from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2LMHeadModel
from pytorch_lightning import LightningModule
import numpy as np
import pickle

PAD_TOKEN_ID = 0

class SimpleTokenizer:
    def __init__(self, sentences, node_and_direction_to_neighbor, valid_turns):
        words = set()
        for sentence in sentences:
            words.update(sentence.split())
        # Reverse sorted so cardinal directions are one or two digit so saved file is smaller        
        self.word_to_id = {word: idx + 1 for idx, word in enumerate(sorted(words, reverse=True))}
        self.id_to_word = {id: word for word, id in self.word_to_id.items()}
        self.pad_token_id = PAD_TOKEN_ID
        self.word_to_id['<pad>'] = self.pad_token_id
        self.id_to_word[self.pad_token_id] = '<pad>'
        self.node_and_direction_to_neighbor = node_and_direction_to_neighbor
        self.valid_turns = valid_turns
    
    def encode(self, sentence):
        return [self.word_to_id.get(word, self.word_to_id['<pad>']) for word in sentence.split()]
    
    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
          token_ids = token_ids.cpu().numpy()
        if token_ids.ndim == 0:
          token_ids = np.array([token_ids])
        return ' '.join(self.id_to_word[id] for id in token_ids if id != self.pad_token_id)


class TextDataset(Dataset):
    def __init__(self, sequences):
        if isinstance(sequences, str):
            with open(f'{sequences}/sequences.pkl', 'rb') as f:
                self.tokenized_sentences = pickle.load(f)
        else:
            assert isinstance(sequences, list)
            self.tokenized_sentences = sequences
           
    def __len__(self):
        return len(self.tokenized_sentences)
    
    def __getitem__(self, idx):
        token_ids = self.tokenized_sentences[idx]
        attention_mask = [1] * len(token_ids) 
        return {
            'input_ids': token_ids,
            'attention_mask': attention_mask,
            'labels': token_ids 
        }



class GPT2Model(LightningModule):
    def __init__(self, tokenizer, vocab_size=50265, n_embd=128, n_layer=12, n_head=4):
        super().__init__()
        self.save_hyperparameters()
        config = GPT2Config(vocab_size=vocab_size, n_embd=n_embd, n_layer=n_layer, n_head=n_head, pad_token_id=tokenizer.pad_token_id)
        self.model = GPT2LMHeadModel(config)
        self.tokenizer = tokenizer
        self.validation_step_outputs = []
        self.train_step_outputs = []

    def forward(self, input_ids, attention_mask=None, labels=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output.loss

    def training_step(self, batch, batch_idx):
        loss = self(batch['input_ids'], batch['attention_mask'], batch['labels'])
        self.log('train_loss', loss, prog_bar=True, logger=True)
        self.train_step_outputs.append({'train_loss': loss})
        return loss
    
    def on_train_epoch_end(self):
        avg_loss = torch.stack([x['train_loss'] for x in self.train_step_outputs]).mean()
        self.log('train_loss', avg_loss, prog_bar=True, sync_dist=True)
        self.train_step_outputs = []

    def validation_step(self, batch, batch_idx):
        loss = self(batch['input_ids'], batch['attention_mask'], batch['labels'])
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        
        success_nodes = 0
        total_nodes = 0 + 1e-6
        bsz, _ = batch['input_ids'].shape
        with torch.no_grad():
          input_ids = batch['input_ids'].to(self.model.device)
          mask = batch['attention_mask'].to(self.model.device)
          outputs = self.model(input_ids, attention_mask=mask, labels=input_ids)
          logits = outputs.logits
          top_preds = torch.argmax(logits, dim=-1)
        
        for i in range(bsz):
          sequence_str = self.tokenizer.decode(batch['input_ids'][i])
          sequence_list = sequence_str.split(" ")
          start_node, end_node = int(sequence_list[0]), int(sequence_list[1])
          current_state = start_node
          for length_of_partial_sequence in range(2, len(sequence_list)):
            top_pred = top_preds[i, length_of_partial_sequence-1]
            top_pred_str = self.tokenizer.decode(top_pred)
            total_nodes += 1
            next_str = sequence_list[length_of_partial_sequence]
            if top_pred_str in self.tokenizer.valid_turns[current_state]:
              success_nodes += 1
            elif top_pred_str == 'end' and current_state == end_node:
              success_nodes += 1
            if next_str != 'end':
              current_state = self.tokenizer.node_and_direction_to_neighbor[(current_state, next_str)]

        self.validation_step_outputs.append({'val_loss': loss, 'total_nodes': total_nodes, 'success_nodes': success_nodes})
        return loss
    
    def on_validation_epoch_end(self):
        # print("Starting")
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        total_nodes = sum([x['total_nodes'] for x in self.validation_step_outputs])
        success_nodes = sum([x['success_nodes'] for x in self.validation_step_outputs])
        same_preds = sum([x['same_preds'] for x in self.validation_step_outputs])
        
        success_rate = success_nodes / total_nodes
        mean_same_preds = same_preds / total_nodes
        
        self.log('val_loss', avg_loss, prog_bar=True, sync_dist=True)
        self.log('success_rate', success_rate, prog_bar=True, sync_dist=True)
        self.log('mean_same_preds', mean_same_preds, prog_bar=True, sync_dist=True)
        
        self.validation_step_outputs = []


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001)
        return optimizer

    def train_dataloader(self):
        return self.trainer.datamodule.train_dataloader()

    def val_dataloader(self):
        return self.trainer.datamodule.val_dataloader()


def collate_fn(batch):
    max_length = max(len(data['input_ids']) for data in batch)
    padded_inputs = []
    attention_masks = []
    labels = []
    for data in batch:
        padding_length = max_length - len(data['input_ids'])
        padded_input = data['input_ids'] + [PAD_TOKEN_ID] * padding_length
        attention_mask = data['attention_mask'] + [0] * padding_length
        padded_inputs.append(padded_input)
        attention_masks.append(attention_mask)
        labels.append(padded_input) 
    return {
        'input_ids': torch.tensor(padded_inputs),
        'attention_mask': torch.tensor(attention_masks),
        'labels': torch.tensor(labels)
    }
