import torch
import os
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.accelerators import find_usable_cuda_devices
from pytorch_lightning.loggers import WandbLogger
from model import SimpleTokenizer, TextDataset, GPT2Model, collate_fn
import wandb
import numpy as np
import pickle
import argparse

   
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', type=int, default=12,
                        help='Number of transformer layers')
    parser.add_argument('--n_embd', type=int, default=768,
                        help='Embedding dimension')
    parser.add_argument('--n_head', type=int, default=12,
                        help='Number of attention heads')
    parser.add_argument('--batch_size_per_gpu', type=int, default=6,
                        help='Batch size per GPU')
    parser.add_argument('--eval_every', type=int, default=5000,
                        help='Evaluation frequency in steps')
    parser.add_argument('--data', type=str, default='shortest-paths',
                        help='Dataset name (one of "shortest-paths", '
                             '"random-walks", "noisy-graphs")')
    parser.add_argument('--model_name', type=str, default='tmp',
                        help='Name of the model (for saving and logging)')
    parser.add_argument('--max_epochs', type=int, default=25,
                        help='Maximum number of epochs')
    parser.add_argument('--use_wandb', type=bool, default=False,
                        help='Whether to use Weights & Biases logging')
    return parser.parse_args()


class DataModule(LightningDataModule):
    def __init__(self, model_dir, batch_size, data):
        super().__init__()
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.num_shards = len(find_usable_cuda_devices())
        self.data_dir = f'data/{data}'
        self.collate_fn = collate_fn

    def prepare_data(self, stage=None):
        shard_dir = os.path.join(self.data_dir, f'{self.num_shards}-shards')
        if os.path.exists(shard_dir):
            print("Loading existing tokenizer...")
            self.tokenizer = torch.load(f"{self.data_dir}/tokenizer.pt")
            print("...done!")
        else:
            print("Loading datasets...")
            with open(f"{self.data_dir}/train_sequences.txt", "r") as f:
                train_sequences = f.read().split("\n")
            with open(f"{self.data_dir}/heldout_sequences.txt", "r") as f:
                heldout_sequences = f.read().split("\n")
            with open(f"{self.data_dir}/tokenizer.pt", "rb") as f:
                tokenizer = torch.load(f"{self.data_dir}/tokenizer.pt")

            # Validate on 1000 heldout sequences during training
            heldout_subsample_size = 1000
            rs = np.random.RandomState(42)
            heldout_sequences = [heldout_sequences[i] for i in rs.choice(
                len(heldout_sequences), heldout_subsample_size, replace=False)]
            print("...done!")

            print("Tokenizing sequences...")
            self.tokenizer = tokenizer
            tokenized_train = [tokenizer.encode(sentence) for sentence in train_sequences]
            tokenized_valid = [tokenizer.encode(sentence) for sentence in heldout_sequences]
            print("...done!")
            
            # Create shards to speed up multi-GPU training
            print("Creating shards...")
            data_size_per_shard = len(tokenized_train) // self.num_shards
            # divide train_sequences into self.num_shards shards
            for shard in range(self.num_shards):
                print(f"  working on shard {shard}...")
                shard_seqs = [tokenized_train[i] for i in range(
                    shard * data_size_per_shard, (shard + 1) * data_size_per_shard)]
                os.makedirs(f'{shard_dir}/{shard}/', exist_ok=True)
                with open(f"{shard_dir}/{shard}/sequences.pkl", 'wb') as file:
                    pickle.dump(shard_seqs, file)
            # Save valid
            os.makedirs(f'{shard_dir}/valid/', exist_ok=True)
            with open(f"{shard_dir}/valid/sequences.pkl", 'wb') as file:
                pickle.dump(tokenized_valid, file)

            print("...done!")

    def setup(self, stage=None):
        pass 

    def train_dataloader(self):
        shard = self.trainer.global_rank
        shard_dir = os.path.join(self.data_dir, f'{self.num_shards}-shards/{shard}')
        ds = TextDataset(shard_dir)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=4, 
                          collate_fn=self.collate_fn)

    def val_dataloader(self):
        shard_dir = os.path.join(self.data_dir, f'{self.num_shards}-shards/valid')
        ds = TextDataset(shard_dir)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=4, 
                          collate_fn=self.collate_fn)


def main():
    args = get_args()
    torch.set_float32_matmul_precision('medium')
    num_gpus = find_usable_cuda_devices()

    batch_size = args.batch_size_per_gpu * len(num_gpus)

    if args.use_wandb:
        wandb_logger = WandbLogger(log_model=None, project='world-model-taxis', name=args.model_name)

    model_dir = f'ckpts/{args.model_name}'
    last_checkpoint = f"{model_dir}/last.ckpt"
    resume_checkpoint = last_checkpoint if os.path.exists(last_checkpoint) else None

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        filename="{epoch}-{step}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
        accelerator='gpu',
        precision="16-mixed",
        devices=num_gpus,
        logger=wandb_logger if args.use_wandb else None,
        val_check_interval=args.eval_every,
        use_distributed_sampler=False,  
    )

    # Instantiate model and datamodule
    data_module = DataModule(
        model_dir, 
        batch_size=batch_size, 
        data=args.data,) 
    data_module.prepare_data()
    model = GPT2Model(data_module.tokenizer, 
                      vocab_size=len(data_module.tokenizer.word_to_id),
                      n_embd=args.n_embd,
                      n_layer=args.num_layers,
                      n_head=args.n_head,
    )
    trainer.fit(model, data_module, ckpt_path=resume_checkpoint)


if __name__ == "__main__":
    main()
