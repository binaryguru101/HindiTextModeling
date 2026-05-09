import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
from nplm_model import NPLM 
from nplm_dataset import NPLMDataset
from WPtokenizer import WordPieceTokenizer
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

# set seed for diterministic model
def set_seed(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train():
    os.makedirs("model", exist_ok=True)
    os.makedirs("losses", exist_ok=True)
    os.makedirs("plot", exist_ok=True)
    
    # hyperparameters
    CONTEXT_SIZES = [3, 5]
    USE_SECOND_HIDDEN = [False, True]
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 512
    BATCH_SIZE = 128
    EPOCHS = 7
    UNK_PROB = 0.01 # unk noise required
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 2e-4
    STEP_SIZE = 5
    GAMMA = 0.5
    CORPUS_PATH = 'wmt-news-crawl-hi.txt'
    CLIP_NORM = 1.0

    # initialize tokenizer
    tkn = WordPieceTokenizer()
    tkn.loadvocab("vocab_q3.txt")
    unk_id = tkn.vocmap["<UNK>"] # 0
    print(unk_id)
    for context_size in CONTEXT_SIZES:
        for use_second_hidden in USE_SECOND_HIDDEN:
            set_seed(7)
            print(
                f"\n=== RUN: context_size={context_size}, "
                f"use_second_hidden={use_second_hidden} ===\n"
            )
            # load training and validation datasets
            train_set = NPLMDataset(tkn, context_size=context_size, corpus_path=CORPUS_PATH, is_validation=False)
            val_set = NPLMDataset(tkn, context_size=context_size, corpus_path=CORPUS_PATH, is_validation=True)

            train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

            # initialize model 
            model = NPLM(len(tkn.vocmap), EMBEDDING_DIM, context_size, HIDDEN_DIM, use_second_hidden=use_second_hidden)
            criterion = nn.CrossEntropyLoss()

            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) 
            
            scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

            best_val_loss = float('inf')

            train_losses = []
            val_losses = []

            print(f"--- STARTING TRAINING (Vocab: {len(tkn.vocmap)}) ---")
            
            for epoch in range(EPOCHS):
                model.train()
                t_loss = 0
                for contexts, targets in train_loader:
                    # 1% UNK replacement
                    mask = (torch.rand(contexts.shape) < UNK_PROB)
                    contexts[mask] = unk_id
                    
                    optimizer.zero_grad()
                    preds = model(contexts)
                    loss = criterion(preds, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_NORM)
                    optimizer.step()
                    t_loss += loss.item()

                scheduler.step()

                model.eval()
                v_loss = 0
                with torch.no_grad():
                    for vc, vt in val_loader:
                        v_preds = model(vc)
                        v_loss += criterion(v_preds, vt).item()

                avg_t = t_loss / len(train_loader)
                avg_v = v_loss / len(val_loader)

                train_losses.append(avg_t)
                val_losses.append(avg_v)

                
                print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_t:.4f} | Val Loss: {avg_v:.4f} | LR: {optimizer.param_groups[0]['lr']}")

                # early stoping (save best model only) to prevent overfiting
                if avg_v < best_val_loss:
                    best_val_loss = avg_v

                    model_path = f"model/nplm_ctx{context_size}_2hid{int(use_second_hidden)}.pth"

                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'vocab_size': len(tkn.vocmap),
                        'embedding_dim': EMBEDDING_DIM,
                        'context_size': context_size,
                        'hidden_dim': HIDDEN_DIM,
                        'use_second_hidden': use_second_hidden,
                        'vocab': tkn.vocmap_inv,
                        'hyperparameters': {
                            'lr': LEARNING_RATE,
                            'step_size': STEP_SIZE,
                            'gamma': GAMMA,
                            'corpus_path': CORPUS_PATH,
                            'weight_decay': WEIGHT_DECAY,
                            'unk_prob': UNK_PROB,
                            'epochs_trained': epoch + 1,
                            'batch_size': BATCH_SIZE,
                            'clip_norm': CLIP_NORM
                        },
                        'metrics': {
                            'best_val_loss': best_val_loss
                        }
                    }, model_path)

                    print(f"  --> New Best Val Loss! Model Saved to {model_path}")

                    np.save(
                        f"losses/nplm_losses_ctx{context_size}_2hid{int(use_second_hidden)}.npy",
                        {"train": train_losses, "val": val_losses}
                    )

            epochs = list(range(1, EPOCHS + 1))

            plt.plot(epochs, train_losses, label="Train Loss")
            plt.plot(epochs, val_losses, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(
                f"NPLM Loss (context_size={context_size}, "
                f"second_hidden={'ON' if use_second_hidden else 'OFF'})"
            )
            plt.legend()
            plt.grid(True)
            plt.savefig(f"plot/nplm_loss_curve_ctx{context_size}_2hid{int(use_second_hidden)}.png")
            plt.close()

if __name__ == "__main__":
    train()