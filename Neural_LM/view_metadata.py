import os
import torch
import numpy as np

def view():
    MODEL_DIR = "model"
    LOSS_DIR = "losses"

    MODEL_FILES = [
        "nplm_ctx3_2hid0.pth",
        "nplm_ctx5_2hid0.pth",
        "nplm_ctx3_2hid1.pth",
        "nplm_ctx5_2hid1.pth",
    ]

    LOSS_FILES = [
        "nplm_losses_ctx3_2hid0.npy",
        "nplm_losses_ctx5_2hid0.npy",
        "nplm_losses_ctx3_2hid1.npy",
        "nplm_losses_ctx5_2hid1.npy",
    ]

    for model_file, loss_file in zip(MODEL_FILES, LOSS_FILES):
        model_path = os.path.join(MODEL_DIR, model_file)
        loss_path = os.path.join(LOSS_DIR, loss_file)

        print("\n" + "=" * 90)
        print(f"MODEL FILE: {model_file}")
        print("=" * 90)

        if not os.path.exists(model_path):
            print(f"[ERROR] Missing model file: {model_path}")
            continue

        ckpt = torch.load(model_path, map_location="cpu")

        print("\n[MODEL ARCHITECTURE]")
        print(f"  vocab_size        : {ckpt['vocab_size']}")
        print(f"  embedding_dim     : {ckpt['embedding_dim']}")
        print(f"  context_size      : {ckpt['context_size']}")
        print(f"  hidden_dim        : {ckpt['hidden_dim']}")
        print(f"  use_second_hidden : {ckpt['use_second_hidden']}")

        print("\n[HYPERPARAMETERS]")
        for k, v in ckpt["hyperparameters"].items():
            print(f"  {k:<15}: {v}")

        print("\n[TRAINING CONFIG]")
        print(f"  step_size   : {ckpt['hyperparameters']['step_size']}")
        print(f"  gamma       : {ckpt['hyperparameters']['gamma']}")
        print(f"  corpus_path : {ckpt['hyperparameters']['corpus_path']}")

        print("\n[METRICS]")
        for k, v in ckpt["metrics"].items():
            print(f"  {k:<20}: {v}")

        print("\n[LOSS HISTORY]")
        if not os.path.exists(loss_path):
            print(f"Loss file not found: {loss_path}")
        else:
            losses = np.load(loss_path, allow_pickle=True).item()
            train_losses = losses["train"]
            val_losses = losses["val"]

            print(f"  Total epochs logged : {len(train_losses)}")
            print("  Train losses per epoch:")
            for i, l in enumerate(train_losses, 1):
                print(f"    Epoch {i:<2}: {l:.6f}")

            print("  Validation losses per epoch:")
            for i, l in enumerate(val_losses, 1):
                print(f"    Epoch {i:<2}: {l:.6f}")

        print("\n[MODEL STATE DICT SUMMARY]")
        state_dict = ckpt["model_state_dict"]
        print(f"  Number of tensors: {len(state_dict)}")
        for name, tensor in state_dict.items():
            print(f"  {name:<40} {tuple(tensor.shape)}")

    print("\nFull metadata + loss inspection complete.\n")

if __name__ == "__main__":
    view()