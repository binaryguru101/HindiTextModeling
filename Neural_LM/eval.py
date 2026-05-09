import torch
import math
import torch.nn as nn
from torch.utils.data import DataLoader
from nplm_model import NPLM
from nplm_dataset import NPLMDataset
from WPtokenizer import WordPieceTokenizer


def calculate_metrics(model, loader, criterion):
    model.eval()
    total_nll = 0.0 # -ve log likelihood
    total_samples = 0
    total_correct = 0

    with torch.no_grad():
        for contexts, targets in loader:
            contexts = contexts.to("cpu")
            targets = targets.to("cpu")

            outputs = model(contexts)
            loss = criterion(outputs, targets)

            batch_size = targets.size(0)
            total_nll += loss.item() * batch_size

            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += batch_size

    avg_nll = total_nll / total_samples
    perplexity = math.exp(avg_nll)
    accuracy = total_correct / total_samples

    return perplexity, accuracy


def run_final_eval(FILE_PATH, input_file="wmt-news-crawl-hi.txt", is_test=False):
    print(f"Evaluating model: {FILE_PATH}")

    model_data = torch.load(FILE_PATH, map_location="cpu")

    tkn = WordPieceTokenizer()
    tkn.loadvocab("vocab_q3.txt")

    model = NPLM(
        model_data['vocab_size'],
        model_data['embedding_dim'],
        model_data['context_size'],
        model_data['hidden_dim'],
        use_second_hidden=model_data['use_second_hidden']
    )
    model.load_state_dict(model_data['model_state_dict'])
    model.to("cpu")
    criterion = nn.CrossEntropyLoss()
    batch_size = model_data["hyperparameters"]["batch_size"]

    if is_test==False:
        train_set = NPLMDataset(tkn, model_data['context_size'], corpus_path=input_file, is_validation=False)
        val_set = NPLMDataset(tkn, model_data['context_size'], corpus_path=input_file, is_validation=True)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        train_ppl, train_acc = calculate_metrics(
            model, train_loader, criterion
        )
        val_ppl, val_acc = calculate_metrics(
            model, val_loader, criterion
        )

        print(f"{'Dataset':<15} | {'Perplexity':<12} | {'Accuracy (%)':<12}")
        print("-" * 45)
        print(f"{'Training':<15} | {train_ppl:<12.4f} | {train_acc * 100:<12.2f}")
        print(f"{'Validation':<15} | {val_ppl:<12.4f} | {val_acc * 100:<12.2f}")
    else:
        test_set = NPLMDataset(tkn, model_data['context_size'], corpus_path=input_file, split_ratio=1, is_validation=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        test_ppl, test_acc = calculate_metrics(
            model, test_loader, criterion
        )

        print(f"{'Dataset':<15} | {'Perplexity':<12} | {'Accuracy (%)':<12}")
        print("-" * 45)
        print(f"{'Test':<15} | {test_ppl:<12.4f} | {test_acc * 100:<12.2f}")



if __name__ == "__main__":
    run_final_eval("model/nplm_ctx3_2hid0.pth")
    run_final_eval("model/nplm_ctx5_2hid0.pth")
    run_final_eval("model/nplm_ctx3_2hid1.pth")
    run_final_eval("model/nplm_ctx5_2hid1.pth")
    run_final_eval("model/nplm_ctx3_2hid0.pth", is_test=True)
    run_final_eval("model/nplm_ctx5_2hid0.pth", is_test=True)
    run_final_eval("model/nplm_ctx3_2hid1.pth", is_test=True)
    run_final_eval("model/nplm_ctx5_2hid1.pth", is_test=True)