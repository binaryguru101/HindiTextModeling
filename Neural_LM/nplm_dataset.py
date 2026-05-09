import torch
from torch.utils.data import Dataset
from WPtokenizer import WordPieceTokenizer

class NPLMDataset(Dataset):
    def __init__(self, tokenizer, context_size=3, corpus_path='wmt-news-crawl-hi.txt', is_validation=False, split_ratio=0.9, debug=False):
        self.tokenizer = tokenizer
        self.context_size = context_size

        self.data = []
        
        # load corpus data
        with open(corpus_path, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
        
        total_lines = len(all_lines)
        split_idx = int(total_lines * split_ratio)
        
        # split
        if is_validation:
            lines = all_lines[split_idx:]
            set_name = "VALIDATION"
            other_set_name = "TRAINING"
        else:
            lines = all_lines[:split_idx]
            set_name = "TRAINING"
            other_set_name = "VALIDATION"

        if debug:
            print(f"\n{'='*70}")
            print(f"DEBUG: {set_name} DATASET ")
            print(f"{'='*70}")
            print(f"Total Lines in File:   {total_lines}")
            print(f"Lines used for {set_name}: {len(lines)}")
            print(f"Lines left for {other_set_name}:  {total_lines - len(lines)}")
            print(f"Context Size (n):      {context_size}")



        # process windows
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line: continue
            
            clean = tokenizer.preprocess(line)
            ids = tokenizer.encode(clean)
            
            # <s> for padding
            bos_id = tokenizer.vocmap["<s>"]
            padding = [bos_id] * (context_size - 1)
            full_seq = padding + ids
            
            # debug trace for the first sentence
            if debug:
                if line_num == 0:
                    print(f"\nProcessing First Sentence: \"{line[:60]}...\"")
                    tkns = [tokenizer.vocmap_inv.get(i, "<UNK>") for i in ids]
                    print(f"Tokens: {tkns}")
                    print(f"{'Step':<5} | {'Context':<35} -> Target")
                    print("-" * 60)

            for i in range(len(full_seq) - context_size):
                context = full_seq[i : i + context_size]
                target = full_seq[i + context_size]
                self.data.append((torch.tensor(context), torch.tensor(target)))
                
                # print first 5 windows for debug
                if debug:
                    if line_num == 0 and i < 5:
                        c_tkns = [tokenizer.vocmap_inv.get(c, "<UNK>") for c in context]
                        t_tkn = tokenizer.vocmap_inv.get(target, "<UNK>")
                        print(f"{i+1:<5} | {str(c_tkns):<35} -> {t_tkn}")

        if debug:
            print(f"\nTotal Sliding Windows Generated: {len(self.data)}")
            print(f"{'='*70}\n")

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

if __name__ == "__main__":
    tkn = WordPieceTokenizer()
    tkn.loadvocab("vocab_q3.txt")
    ds = NPLMDataset(tkn, context_size=5, debug=True)
    ds = NPLMDataset(tkn, context_size=5, is_validation=True, debug=True)