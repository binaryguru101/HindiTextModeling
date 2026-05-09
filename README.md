# HindiNLP Toolkit Report

# 1. WordPiece Tokenizer

## Overview

The tokenizer was implemented completely from scratch following the specifications provided in the assignment. The pipeline includes preprocessing, vocabulary generation, encoding, and decoding modules.

---

## Preprocessing

The preprocessing stage uses Python’s `unicodedata` module to normalize all input text using Unicode NFKC normalization. After normalization, punctuation symbols are separated from surrounding characters by whitespace.

This preprocessing ensures consistency across Unicode representations and simplifies downstream tokenization.

---

## Vocabulary Training

The vocabulary generation process follows the WordPiece training algorithm.

### Corpus Construction

The corpus is converted into a frequency table of unique words. For each word:
- The first character is preserved as-is.
- Every subsequent character is prefixed with `##`.

Example:

```text
दिल्ली → द ##ि ##ल ##् ##ल ##ी
```

Each corpus entry is stored as:

```text
(tokens, frequency)
```

---

## Character and Pair Statistics

The following helper functions were implemented:

### `get_char_counts`
Computes the frequency of every token appearing in the corpus.

### `get_pairwise_counts`
Computes frequencies for adjacent token pairs.

### `get_pair_score`
Calculates pair merge scores using:

```text
score(a,b) = occ(a,b) / (occ(a) * occ(b))
```

### `get_highest_likelihood`
Selects the highest-scoring pair.
Lexicographic ordering is used to break ties.

---

## Merge Operation

The merge function combines selected token pairs throughout the corpus while preserving word frequencies.

Examples:

```text
s + ##a → sa
##s + ##a → ##sa
```

The corpus is updated after every merge step until the target vocabulary size is reached.

---

## Training Procedure

The `train_wordpiece` function repeatedly:
1. Computes token statistics
2. Computes pair scores
3. Selects the best merge candidate
4. Merges the selected pair

This process continues until the required vocabulary size is reached.

The implementation also constructs:
- `vocmap` → token-to-ID mapping
- `vocmap_inv` → ID-to-token mapping

---

## Encoding

The encoder implements greedy longest-match-first tokenization.

For each word:
- The longest valid token is searched greedily.
- Unknown words are mapped to `<UNK>`.

Special tokens:
- `<UNK>` = 0
- `<s>` = 1
- `</s>` = 2

BOS and EOS tokens are appended around every sentence.

---

## Decoding

The decoder reconstructs text by:
- Converting token IDs back to text
- Merging tokens beginning with `##`
- Joining remaining tokens with whitespace

---

## Deliverables

- `wordpiece.py`
- Preprocessed corpus
- Vocabulary file
- Tokenized outputs
- Token ID outputs
- Decoded outputs

---

# 2. FastText Word Representations

## Overview

FastText word representations were implemented using character n-grams and a Skip-Gram model with negative sampling.

The implementation follows the ideas introduced in:

> Enriching Word Vectors with Subword Information

---

## Model Components

### 1. WordPieceTokenizer

The tokenizer from Question 1 was reused directly.

---

### 2. NgramGenerator

Responsible for:
- Character n-gram extraction
- Vocabulary construction
- N-gram indexing

The implementation:
- Adds boundary symbols around words
- Extracts character n-grams of lengths:
  - 3
  - 4
  - 5
- Includes the entire word as a special sequence

---

### 3. Trainer

Implements the Skip-Gram with Negative Sampling (SGNS) training pipeline.

Trainable parameters:
- `Z` → n-gram embedding matrix
- `V` → context/output embedding matrix

Implemented features:
- Negative sampling
- Frequent-word subsampling
- Skip-Gram pair generation
- Training loop
- Nearest-neighbor retrieval
- OOV handling

---

## Important Methods

### `extract_ngrams`
Generates all character n-grams and full-word tokens.

### `ngram_vocab`
Builds:
- `ngram2id`
- `id2ngram`

### `word_to_ngram_ids`
Converts a word into its constituent n-gram IDs.

### `write_bags_file`
Writes n-gram bags to file.

### `generate_skipgram_pairs`
Creates training pairs for SGNS.

### `sgns_step`
Performs one optimization step.

### `nearest_neighbors`
Computes cosine similarity between word vectors.

### `important_ngrams`
Computes importance scores for n-grams.

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Embedding Dimension | 40 |
| N-Gram Lengths | 3–5 |
| Training Objective | Skip-Gram |
| Negative Sampling | Enabled |
| Subsampling | Enabled |

---

## Training Analysis

The training loss decreased significantly during early epochs and stabilized after approximately 3.5 epochs, indicating convergence. :contentReference[oaicite:0]{index=0}

---

## OOV Word Experiment

An out-of-vocabulary word:

```text
दिल्लीवालापन
```

was used to demonstrate FastText’s subword capabilities.

Even though the word was unseen during training, the model successfully generated a meaningful vector representation using known subwords and retrieved semantically related neighbors. :contentReference[oaicite:1]{index=1}

---

## Morphological Analysis

Important n-grams were analyzed for both in-vocabulary and out-of-vocabulary words.

The highest-scoring n-grams often corresponded to meaningful morphemes, demonstrating that FastText captures morphological structure effectively. :contentReference[oaicite:2]{index=2}

---

## Deliverables

- `realFastText.ipynb`
- `q2.py`
- `saved_params.pt`
- `ngram_bags.txt`
- Training plots
- OOV analysis outputs

---

# 3. Neural Probabilistic Language Model (NPLM)

## Overview

A feed-forward Neural Probabilistic Language Model inspired by Bengio et al. was implemented for next-token prediction.

The model predicts:

```text
P(w_t | w_{t-n+1}, ..., w_{t-1})
```

using distributed word representations.

---

## Data Pipeline

The pipeline:
- Uses the tokenizer from Q1
- Generates sliding-window context-target pairs
- Uses `<s>` tokens for padding
- Maps unknown tokens to `<UNK>`
- Reserves the last 10% of the corpus for validation

The provided vocabulary file `vocab_q3.txt` was used directly.

---

## Model Architecture

The architecture consists of:

- Word embedding lookup table
- Concatenation of context embeddings
- Hidden feed-forward layers
- Tanh activation
- Output projection layer

---

## Ablation Variants

Two architecture variants:
1. Single hidden layer
2. Two hidden layers

Two context sizes:
1. Context size = 3
2. Context size = 5

Total configurations:
- Context 3 + 1 hidden layer
- Context 5 + 1 hidden layer
- Context 3 + 2 hidden layers
- Context 5 + 2 hidden layers

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Context Sizes | 3, 5 |
| Embedding Dimension | 128 |
| Hidden Layer Size | 512 |
| Batch Size | 128 |
| Epochs | 7 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Weight Decay | 2e-4 |
| Gradient Clipping | 1.0 |
| UNK Replacement Probability | 0.01 |
| Scheduler | StepLR |
| Step Size | 5 |
| Gamma | 0.5 |
| Loss Function | CrossEntropyLoss |

---

## Training Details

Early stopping was implemented.

All four experimental configurations achieved their best validation loss at epoch 6, and therefore epoch 6 checkpoints were saved for final evaluation. :contentReference[oaicite:3]{index=3}

---

## Results

### Model 1 — Context Size 3, Single Hidden Layer

| Dataset | Perplexity | Accuracy (%) |
|---|---|---|
| Training | 97.6402 | 25.96 |
| Validation | 302.6189 | 19.67 |

---

### Model 2 — Context Size 5, Single Hidden Layer

| Dataset | Perplexity | Accuracy (%) |
|---|---|---|
| Training | 81.9550 | 26.84 |
| Validation | 316.5025 | 19.53 |

---

### Model 3 — Context Size 3, Two Hidden Layers

| Dataset | Perplexity | Accuracy (%) |
|---|---|---|
| Training | 72.7672 | 28.13 |
| Validation | 305.3726 | 19.71 |

---

### Model 4 — Context Size 5, Two Hidden Layers

| Dataset | Perplexity | Accuracy (%) |
|---|---|---|
| Training | 67.0886 | 28.40 |
| Validation | 311.3570 | 19.64 |

---

## Observations

Increasing:
- Context size
- Model depth

improved training perplexity and training accuracy.

However, validation performance remained relatively unchanged across experiments, suggesting limited generalization improvements from increased model capacity alone. :contentReference[oaicite:4]{index=4}

---

## Files

- `WPtokenizer.py`
- `main.py`
- `nplm_dataset.py`
- `nplm_model.py`
- `nplm_trainer.py`
- `eval.py`
- `nplm_inference.py`
- `view_metadata.py`

Additional directories:
- `plot/`
- `losses/`
- `model/`
- `generated_out/`

---

# Assumptions

- Users may evaluate on unseen corpora
- Test-mode evaluation computes metrics on the full dataset
- Early stopping is permitted
- Fixed random seeds are permitted

---

# References

1. Unicode Standard Documentation  
2. spaCy Tokenizer API  
3. Bojanowski et al. — *Enriching Word Vectors with Subword Information*  
4. Mikolov et al. — *Distributed Representations of Words and Phrases and their Compositionality*  
5. Bengio et al. — *A Neural Probabilistic Language Model*
