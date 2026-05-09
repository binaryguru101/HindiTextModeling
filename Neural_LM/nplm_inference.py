import os
import torch
from nplm_model import NPLM
from WPtokenizer import WordPieceTokenizer
import unicodedata

def undo_preprocess_spacing(text):
    out = []
    i = 0

    while i < len(text):
        ch = text[i]

        if unicodedata.category(ch).startswith("P"):
            if out and out[-1] == " ":
                out.pop()

            out.append(ch)

            if unicodedata.category(ch) == "Pd":
                if i + 1 < len(text) and text[i + 1] == " ":
                    i += 1
        else:
            out.append(ch)

        i += 1

    return "".join(out)


def generate(seed_file, model_file, k):
    os.makedirs("genrated_out", exist_ok=True)
    model_data = torch.load(model_file, map_location="cpu")
    print(f"Generating using model: {model_file}")

    seed_base = os.path.splitext(os.path.basename(seed_file))[0]
    model_base = os.path.splitext(os.path.basename(model_file))[0]
    output_file = f"genrated_out/{seed_base}__{model_base}_generated.txt"

    tkn = WordPieceTokenizer()
    tkn.loadvocab("vocab_q3.txt")

    model = NPLM(
        model_data['vocab_size'],
        model_data['embedding_dim'],
        model_data['context_size'],
        model_data['hidden_dim'],
        use_second_hidden=model_data['use_second_hidden']
    ).to("cpu")

    model.load_state_dict(model_data['model_state_dict'])
    model.eval() # set internal flag taht no traing will be done for this model now

    context_size = model_data['context_size']
    bos_id = tkn.vocmap["<s>"]
    eos_id = tkn.vocmap["</s>"]

    with open(seed_file, "r", encoding="utf-8") as f:
        seeds = [line.strip() for line in f if line.strip()]

    outputs = []

    for seed_text in seeds:
        clean = tkn.preprocess(seed_text)
        ids = tkn.encode(clean)

        if len(ids) < context_size:
            current_context = [bos_id] * (context_size - len(ids)) + ids
        else:
            current_context = ids[-context_size:]

        generated_ids = []
        words_generated = 0

        while words_generated < k:
            input_tensor = torch.tensor([current_context]).to("cpu")

            with torch.no_grad():
                output = model(input_tensor)
                next_id = torch.argmax(output, dim=1).item()

            if next_id == eos_id:
                break
            if next_id == bos_id:
                continue

            token = tkn.vocmap_inv.get(next_id, "<UNK>")

            if not token.startswith("##") and token not in ["<UNK>"]:
                words_generated += 1

            generated_ids.append(next_id)
            current_context = current_context[1:] + [next_id]

        decoded = tkn.decode(ids + generated_ids)
        decoded = undo_preprocess_spacing(decoded)
        outputs.append(decoded)

    with open(output_file, "w", encoding="utf-8") as f:
        for line in outputs:
            f.write(line + "\n")

    print(f"Output written to: {output_file}")


if __name__ == "__main__":
    seed_file = "wmt-news-crawl-hi.txt"
    k = 20

    model_files = [
        "nplm_ctx3_2hid0.pth",
        "nplm_ctx5_2hid0.pth",
        "nplm_ctx3_2hid1.pth",
        "nplm_ctx5_2hid1.pth",
    ]

    for model_file in model_files:
        generate(
            seed_file=seed_file,
            model_file=model_file,
            k=k
        )
