import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

ROOT = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description="Extract BERT sentence embeddings for the text dataset.")
parser.add_argument(
    "--sentences",
    type=Path,
    default=ROOT / "data" / "emotional_text" / "sentences.csv",
    help="csv with id,label,text columns, or the 'id text' txt file (default: data/emotional_text/sentences.csv)",
)
parser.add_argument(
    "--out-dir",
    type=Path,
    help="where to save X.npy/y.npy/groups.npy (default: embeddings/text/<data dir name>)",
)
parser.add_argument("--model", default="bert-base-uncased", help="huggingface model name")
parser.add_argument("--batch-size", type=int, default=16)
args = parser.parse_args()
out_dir = args.out_dir or ROOT / "embeddings" / "text" / args.sentences.parent.name

tokenizer = BertTokenizer.from_pretrained(args.model)
model = BertModel.from_pretrained(args.model)
model.eval()


def load_data(sentences_file):
    ids, texts, true_labels = [], [], []
    with open(sentences_file, newline="", encoding="utf-8") as f:
        if sentences_file.suffix.lower() == ".csv":
            for row in csv.DictReader(f):
                ids.append(row["id"])
                texts.append(row["text"])
                true_labels.append(row["label"])
        else:
            # "s01  It is almost noon." - these sentences are neutral by construction
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                sentence_id, text = line.split(maxsplit=1)
                ids.append(sentence_id)
                texts.append(text)
                true_labels.append("neutral")

    return ids, texts, true_labels


def embed(texts):
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoded_input)

    # mean pool the token vectors, ignoring padding, to get one vector per sentence
    token_embeddings = output.last_hidden_state
    mask = encoded_input["attention_mask"].unsqueeze(-1).float()
    return ((token_embeddings * mask).sum(dim=1) / mask.sum(dim=1)).numpy()


ids, texts, true_labels = load_data(args.sentences)
embeddings = [embed(texts[i:i + args.batch_size]) for i in range(0, len(texts), args.batch_size)]
X = np.concatenate(embeddings)
y = np.array(true_labels)
# sentence id (sad_01, happy_01, ...) used to group the train/test split
groups = np.array(ids)
print(X.shape, y.shape, groups.shape)

out_dir.mkdir(parents=True, exist_ok=True)
np.save(out_dir / "X.npy", X)
np.save(out_dir / "y.npy", y)
np.save(out_dir / "groups.npy", groups)
print("saved embeddings to", out_dir)
