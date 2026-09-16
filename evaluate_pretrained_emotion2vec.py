"""Zero-shot baseline: emotion2vec's own classifier head on the fusion test set.

Scored on exactly the rows train_fusion.py tests on - the neutral-wording
recordings of sentences 21-25 and every emotion-bearing recording - so the
numbers line up with the trained classifiers.

emotion2vec predicts nine emotions. The argmax is restricted to sad/happy/neutral
so the model answers the same three-way question the trained classifiers answer.
"""

import argparse
import re
from pathlib import Path

import numpy as np
from funasr import AutoModel

ROOT = Path(__file__).resolve().parent
LABELS = ["sad", "happy", "neutral"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--neutral-dir",
        type=Path,
        default=ROOT / "data" / "neutral_text",
        help="folder with the neutral-wording recordings (default: data/neutral_text)",
    )
    parser.add_argument(
        "--emotional-dir",
        type=Path,
        default=ROOT / "data" / "emotional_text",
        help="folder with the emotion-bearing recordings (default: data/emotional_text)",
    )
    parser.add_argument(
        "--train-until",
        type=int,
        default=20,
        help="neutral-wording sentences above this number are the test half (default: 20)",
    )
    parser.add_argument("--verbose", action="store_true", help="also print every sample")
    return parser.parse_args()


def load_test(data_dir, above=0):
    files, labels = [], []
    for emotion in LABELS:
        for path in sorted((data_dir / emotion).glob("*.wav")):
            if int(re.search(r"(\d+)$", path.stem).group(1)) > above:
                files.append(path)
                labels.append(emotion)

    return files, labels


def predict(result):
    """emotion2vec returns nine scored labels such as '生气/angry'."""
    labels = np.array([label.split("/")[-1] for label in result["labels"]])
    scores = np.array(result["scores"])
    # drop angry, disgusted, fearful, other, surprised and unknown
    keep = np.isin(labels, LABELS)
    labels, scores = labels[keep], scores[keep]

    return labels[scores.argmax()]


def cell(y_true, pred, mask=slice(None)):
    hit = y_true[mask] == pred[mask]
    return f"{hit.mean():.3f} ({hit.sum()}/{len(hit)})"


def report(name, y_true, pred, is_neutral):
    print(
        f"{name:<14}overall {cell(y_true, pred)}"
        f"   neutral-text {cell(y_true, pred, is_neutral)}"
        f"   emotional-text {cell(y_true, pred, ~is_neutral)}"
    )


def main():
    args = parse_args()

    neutral_files, neutral_labels = load_test(args.neutral_dir, above=args.train_until)
    emo_files, emo_labels = load_test(args.emotional_dir)
    files = neutral_files + emo_files
    y_true = np.array(neutral_labels + emo_labels)
    is_neutral = np.concatenate(
        [np.ones(len(neutral_labels), bool), np.zeros(len(emo_labels), bool)]
    )

    model = AutoModel(model="iic/emotion2vec_plus_base")
    results = model.generate([str(path) for path in files], granularity="utterance")

    pred = np.array([predict(r) for r in results])
    report("emotion2vec", y_true, pred, is_neutral)

    if args.verbose:
        print("\nfile                      half       true      predicted")
        for i, path in enumerate(files):
            half = "neutral" if is_neutral[i] else "emotional"
            name = f"{path.parent.name}/{path.name}"
            print(f"{name:<26}{half:<11}{y_true[i]:<10}{pred[i]}")


if __name__ == "__main__":
    main()
