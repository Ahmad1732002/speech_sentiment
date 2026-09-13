import argparse
from pathlib import Path
from funasr import AutoModel
import numpy as np

ROOT = Path(__file__).resolve().parent
EMBEDDINGS_DIR = ROOT / "embeddings"

parser = argparse.ArgumentParser(description="Extract emotion2vec embeddings for the audio dataset.")
parser.add_argument(
    "--data-dir",
    type=Path,
    default=ROOT / "data",
    help="folder containing sad/, happy/, neutral/ subfolders (default: ./data next to this script)",
)
args = parser.parse_args()

model = AutoModel(model="iic/emotion2vec_plus_base")

def load_data(data_dir):
    emotions = ["sad", "happy", "neutral"]
    samples = []
    for i in range(1, 26):
        if i < 10:
            samples.append(f"s0{i}")
        else:
            samples.append(f"s{i}")

    wav_files = []
    true_labels = []
    for emotion in emotions:
        for sample in samples:
            wav_files.append(str(Path(data_dir) / emotion / f"{sample}.wav"))
            true_labels.append(emotion)

    return wav_files, true_labels


wav_files, true_labels = load_data(args.data_dir)
res = model.generate(wav_files, granularity="utterance", extract_embedding=True)
embeddings = [r["feats"] for r in res]
X = np.stack(embeddings)
y = np.array(true_labels)
# sentence id (s01, s02, ...) used to group the train/test split
groups = np.array([Path(p).stem for p in wav_files])
print(X.shape, y.shape, groups.shape)

EMBEDDINGS_DIR.mkdir(exist_ok=True)
np.save(EMBEDDINGS_DIR / "X.npy", X)
np.save(EMBEDDINGS_DIR / "y.npy", y)
np.save(EMBEDDINGS_DIR / "groups.npy", groups)
print("saved embeddings to", EMBEDDINGS_DIR)
