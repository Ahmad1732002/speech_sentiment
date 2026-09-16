import argparse
from pathlib import Path
from funasr import AutoModel
import numpy as np

ROOT = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description="Extract emotion2vec embeddings for the audio dataset.")
parser.add_argument(
    "--data-dir",
    type=Path,
    default=ROOT / "data" / "neutral_text",
    help="folder containing sad/, happy/, neutral/ subfolders (default: data/neutral_text)",
)
parser.add_argument(
    "--out-dir",
    type=Path,
    help="where to save X.npy/y.npy/groups.npy (default: embeddings/audio/<data dir name>)",
)
args = parser.parse_args()
out_dir = args.out_dir or ROOT / "embeddings" / "audio" / args.data_dir.name

model = AutoModel(model="iic/emotion2vec_plus_base")

def load_data(data_dir):
    emotions = ["sad", "happy", "neutral"]
    audio_files = []
    true_labels = []
    for emotion in emotions:
        files = sorted((Path(data_dir) / emotion).glob("*.wav"))
        audio_files.extend(files)
        true_labels.extend([emotion] * len(files))

    return audio_files, true_labels


audio_files, true_labels = load_data(args.data_dir)
res = model.generate([str(p) for p in audio_files], granularity="utterance", extract_embedding=True)
embeddings = [r["feats"] for r in res]
X = np.stack(embeddings)
y = np.array(true_labels)
# sentence id (s01, sad_21, ...) used to group the train/test split and to pair with text
groups = np.array([p.stem for p in audio_files])
print(X.shape, y.shape, groups.shape)

out_dir.mkdir(parents=True, exist_ok=True)
np.save(out_dir / "X.npy", X)
np.save(out_dir / "y.npy", y)
np.save(out_dir / "groups.npy", groups)
print("saved embeddings to", out_dir)
