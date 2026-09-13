from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent
EMBEDDINGS_DIR = ROOT / "embeddings"

# run extract_embeddings.py first to create these files
X = np.load(EMBEDDINGS_DIR / "X.npy")
y = np.load(EMBEDDINGS_DIR / "y.npy")
groups = np.load(EMBEDDINGS_DIR / "groups.npy")

# group by sentence id so a sentence is never in both train and test
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print("train:", X_train.shape, "test:", X_test.shape)
print("test sentences:", sorted(set(groups[test_idx])))
