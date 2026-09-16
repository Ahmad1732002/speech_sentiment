from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


ROOT = Path(__file__).resolve().parent
EMBEDDINGS_DIR = ROOT / "audio_embeddings"

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

logreg = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, C=1.0),
)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print("logistic regression accuracy:", accuracy_score(y_test, y_pred))

mlp = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(64,),
        alpha=1e-2,
        max_iter=2000,
        random_state=42,
    ),
)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
print("MLP accuracy:", accuracy_score(y_test, y_pred))

labels = ["sad", "happy", "neutral"]
print(classification_report(y_test, y_pred, labels=labels))
print(confusion_matrix(y_test, y_pred, labels=labels))
