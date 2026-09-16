"""Fuse an audio emotion classifier with a text sentiment classifier.

The two classifiers are trained on datasets that isolate one cue each. The audio
classifier sees the neutral-wording recordings, where emotion lives only in the
voice. The text classifier sees the emotion-bearing sentences, where it lives only
in the wording. At test time the audio classifier answers whenever its top
probability clears a threshold, and the text classifier answers otherwise.

Sentences 1-20 train and 21-25 test, so no sentence appears in both halves, and
C and the threshold are chosen by cross-validation on the training data alone.
"""

import argparse
import re
from pathlib import Path
from typing import NamedTuple

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
EMBEDDINGS = ROOT / "embeddings"
LABELS = ["sad", "happy", "neutral"]
C_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--audio-neutral",
        type=Path,
        default=EMBEDDINGS / "audio" / "neutral_text",
        help="emotion2vec embeddings of the neutral-wording recordings",
    )
    parser.add_argument(
        "--audio-emotional",
        type=Path,
        default=EMBEDDINGS / "audio" / "emotional_text",
        help="emotion2vec embeddings of the emotion-bearing recordings",
    )
    parser.add_argument(
        "--text-emotional",
        type=Path,
        default=EMBEDDINGS / "text" / "emotional_text",
        help="BERT embeddings of the emotion-bearing sentences",
    )
    parser.add_argument(
        "--text-neutral",
        type=Path,
        default=EMBEDDINGS / "text" / "neutral_text",
        help="BERT embeddings of the neutral sentences",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=ROOT / "models",
        help="where to save the trained classifiers (default: models)",
    )
    parser.add_argument(
        "--train-until",
        type=int,
        default=20,
        help="sentence numbers up to this train, the rest test (default: 20)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="folds used to choose C and the threshold (default: 5)",
    )
    parser.add_argument(
        "--audio-c",
        type=float,
        help="C for the audio classifier (default: cross-validated on the training data)",
    )
    parser.add_argument(
        "--text-c",
        type=float,
        help="C for the text classifier (default: cross-validated on the training data)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="trust audio when its top probability is at least this "
        "(default: cross-validated on the training data)",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Platt-scale both sets of probabilities on held-out folds of the training data",
    )
    parser.add_argument(
        "--rule",
        choices=["audio", "both"],
        default="audio",
        help="audio: fall back whenever audio is unsure. both: fall back only if text is surer",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="also print the threshold sweep, the confusion matrix and the per-sample table",
    )
    return parser.parse_args()


class Split(NamedTuple):
    """Training sets for each modality, and a test set aligned row by row."""

    X_audio_train: np.ndarray
    y_audio_train: np.ndarray
    X_text_train: np.ndarray
    y_text_train: np.ndarray
    X_audio_test: np.ndarray
    X_text_test: np.ndarray
    y_test: np.ndarray
    test_ids: np.ndarray
    is_neutral: np.ndarray  # which test rows came from the neutral-wording half


class Prediction(NamedTuple):
    label: np.ndarray
    confidence: np.ndarray


def load(embeddings_dir):
    X = np.load(embeddings_dir / "X.npy")
    y = np.load(embeddings_dir / "y.npy")
    groups = np.load(embeddings_dir / "groups.npy")
    return X, y, groups


def sentence_numbers(ids):
    # s01 -> 1, sad_21 -> 21
    return np.array([int(re.search(r"(\d+)$", sentence_id).group(1)) for sentence_id in ids])


def rows_for(ids, lookup_ids):
    # the neutral sentences map many-to-one: s21 is spoken in all three emotions
    index = {sentence_id: i for i, sentence_id in enumerate(lookup_ids)}
    return [index[sentence_id] for sentence_id in ids]


def build_split(args):
    X_audio_neutral, y_audio_neutral, audio_neutral_ids = load(args.audio_neutral)
    X_audio_emo, y_audio_emo, audio_emo_ids = load(args.audio_emotional)
    X_text_emo, y_text_emo, text_emo_ids = load(args.text_emotional)
    X_text_neutral, _, text_neutral_ids = load(args.text_neutral)

    trains_audio = sentence_numbers(audio_neutral_ids) <= args.train_until
    trains_text = sentence_numbers(text_emo_ids) <= args.train_until

    # test half 1: neutral wording, so only the voice carries the emotion
    neutral_ids = audio_neutral_ids[~trains_audio]
    X_audio_1 = X_audio_neutral[~trains_audio]
    y_1 = y_audio_neutral[~trains_audio]
    X_text_1 = X_text_neutral[rows_for(neutral_ids, text_neutral_ids)]

    # test half 2: emotion-bearing wording, recorded
    X_text_2 = X_text_emo[rows_for(audio_emo_ids, text_emo_ids)]

    return Split(
        X_audio_train=X_audio_neutral[trains_audio],
        y_audio_train=y_audio_neutral[trains_audio],
        X_text_train=X_text_emo[trains_text],
        y_text_train=y_text_emo[trains_text],
        X_audio_test=np.concatenate([X_audio_1, X_audio_emo]),
        X_text_test=np.concatenate([X_text_1, X_text_2]),
        y_test=np.concatenate([y_1, y_audio_emo]),
        test_ids=np.concatenate([neutral_ids, audio_emo_ids]),
        is_neutral=np.concatenate([np.ones(len(y_1), bool), np.zeros(len(y_audio_emo), bool)]),
    )


def select_c(X, y, folds):
    search = GridSearchCV(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        {"logisticregression__C": C_GRID},
        cv=folds,
    )
    search.fit(X, y)
    return search.best_params_["logisticregression__C"]


def fit_classifier(X, y, C, calibrate):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=C))
    if calibrate:
        # the gate only works if the probabilities rank the predictions honestly
        clf = CalibratedClassifierCV(clf, method="sigmoid", cv=5)
    return clf.fit(X, y)


def select_threshold(audio_clf, text_clf, split, folds):
    """Choose the gate from training data only, never from the test set.

    The training audio is the neutral-wording half, where text is at chance, so
    cross-validating the fused accuracy directly would always say "never fall back".
    Each side is estimated on its own training set instead: how often audio is right
    at a given confidence, and how often text is right at all.
    """
    classes = np.unique(split.y_audio_train)
    oof_proba = cross_val_predict(
        audio_clf, split.X_audio_train, split.y_audio_train, cv=folds, method="predict_proba"
    )
    audio_is_right = classes[oof_proba.argmax(axis=1)] == split.y_audio_train
    audio_confidence = oof_proba.max(axis=1)

    text_oof = cross_val_predict(text_clf, split.X_text_train, split.y_text_train, cv=folds)
    text_accuracy = (text_oof == split.y_text_train).mean()

    best_threshold, best_expected = 0.0, -1.0
    for candidate in np.arange(0.35, 1.0, 0.01):
        trusted = audio_confidence >= candidate
        # audio answers the trusted rows, text is expected to get text_accuracy of the rest
        expected = (audio_is_right[trusted].sum() + text_accuracy * (~trusted).sum()) / len(trusted)
        if expected > best_expected:
            best_threshold, best_expected = candidate, expected

    return best_threshold


def predict(clf, X):
    proba = clf.predict_proba(X)
    return Prediction(clf.classes_[proba.argmax(axis=1)], proba.max(axis=1))


def fuse(audio, text, threshold, rule):
    """Take the audio answer where it is confident enough, otherwise the text one."""
    use_audio = audio.confidence >= threshold
    if rule == "both":
        # when audio is unsure, only hand over if text is surer than it
        use_audio |= text.confidence <= audio.confidence
    return use_audio, np.where(use_audio, audio.label, text.label)


def provenance(given):
    return "set" if given is not None else "cv"


def accuracy(y_true, pred, mask=slice(None)):
    return (y_true[mask] == pred[mask]).mean()


def cell(y_true, pred, mask=slice(None)):
    hit = y_true[mask] == pred[mask]
    return f"{hit.mean():.3f} ({hit.sum()}/{len(hit)})"


def report(name, y_true, pred, is_neutral):
    print(
        f"{name:<7}overall {cell(y_true, pred)}"
        f"   neutral-text {cell(y_true, pred, is_neutral)}"
        f"   emotional-text {cell(y_true, pred, ~is_neutral)}"
    )


def print_details(split, audio, text, fused_pred, uses_audio, rule):
    print("\nthreshold  text_used  fused_accuracy")
    for candidate in np.arange(0.4, 1.01, 0.05):
        used, fused = fuse(audio, text, candidate, rule)
        print(f"{candidate:<11.2f}{(~used).sum():<11d}{accuracy(split.y_test, fused):.3f}")

    print()
    print(classification_report(split.y_test, fused_pred, labels=LABELS, zero_division=0))
    print(confusion_matrix(split.y_test, fused_pred, labels=LABELS))

    print("\nid            half       true      audio(conf)     text(conf)      fused")
    for i, sentence_id in enumerate(split.test_ids):
        half = "neutral" if split.is_neutral[i] else "emotional"
        audio_cell = f"{audio.label[i]}({audio.confidence[i]:.2f})"
        text_cell = f"{text.label[i]}({text.confidence[i]:.2f})"
        print(
            f"{sentence_id:<14}{half:<11}{split.y_test[i]:<10}{audio_cell:<16}{text_cell:<16}"
            f"{fused_pred[i]} <- {'audio' if uses_audio[i] else 'text'}"
        )


def main():
    args = parse_args()
    split = build_split(args)

    audio_c = args.audio_c
    if audio_c is None:
        audio_c = select_c(split.X_audio_train, split.y_audio_train, args.folds)
    text_c = args.text_c
    if text_c is None:
        text_c = select_c(split.X_text_train, split.y_text_train, args.folds)

    audio_clf = fit_classifier(split.X_audio_train, split.y_audio_train, audio_c, args.calibrate)
    text_clf = fit_classifier(split.X_text_train, split.y_text_train, text_c, args.calibrate)
    assert (audio_clf.classes_ == text_clf.classes_).all(), "classifiers disagree on class order"

    threshold = args.threshold
    if threshold is None:
        threshold = select_threshold(audio_clf, text_clf, split, args.folds)

    args.models_dir.mkdir(exist_ok=True)
    joblib.dump(audio_clf, args.models_dir / "audio_clf.joblib")
    joblib.dump(text_clf, args.models_dir / "text_clf.joblib")

    audio = predict(audio_clf, split.X_audio_test)
    text = predict(text_clf, split.X_text_test)
    uses_audio, fused_pred = fuse(audio, text, threshold, args.rule)

    print(
        f"audio C={audio_c} ({provenance(args.audio_c)}), "
        f"text C={text_c} ({provenance(args.text_c)}), "
        f"threshold={threshold:.2f} ({provenance(args.threshold)})"
        f"   cv = {args.folds}-fold on the training data"
    )
    report("audio", split.y_test, audio.label, split.is_neutral)
    report("text", split.y_test, text.label, split.is_neutral)
    report("fused", split.y_test, fused_pred, split.is_neutral)
    print(f"text used {(~uses_audio).sum()}/{len(split.y_test)}")

    if args.verbose:
        print_details(split, audio, text, fused_pred, uses_audio, args.rule)


if __name__ == "__main__":
    main()
