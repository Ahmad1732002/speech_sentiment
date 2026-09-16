"""End-to-end demo: one recording in, one emotion out.

                    ┌─ emotion2vec ─> audio embedding ─> audio classifier ─┐
    wav ─> resample ┤                                                      ├─> gate ─> emotion
                    └─ Whisper ─> transcription ─> BERT ─> text classifier ┘

The gate keeps the audio answer when the audio classifier is confident enough,
and falls back to the text answer otherwise.

Both classifiers and the threshold come from models/, written by train_fusion.py,
so this demo runs the settings that were cross-validated there - nothing is
hardcoded here.

    python demo.py                  launch the web UI
    python demo.py recording.wav    classify one file in the terminal
"""

import argparse
import json
from pathlib import Path
from typing import NamedTuple

import joblib
import librosa
import numpy as np
import torch
from funasr import AutoModel
from transformers import (
    BertModel,
    BertTokenizer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

ROOT = Path(__file__).resolve().parent
SAMPLING_RATE = 16000  # both Whisper and emotion2vec expect 16 kHz mono
EXAMPLES = [
    ROOT / "data" / "emotional_text" / "sad" / "sad_21.wav",
    ROOT / "data" / "emotional_text" / "happy" / "happy_21.wav",
    ROOT / "data" / "neutral_text" / "sad" / "s21.wav",
    ROOT / "data" / "neutral_text" / "happy" / "s21.wav",
]


class Result(NamedTuple):
    transcription: str
    audio_scores: dict  # emotion -> probability, from how the voice sounds
    text_scores: dict  # emotion -> probability, from what the words say
    trusted_audio: bool
    prediction: str

    @property
    def audio_label(self):
        return max(self.audio_scores, key=self.audio_scores.get)

    @property
    def text_label(self):
        return max(self.text_scores, key=self.text_scores.get)

    @property
    def audio_confidence(self):
        return self.audio_scores[self.audio_label]

    @property
    def text_confidence(self):
        return self.text_scores[self.text_label]


class Pipeline:
    """Holds every model, so they load once instead of once per recording."""

    def __init__(self, models_dir):
        self.audio_clf = joblib.load(models_dir / "audio_clf.joblib")
        self.text_clf = joblib.load(models_dir / "text_clf.joblib")
        self.threshold = json.loads((models_dir / "fusion.json").read_text())["threshold"]

        self.emotion2vec = AutoModel(model="iic/emotion2vec_plus_base")
        self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        self.whisper = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
        self.whisper.config.forced_decoder_ids = None
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased").eval()

    def transcribe(self, audio):
        """Speech to text, so the text branch has something to read."""
        features = self.whisper_processor(
            audio, sampling_rate=SAMPLING_RATE, return_tensors="pt"
        ).input_features
        predicted_ids = self.whisper.generate(features)
        return self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

    def audio_embedding(self, wav_path):
        """One 768-dim vector describing how the utterance sounds."""
        result = self.emotion2vec.generate(
            str(wav_path), granularity="utterance", extract_embedding=True
        )
        return np.asarray(result[0]["feats"]).reshape(1, -1)

    def text_embedding(self, text):
        """One 768-dim vector describing what the utterance says.

        Mean pooling over the token vectors, matching extract_text_embedding.py - the
        two must agree or the classifier sees features it was never trained on.
        """
        encoded_input = self.bert_tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = self.bert(**encoded_input)

        mask = encoded_input["attention_mask"].unsqueeze(-1).float()
        return ((output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)).numpy()

    def run(self, wav_path):
        audio, _ = librosa.load(wav_path, sr=SAMPLING_RATE, mono=True)

        # branch 1: how it sounds
        audio_scores = scores(self.audio_clf, self.audio_embedding(wav_path))

        # branch 2: what it says
        transcription = self.transcribe(audio)
        text_scores = scores(self.text_clf, self.text_embedding(transcription))

        # the gate
        audio_label = max(audio_scores, key=audio_scores.get)
        text_label = max(text_scores, key=text_scores.get)
        trusted_audio = audio_scores[audio_label] >= self.threshold

        return Result(
            transcription=transcription,
            audio_scores=audio_scores,
            text_scores=text_scores,
            trusted_audio=trusted_audio,
            prediction=audio_label if trusted_audio else text_label,
        )


def scores(clf, X):
    return {label: float(p) for label, p in zip(clf.classes_, clf.predict_proba(X)[0])}


def gate_sentence(result, threshold):
    if result.trusted_audio:
        return (
            f"Audio is **{result.audio_confidence:.2f}** confident, at or above the "
            f"**{threshold:.2f}** threshold, so its answer stands."
        )
    return (
        f"Audio is only **{result.audio_confidence:.2f}** confident, below the "
        f"**{threshold:.2f}** threshold, so the text answer is used instead."
    )


def print_result(wav_path, result, threshold):
    print(f"\nfile           {wav_path}")
    print(f'transcription  "{result.transcription}"')
    print(f"\naudio branch   {result.audio_label:<10}confidence {result.audio_confidence:.2f}")
    print(f"text branch    {result.text_label:<10}confidence {result.text_confidence:.2f}")
    verdict = "confident, its answer stands" if result.trusted_audio else "unsure, falling back to text"
    print(f"gate           threshold {threshold:.2f} -> audio {verdict}")
    print(f"\nprediction:    {result.prediction}\n")


def launch_ui(pipeline, share):
    import gradio as gr

    def analyse(wav_path):
        if not wav_path:
            raise gr.Error("Record or upload something first.")

        result = pipeline.run(wav_path)
        return (
            {result.prediction: 1.0},
            result.transcription,
            result.audio_scores,
            result.text_scores,
            gate_sentence(result, pipeline.threshold),
        )

    with gr.Blocks(title="Speech emotion recognition") as ui:
        gr.Markdown(
            "# Speech emotion recognition\n"
            "The voice is read by **emotion2vec**, the words by **Whisper + BERT**. "
            "The voice decides unless it is unsure, in which case the words do."
        )

        with gr.Row():
            with gr.Column():
                recording = gr.Audio(
                    sources=["upload", "microphone"], type="filepath", label="Recording"
                )
                analyse_button = gr.Button("Analyse", variant="primary")
                gr.Examples(
                    examples=[[str(path)] for path in EXAMPLES if path.exists()],
                    inputs=recording,
                    label="Or try one of these",
                )

            with gr.Column():
                prediction = gr.Label(label="Prediction", num_top_classes=1)
                transcription = gr.Textbox(label="What Whisper heard", interactive=False)
                gate = gr.Markdown()

        gr.Markdown("### How each branch voted")
        with gr.Row():
            audio_scores = gr.Label(label="Voice (emotion2vec)")
            text_scores = gr.Label(label="Words (BERT)")

        analyse_button.click(
            fn=analyse,
            inputs=recording,
            outputs=[prediction, transcription, audio_scores, text_scores, gate],
        )

    ui.launch(share=share)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "wav", type=Path, nargs="?", help="classify this file and exit, instead of opening the UI"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=ROOT / "models",
        help="classifiers and settings from train_fusion.py (default: models)",
    )
    parser.add_argument("--share", action="store_true", help="expose the UI on a public gradio link")
    return parser.parse_args()


def main():
    args = parse_args()
    print("loading models ...")
    pipeline = Pipeline(args.models_dir)

    if args.wav:
        print_result(args.wav, pipeline.run(args.wav), pipeline.threshold)
    else:
        launch_ui(pipeline, args.share)


if __name__ == "__main__":
    main()
