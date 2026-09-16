import argparse
from pathlib import Path
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

ROOT = Path(__file__).resolve().parent
SAMPLING_RATE = 16000  # whisper expects 16 kHz audio

parser = argparse.ArgumentParser(description="Transcribe the audio dataset with Whisper.")
parser.add_argument(
    "--data-dir",
    type=Path,
    default=ROOT / "data" / "neutral_text",
    help="folder containing sad/, happy/, neutral/ subfolders (default: data/neutral_text)",
)
args = parser.parse_args()

def load_data(data_dir):
    emotions = ["sad", "happy", "neutral"]
    samples = []
    for i in range(1, 3):
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


# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
model.config.forced_decoder_ids = None

wav_files, true_labels = load_data(args.data_dir)

for path, label in zip(wav_files, true_labels):
    # read the wav file and resample to 16 kHz mono
    audio, _ = librosa.load(path, sr=SAMPLING_RATE, mono=True)
    input_features = processor(audio, sampling_rate=SAMPLING_RATE, return_tensors="pt").input_features

    # generate token ids
    predicted_ids = model.generate(input_features)
    # decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    rel_path = Path(path).relative_to(args.data_dir).as_posix()
    print(f"{rel_path:<16} {label:<8} {transcription.strip()}")
