from pathlib import Path

from funasr import AutoModel

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "neutral_text"

model = AutoModel(model="iic/emotion2vec_plus_base")


# wav_file = f"{model.model_path}/example/test.wav"
emotions = ["sad", "happy", "neutral"]
samples = [f"s{i}" for i in range(21, 26)]

wav_files = []
true_labels = []
for emotion in emotions:
    for sample in samples:
        wav_files.append(str(DATA_DIR / emotion / f"{sample}.wav"))
        true_labels.append(emotion)

print('wav_files:', wav_files)
res = model.generate(wav_files, granularity="utterance", extract_embedding=True)

print(f"\n{'file':<22} {'true':<10} {'predicted':<12} {'score':<8} correct")
print("-" * 62)

correct = 0
for path, true_label, r in zip(wav_files, true_labels, res):
    score, label = max(zip(r["scores"], r["labels"]))
    pred = label.split("/")[-1]
    hit = pred == true_label
    correct += hit
    rel_path = Path(path).relative_to(DATA_DIR).as_posix()
    print(f"{rel_path:<22} {true_label:<10} {pred:<12} {score:<8.3f} {'yes' if hit else 'NO'}")

print(f"\naccuracy: {correct}/{len(wav_files)} = {correct / len(wav_files):.1%}")



# print(res)
