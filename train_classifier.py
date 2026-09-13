from funasr import AutoModel
import numpy as np

model = AutoModel(model="iic/emotion2vec_plus_base")

def load_data():
    emotions = ["sad", "happy", "neutral"]
    samples = [f"s{i}" for i in range(1, 26)]

    wav_files = []
    true_labels = []
    for emotion in emotions:
        for sample in samples:
            wav_files.append(f"data/{emotion}/{sample}.wav")
            true_labels.append(emotion)
    
    return wav_files, true_labels


# # wav_file = f"{model.model_path}/example/test.wav"
# emotions = ["sad", "happy", "neutral"]
# samples = [f"s{i}" for i in range(21, 26)]

# wav_files = []
# true_labels = []
# for emotion in emotions:
#     for sample in samples:
#         wav_files.append(f"data/{emotion}/{sample}.wav")
#         true_labels.append(emotion)

wav_files, true_labels = load_data()
print('wav_files:', wav_files)
res = model.generate(wav_files, granularity="utterance", extract_embedding=True)
embeddings = [r["feats"] for r in res]
X=np.stack(embeddings)
y=np.array(true_labels)
print(X.shape,y.shape)
print(res[0].keys)
np.save("audio_embeddings/X.npy", X)
np.save("audio_embeddings/y.npy", y)
# later: X = np.load("outputs/X.npy"); y = np.load("outputs/y.npy", allow_pickle=True)

