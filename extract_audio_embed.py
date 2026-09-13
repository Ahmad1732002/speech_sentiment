from funasr import AutoModel

model = AutoModel(model="iic/emotion2vec_plus_base")


wav_file = f"{model.model_path}/example/test.wav"
print('wav_file_location:',wav_file)
res = model.generate(wav_file, output_dir="./outputs", granularity="utterance", extract_embedding=True)
print(res[0]["feats"].shape)
audio_embedding = res[0]["feats"].numpy()


# print(res)
