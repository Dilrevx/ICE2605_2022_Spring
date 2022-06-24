import librosa, soundfile

original, sr = librosa.load(path="Obama.mp3", sr=None, duration=30)
soundfile.write('../2.wav',original, sr)
