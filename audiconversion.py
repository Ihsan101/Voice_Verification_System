import librosa
import numpy as np
import matplotlib.pyplot as plt

def extract_mel_spectrogram(file_path, n_mels=128, duration=3, sr=16000):
    audio, sr = librosa.load(file_path, sr=sr, duration=duration)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

mel_spec = extract_mel_spectrogram("recording.wav")
print(mel_spec)
print('''
      
ARRAY SHOWN NOW
   
      ''')
print(np.array(mel_spec).reshape(-1, 128, 94, 1))
plt.imshow(mel_spec, cmap='viridis')
plt.title("Mel Spectrogram")
plt.colorbar()
plt.show()