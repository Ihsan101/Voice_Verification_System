import os
import librosa
import numpy as np


def extract_mel_spectrogram(file_path, sr=16000, n_mels=128, duration=3, fixed_width=94):
    audio, _ = librosa.load(file_path, sr=sr, duration=duration)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    if mel_spec_db.shape[1] < fixed_width:
        padding = fixed_width - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, padding)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :fixed_width]
    return mel_spec_db

def load_data(data_dir, fixed_width=94):
    X, Y = [], []
    
    my_voice_dir = os.path.join(data_dir, "my_voice")
    print(f"Provessing in {my_voice_dir}!")
    for file_name in os.listdir(my_voice_dir):
        if file_name.endswith(".wav"):
            file_path = os.path.join(my_voice_dir, file_name)
            mel_spec = extract_mel_spectrogram(file_path, fixed_width=fixed_width)
            X.append(mel_spec)
            Y.append(1)

    not_my_voice_dir = os.path.join(data_dir, "not_my_voice")
    print(f"Provessing in {not_my_voice_dir}!")
    for file_name in os.listdir(not_my_voice_dir):
        if file_name.endswith(".wav"):
            file_path = os.path.join(not_my_voice_dir, file_name)
            mel_spec = extract_mel_spectrogram(file_path, fixed_width=fixed_width)
            X.append(mel_spec)
            Y.append(0)

    X = np.array(X).reshape(-1, 128, fixed_width, 1)
    Y = np.array(Y)
    return X, Y


train_data_dir = "data/train"
test_data_dir = "data/test"
val_data_dir = "data/val"

X_train, y_train = load_data(train_data_dir)
X_test, y_test = load_data(test_data_dir)
X_val, y_val = load_data(val_data_dir)

np.save("models/X_train.npy", X_train)
np.save("models/y_train.npy", y_train)
np.save("models/X_test.npy", X_test)
np.save("models/y_test.npy", y_test)
np.save("models/X_val.npy", X_val)
np.save("models/y_val.npy", y_val)