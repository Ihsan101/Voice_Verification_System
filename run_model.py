import numpy as np
import sounddevice as sd
import tensorflow as tf
import librosa


model = tf.keras.models.load_model("final_models/voice_verification_model.h5")


sr = 16000 
duration = 3 


def record_audio(duration, sample_rate):
    print("Recording... Please speak")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Recording complete!")
    return np.squeeze(audio) 

def preprocess_audio(audio, sample_rate):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    if mel_spec_db.shape[1] < 94:
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, 94 - mel_spec_db.shape[1])), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :94]

    mel_spec_db = mel_spec_db / 255.0

    mel_spec_db = mel_spec_db.reshape(1, 128, 94, 1)

    return mel_spec_db

def make_prediction(model, processed_audio):
    prediction = model.predict(processed_audio)
    return prediction[0][0]

def display_result(prediction, threshold=0.95):
    if prediction > threshold:
        print("Verified: This is your voice!")
    else:
        print("Not Verified: This is NOT your voice!")

audio = record_audio(duration, sr)
    

processed_audio = preprocess_audio(audio, sr)
    

prediction = make_prediction(model, processed_audio)
    

display_result(prediction, 0.5)

