import customtkinter as ctk
import numpy as np
import sounddevice as sd
import tensorflow as tf
import librosa
import tkinter.messagebox as messagebox


# Load the trained model
model = tf.keras.models.load_model("final_models/voice_verification_model.h5")

# Set sample rate and duration for recording
sr = 16000
duration = 3

# Function to record audio using sounddevice
def record_audio(duration, sample_rate):
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    return np.squeeze(audio)  # Remove single-dimensional entries

# Preprocess the recorded audio for the model
def preprocess_audio(audio, sample_rate):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Pad or crop to a fixed size
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

def display_result(prediction, threshold=0.5):
    if prediction > threshold:
        messagebox.showinfo("Verification Result", "Verified: This is your voice!")
    else:
        messagebox.showwarning("Verification Result", "Not Verified: This is NOT your voice!")

def record_and_verify():
    status_label.configure(text="Recording... Please speak")
    app.update_idletasks()
    
    audio = record_audio(duration, sr)
    status_label.configure(text="Recording complete! Processing...")
    app.update_idletasks()

    processed_audio = preprocess_audio(audio, sr)
    prediction = make_prediction(model, processed_audio)
    display_result(prediction)
    status_label.configure(text="Ready to record")

app = ctk.CTk()
app.title("Voice Verification System")
app.geometry("400x300")

frame = ctk.CTkFrame(master=app)
frame.pack(pady=20, padx=20, fill="both", expand=True)

title_label = ctk.CTkLabel(master=frame, text="Autonomous Voice Verification System", font=("Arial", 18))
title_label.pack(pady=10)


status_label = ctk.CTkLabel(master=frame, text="Ready to record", font=("Arial", 14))
status_label.pack(pady=10)


record_button = ctk.CTkButton(master=frame, text="Record & Verify", command=record_and_verify)
record_button.pack(pady=20)

app.mainloop()
