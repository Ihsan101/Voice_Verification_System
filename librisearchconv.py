import os
from pydub import AudioSegment
import random

source_dir = "data/LibriSpeech"
destination_80 = "data/train/not_my_voice"
destination_15 = "data/test/not_my_voice"
destination_5 = "data/val/not_my_voice"


flac_files = []
for root, _, files in os.walk(source_dir):
    for file in files:
        if file.endswith(".flac"):
            flac_files.append(os.path.join(root, file))


random.shuffle(flac_files)


total_files = len(flac_files)
split_80 = int(0.8 * total_files)
split_95 = int(0.95 * total_files)


files_80 = flac_files[:split_80] 
files_15 = flac_files[split_80:split_95]
files_5 = flac_files[split_95:]

def convert_and_move(files, destination):
    for flac_file_path in files:
        file_name = os.path.splitext(os.path.basename(flac_file_path))[0] + ".wav"
        wav_file_path = os.path.join(destination, file_name)

        try:
            audio = AudioSegment.from_file(flac_file_path, format="flac")
            audio.export(wav_file_path, format="wav")
            print(f"Converted and moved: {flac_file_path} -> {wav_file_path}")

        except Exception as e:
            print(f"Error converting {flac_file_path}: {e}")


print("Processing 80% data to 'train' folder...")
convert_and_move(files_80, destination_80)

print("Processing 15% data to 'test' folder...")
convert_and_move(files_15, destination_15)

print("Processing 5% data to 'val' folder...")
convert_and_move(files_5, destination_5)

print("DONE YOOOO!")
