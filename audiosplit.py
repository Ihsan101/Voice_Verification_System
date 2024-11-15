from pydub import AudioSegment
import os
import random

input_file = "recording.wav"
output_dir = "data/all_audio/"

audio = AudioSegment.from_wav(input_file)

segment_duration = 3000 #milliseconds

total_segments = len(audio) // segment_duration

segments_80 = int(0.8 * total_segments)
segments_15 = int(0.15 * total_segments)
segments_5 = total_segments - segments_80 - segments_15

segment_indices = list(range(total_segments))
random.shuffle(segment_indices)

for i, idx in enumerate(segment_indices):
    start_time = idx * segment_duration
    end_time = min((idx + 1) * segment_duration, len(audio))
    segment = audio[start_time:end_time]

    if i < segments_80:
        filename = os.path.join(output_dir, f"train_data_{i + 1}.wav")
    elif i < segments_80 + segments_15:
        filename = os.path.join(output_dir, f"test_data_{i + 1}.wav")
    else:
        filename = os.path.join(output_dir, f"val_data_{i + 1}.wav")

    segment.export(filename, format="wav")
    print(f"Done w/{filename}")

print("Finished Finally!")
