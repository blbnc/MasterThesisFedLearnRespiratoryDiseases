import librosa
import os
import numpy as np
import soundfile as sf

def parse_cycles_file(cycles_file):
    """
    Parse the respiratory cycles file to extract time intervals and labels.
    File format: start_time, end_time, crackles, wheezes
    """
    cycles = []
    with open(cycles_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')  # Tab-separated values
            start_time, end_time = float(parts[0]), float(parts[1])
            crackles, wheezes = int(parts[2]), int(parts[3])
            cycles.append((start_time, end_time, crackles, wheezes))
    return cycles

def extract_cycles_from_audio(audio_file, cycles, sample_rate):
    """
    Extract respiratory cycles from an audio file based on time intervals.
    """
    y, sr = librosa.load(audio_file, sr=sample_rate)
    extracted_cycles = []
    for start_time, end_time, crackles, wheezes in cycles:
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        cycle_audio = y[start_sample:end_sample]
        extracted_cycles.append((cycle_audio, crackles, wheezes))
    return extracted_cycles

def save_cycles(cycles, output_dir, base_name, sample_rate):
    """
    Save extracted cycles to individual audio files with labels in their filenames.
    """
    for i, (cycle_audio, crackles, wheezes) in enumerate(cycles):
        label_suffix = f"_crackles_{crackles}_wheezes_{wheezes}"
        output_path = os.path.join(output_dir, f"{base_name}_cycle_{i}{label_suffix}.wav")
        sf.write(output_path, cycle_audio, sample_rate)
        print(f"Saved: {output_path}")

# Example usage
dataset_path = 'ICBHI_final_database'
output_dir = 'ICBHI_cycles'
os.makedirs(output_dir, exist_ok=True)

for root, _, files in os.walk(dataset_path):
    for file in files:
        if file.endswith('.txt'):  # Cycle definitions
            base_name = os.path.splitext(file)[0]
            cycles_file = os.path.join(root, file)
            audio_file = os.path.join(root, f"{base_name}.wav")
            
            if os.path.exists(audio_file):
                cycles = parse_cycles_file(cycles_file)
                extracted_cycles = extract_cycles_from_audio(audio_file, cycles, sample_rate=16000)
                save_cycles(extracted_cycles, output_dir, base_name, sample_rate=16000)