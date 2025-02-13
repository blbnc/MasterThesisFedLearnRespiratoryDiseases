import json

def parse_json(json_file):
    """
    Parse the JSON annotation file to extract relevant information.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    record_annotation = data.get("record_annotation")
    event_annotations = data.get("event_annotation", [])
    return record_annotation, event_annotations


import librosa
import os
import numpy as np
import soundfile as sf

def segment_audio(audio_file, event_annotations, sample_rate=16000):
    """
    Segment the audio file based on event annotations.
    """
    y, sr = librosa.load(audio_file, sr=sample_rate)
    segments = []

    for event in event_annotations:
        start_time = int(event['start']) / 1000  # Convert to seconds
        end_time = int(event['end']) / 1000     # Convert to seconds
        segment_type = event['type']

        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        audio_segment = y[start_sample:end_sample]
        segments.append((audio_segment, segment_type))

    return segments

def segment_audio_to_fixed_length(audio_file, event_annotations, sample_rate=16000, length=7):
    """
    Segment the audio file based on event annotations and fixed length.
    """
    y, sr = librosa.load(audio_file, sr=sample_rate)
    segments = []
    start = 0
    end = length
    while end * sr < len(y): # prob change to start < len(y) - length
        bContainsAtLeastOneCycle = False
        bContainsWheeze = False
        bContainsCrackle = False
        for event in event_annotations:
            if(int(event['start']) >= start * sr and int(event['end']) <= end * sr):
                bContainsAtLeastOneCycle = True
                if event['type'] == "Fine Crackle" or event['type'] == "Coarse Crackle":
                    bContainsCrackle = True
                elif event['type'] == "Wheeze":
                    bContainsWheeze = True
                elif event['type'] != "Normal":
                    print(f"Unknown event type: {event['type']}")
                    return []
        if bContainsAtLeastOneCycle:
            if end * sr > len(y): # this obv. does nothing because of while loop condition
                end = len(y) // sr
            segments.append((y[start * sr:end * sr], bContainsCrackle, bContainsWheeze))
        start += length
        end += length

    return segments

def save_segments(segments, output_dir, base_name, sample_rate):
    """
    Save segmented audio files with labels.
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, (audio_segment, bContainsCrackle, bContainsWheeze) in enumerate(segments):
        if(bContainsCrackle == True):
            segment_type = "crackles"
        if(bContainsWheeze == True):
            segment_type = "wheezes"
        if(bContainsCrackle == True and bContainsWheeze == True):
            segment_type = "crackles_wheezes"
        if(bContainsCrackle == False and bContainsWheeze == False):
            segment_type = "normal"
        output_path = os.path.join(output_dir, f"{base_name}_segment_{i}_{segment_type}.wav")
        sf.write(output_path, audio_segment, sample_rate)
        print(f"Saved: {output_path}")


def process_dataset(wav_path, json_path, output_dir, sample_rate=16000, length = 7):
    """
    Process the dataset with JSON annotations and save segmented audio.
    The audio files are in `wav_path` and the JSON files are in `json_path`.
    """
    for root, _, files in os.walk(json_path):
        for file in files:
            if file.endswith('.json'):  # JSON annotation file
                base_name = os.path.splitext(file)[0]
                json_file = os.path.join(root, file)
                audio_file = os.path.join(wav_path, f"{base_name}.wav")  # Match JSON to corresponding audio file

                if os.path.exists(audio_file):
                    # Parse JSON and extract segments
                    record_annotation, event_annotations = parse_json(json_file)
                    segments = segment_audio_to_fixed_length(audio_file, event_annotations, sample_rate, length)
                    
                    # Save the segmented audio
                    save_segments(segments, output_dir, base_name, sample_rate)
                else:
                    print(f"Audio file missing for: {base_name}")


wav_path = '../SPRSound/Classification/train_classification_wav'  # Folder containing .wav files
json_path = '../SPRSound/Classification/train_classification_json'  # Folder containing .json files
output_dir = 'H:/Sound/ICBHI_7s_fixed_length'  # Folder where segmented audio will be saved

process_dataset(wav_path, json_path, output_dir, sample_rate=16000, length=7)