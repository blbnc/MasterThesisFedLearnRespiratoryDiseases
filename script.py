
import librosa
from scipy.signal import butter, filtfilt
import numpy as np
import os
import matplotlib.pyplot as plt
import soundfile as sf

# Real-world collected audio samples are usually of low SNR (signal-to-noise ratio). 
# For model development, proper de-noising is generally the first step before further processing. 
# For lung sounds associated with crackle and wheeze, as suggested by the previous studies, 
# re-sampling audio recordings to 4 KHz and deploying a fifth-order Butterworth band-pass filter 
# having 100–200 Hz cut-off frequencies can effectively eliminate the environmental noise such as heartbeat, 
# motion artifacts, and audio sounds.32,33 After that, respiratory cycles (inspiratory–expiatory periods) 
# could be identified to further increase the SNR. Microphone-acquired audio data usually needs a sound-type 
# check to avoid including improper sound modality, which can be performed manually or automatically. 
# 34 For instance, researchers developed a cough detector to select high-quality cough recordings for experiments.26,28,29 
# Some studies proposed to extract single cough clips from audio recordings as model inputs, 35 as they think this further 
# increases the SNR, while most researchers used the complete recordings because they hypothesize that silence frames between 
# multiple coughs are also informative.36,37 Subsequently, temporal features can be extracted directly, and usually, audio segments
# will be transferred into spectrograms via short-time Fourier transforms. https://pmc.ncbi.nlm.nih.gov/articles/PMC9791302/

def save_filtered_audio(input_path, output_path, filtered_audio, sample_rate):
    """
    Save the filtered audio to disk.
    """
    sf.write(output_path, filtered_audio, sample_rate)
    print(f"Saved filtered audio to {output_path}")


def butter_lowpass_filter(data, cutoff_freq, sample_rate, order=5):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    print(f"Filtered audio shape: {filtered_data.shape}")
    return filtered_data

def butter_bandpass_filter(data, lowcut, highcut, sample_rate, order=5):
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def stream_audio_dataset(dataset_path, batch_size=32, target_sr=None, save_to_disk = False):
    # Get all audio file paths in the dataset path
    audio_files = [os.path.join(root, file) for root, dirs, files in os.walk(dataset_path) for file in files]
 
    # Shuffle the audio files for randomness
    np.random.shuffle(audio_files)
 
    for i in range(0, len(audio_files), batch_size):
        batch_paths = audio_files[i:i + batch_size]
 
        for file_path in batch_paths:
            # Check if file exists
            if not os.path.exists(file_path):
               continue
            # Load and preprocess each audio file
            y, sr = librosa.load(file_path, sr=target_sr)
 
            # Resampling
            if target_sr is not None and sr != target_sr:
                y = librosa.resample(y, sr, target_sr)
                sr = target_sr
 
            #filtered_audio = butter_lowpass_filter(y, cutoff_freq=200, sample_rate=sr)
            filtered_audio = butter_bandpass_filter(y, lowcut=100, highcut=800, sample_rate=sr)

            if(save_to_disk):
                output_path = file_path.replace(dataset_path, f"{dataset_path}_filtered")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                save_filtered_audio(file_path, output_path, filtered_audio, sr)
  
# Load the dataset folder
dataset_path = 'ICBHI_cycles'
 
stream_audio_dataset(dataset_path, batch_size=32, target_sr=4000, save_to_disk=True)