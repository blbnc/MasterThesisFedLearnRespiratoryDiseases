import librosa
import numpy as np


def extract_features(audio_path, sample_rate=4000, n_mfcc=13, n_fft=2048):
    """
    Extract MFCC features from an audio file.
    """
    y, sr = librosa.load(audio_path, sr=sample_rate)

    n_fft = min(n_fft, len(y))
    # y = pad_audio(y, n_fft) -> This would add zeros to fit target length

    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft)
    # Take the mean of MFCCs over time
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean


# This might be illegal, idk
def pad_audio(audio, target_length):
    if len(audio) < target_length:
        # Pad with zeros to reach the target length
        padding = target_length - len(audio)
        audio = np.pad(audio, (0, padding), mode='constant')
    return audio


import os
import pandas as pd

def process_data_for_ml(data_dir, output_csv, sample_rate=4000, n_mfcc=13):
    """
    Process the segmented audio dataset and save features and labels to a CSV file.
    """
    features = []
    labels = []

    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):  # Process only audio files
                # Extract features
                file_path = os.path.join(root, file)
                mfcc = extract_features(file_path, sample_rate, n_mfcc)

                # Extract label from the filename (e.g., "Normal" or "Abnormal") // THIS IS JUST FOR SPRSOUND files TODO: Adjust file naming for ICBHI files to accomodate
                label = "Normal" if "Normal" in file else "Abnormal"

                features.append(mfcc)
                labels.append(label)

    # Convert to a DataFrame for easy manipulation
    df = pd.DataFrame(features)
    df['label'] = labels

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Features and labels saved to {output_csv}")

def process_data_for_multiclass_ml(data_dir, output_csv, sample_rate=4000, n_mfcc=13):
    """
    Process the segmented audio dataset and save features and multi-class labels to a CSV file.
    """
    features = []
    labels = []

    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.wav'):  # Process only audio files
                # Extract features
                file_path = os.path.join(root, file)
                mfcc = extract_features(file_path, sample_rate, n_mfcc)

                # Extract label from the filename (e.g., "Wheeze", "Fine Crackle")
                if "Wheeze+Crackle" in file:
                    label = "Wheeze+Crackle"
                elif "Wheeze" in file:
                    label = "Wheeze"
                elif "Fine Crackle" in file:
                    label = "Fine Crackle"
                elif "Coarse Crackle" in file:
                    label = "Coarse Crackle"
                else:
                    label = "Normal"

                features.append(mfcc)
                labels.append(label)

    # Convert to a DataFrame for easy manipulation
    df = pd.DataFrame(features)
    df['label'] = labels

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Features and labels saved to {output_csv}")


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def prepare_data_for_training(csv_file):
    """
    Load features and labels from a CSV file, normalize them, and split into train/test sets.
    """
    # Load data
    data = pd.read_csv(csv_file)

    # Separate features and labels
    X = data.iloc[:, :-1].values  # All columns except the last
    y = data['label'].values

    # Encode labels (e.g., "Normal" -> 0, "Abnormal" -> 1)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, encoder



data_dir = 'SPRSound/Classification/train_classification_cycles_filtered'  # Directory with segmented audio files
output_csv = 'SPRSound_audio_features.csv'   # Save extracted features and labels

# Step 1: Extract features and labels
process_data_for_multiclass_ml(data_dir, output_csv)

# Step 2: Prepare data for training
X_train, X_test, y_train, y_test, encoder = prepare_data_for_training(output_csv)

# Labels encoding information
print("Classes:", encoder.classes_)

# Example shapes
print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)