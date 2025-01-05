from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import math
import pandas as pd
import numpy as np
import os


from tes.ml_models import CNN6

import librosa
import numpy as np


def generate_spectrogram(audio_path, sample_rate=4000, n_fft=2048, hop_length=512, n_mels=128):
    """
    Generate a log Mel spectrogram from an audio file.
    Args:
        audio_path (str): Path to the audio file.
        sample_rate (int): Sampling rate for the audio.
        n_fft (int): Number of FFT components.
        hop_length (int): Hop length for STFT.
        n_mels (int): Number of Mel bands.
    Returns:
        np.ndarray: Log Mel spectrogram.
    """
    y, sr = librosa.load(audio_path, sr=sample_rate)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec


def pad_or_truncate(spectrogram, target_shape=(128, 128)):
    """
    Pad or truncate spectrogram to ensure consistent shape.
    Args:
        spectrogram (np.ndarray): Input spectrogram.
        target_shape (tuple): Target shape (height, width).
    Returns:
        np.ndarray: Padded or truncated spectrogram.
    """
    height, width = spectrogram.shape
    target_height, target_width = target_shape

    # Pad or truncate height
    if height < target_height:
        pad_height = target_height - height
        spectrogram = np.pad(spectrogram, ((0, pad_height), (0, 0)), mode='constant')
    else:
        spectrogram = spectrogram[:target_height, :]

    # Pad or truncate width
    if width < target_width:
        pad_width = target_width - width
        spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant')
    else:
        spectrogram = spectrogram[:, :target_width]

    return spectrogram

# Extract labels from filenames
def extract_label_from_filename(filename):
    """
    Extract label (e.g., 'Fine Crackle') from the filename.
    """
    return filename.split('_')[-1].replace('.wav', '')

from sklearn.model_selection import train_test_split

def load_datasets(data_dir):
    # Load spectrograms and labels
    spectrograms = []  # List of spectrograms
    labels = []        # List of corresponding labels

    # List of audio files
    audio_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]

    # Create a list of file paths and corresponding labels
    file_paths = [os.path.join(data_dir, f) for f in audio_files]
    labels = [extract_label_from_filename(f) for f in audio_files]
    print(set(labels))

    # Encode labels numerically
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    # Target shape for spectrograms
    target_shape = (128, 128)

    # Generate spectrograms and process labels
    spectrograms = []
    for file_path in file_paths:
        spec = generate_spectrogram(file_path)
        spec = pad_or_truncate(spec, target_shape)
        spectrograms.append(spec)

    # Convert to NumPy array and add channel dimension for CNN
    X = np.array(spectrograms)
    X = X[:, np.newaxis, :, :]  # Shape: (num_samples, channels, height, width)

    print(f"Spectrogram dataset shape: {X.shape}")
   
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print("Train labels range:", min(y_train), max(y_train))
    print("Test labels range:", min(y_test), max(y_test))

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn as nn
    import torch.optim as optim

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train).float()
    y_train_tensor = torch.tensor(y_train).long()
    X_test_tensor = torch.tensor(X_test).float()
    y_test_tensor = torch.tensor(y_test).long()

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader, label_encoder


def partition_dataset(dataset, num_clients):
    """
    Partition a dataset into subsets for federated clients.
    """
    dataset_length = len(dataset)
    base_size = dataset_length // num_clients
    remainder = dataset_length % num_clients

    # Create lengths for each subset
    lengths = [base_size] * num_clients
    for i in range(remainder):
        lengths[i] += 1  # Distribute the remainder

    # Perform the partitioning
    subsets = random_split(dataset, lengths)
    return subsets

''' ML Test
# Define model, loss, and optimizer
model = CNN6(num_classes=len(np.unique(y)))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")


from sklearn.metrics import classification_report

# Evaluate
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))
'''