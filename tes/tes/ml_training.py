from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import math
import pandas as pd
import numpy as np
import os

import librosa
import numpy as np
import matplotlib.pyplot as plt

def plot_mel_spectrogram(mel_spectrogram, sample_rate, hop_length, title="Mel Spectrogram"):
    """
    Plot a Mel spectrogram.
    Args:
        mel_spectrogram (np.ndarray): Mel spectrogram.
        sample_rate (int): Sampling rate of the audio.
        hop_length (int): Hop length used for STFT.
        title (str): Title of the plot.
    """
    #plt.figure(figsize=(10, 6))

    plt.figure(figsize=(4, 4))
    librosa.display.specshow(mel_spectrogram, sr=sample_rate, x_axis='time', y_axis='mel', fmax=2000)
    plt.axis('off')
    librosa.display.specshow(mel_spectrogram, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='mel', cmap='viridis')
    #plt.colorbar(format='%+2.0f dB')
    #plt.title(title)
    #plt.xlabel("Time")
    #plt.ylabel("Mel Frequency (Hz)")
    plt.tight_layout()
    plt.show()

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
    print(audio_path)
    plot_mel_spectrogram(log_mel_spec, sample_rate=sr, hop_length=hop_length)
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

    print(labels)
    labels1 = ["Normal" if x == "normal" else "Abnormal" for x in labels]
    labels2 = labels
    labels3 = ["Wheeze" if x == "wheezes" or x == "crackles_wheezes" else "Non-Wheeze" for x in labels]
    labels4 = ["Crackle" if x == "crackles" or x == "crackles_wheezes" else "Non-Crackle" for x in labels]

    labels = [labels1, labels2, labels3, labels4]

    print(set(labels[0]))
    print(set(labels[1]))
    print(set(labels[2]))
    print(set(labels[3]))

    y = [LabelEncoder().fit_transform(x) for x in labels]
    print(y)

    # Target shape for spectrograms
    target_shape = (126, 55)

    # Generate spectrograms and process labels
    spectrograms = []
    for file_path in file_paths:
        spec = generate_spectrogram(file_path)
        #print(spec.shape)
        spec = pad_or_truncate(spec, target_shape)
        #print(file_path)
        spectrograms.append(spec)

    #plot_mel_spectrogram(spectrograms[0], sample_rate=4000, hop_length=128)
    #plot_mel_spectrogram(spectrograms[1], sample_rate=16000, hop_length=128)
    #plot_mel_spectrogram(spectrograms[2], sample_rate=16000, hop_length=128)
    # Convert to NumPy array and add channel dimension for CNN
    X = np.array(spectrograms)
    X = X[:, np.newaxis, :, :]  # Shape: (num_samples, channels, height, width)

    print(f"Spectrogram dataset shape: {X.shape}")
   
    # Train-test split

    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y[0], test_size=0.2, stratify=y[0], random_state=42)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y[1], test_size=0.2, stratify=y[1], random_state=42)
    X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y[2], test_size=0.2, stratify=y[2], random_state=42)
    X_train4, X_test4, y_train4, y_test4 = train_test_split(X, y[3], test_size=0.2, stratify=y[3], random_state=42)

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    # Create DataLoaders
    train_dataset1 = TensorDataset(torch.tensor(X_train1).float(), torch.tensor(y_train1).long())
    test_dataset1 = TensorDataset(torch.tensor(X_test1).float(), torch.tensor(y_test1).long())
    train_dataset2 = TensorDataset(torch.tensor(X_train2).float(), torch.tensor(y_train2).long())
    test_dataset2 = TensorDataset(torch.tensor(X_test2).float(), torch.tensor(y_test2).long())
    train_dataset3 = TensorDataset(torch.tensor(X_train3).float(), torch.tensor(y_train3).long())
    test_dataset3 = TensorDataset(torch.tensor(X_test3).float(), torch.tensor(y_test3).long())
    train_dataset4 = TensorDataset(torch.tensor(X_train4).float(), torch.tensor(y_train4).long())
    test_dataset4 = TensorDataset(torch.tensor(X_test4).float(), torch.tensor(y_test4).long())

    train_loaders = [
        DataLoader(train_dataset1, batch_size=32, shuffle=True),
        DataLoader(train_dataset2, batch_size=32, shuffle=True),
        DataLoader(train_dataset3, batch_size=32, shuffle=True),
        DataLoader(train_dataset4, batch_size=32, shuffle=True)
    ]
    test_loaders = [
        DataLoader(test_dataset1, batch_size=32, shuffle=False),
        DataLoader(test_dataset2, batch_size=32, shuffle=False),
        DataLoader(test_dataset3, batch_size=32, shuffle=False),
        DataLoader(test_dataset4, batch_size=32, shuffle=False)
    ]
    

    return train_loaders, test_loaders


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





class CNN6(nn.Module):
    def __init__(self, num_classes):
        super(CNN6, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = self.pool(torch.relu(self.conv5(x)))
        x = self.global_avg_pool(torch.relu(self.conv6(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for spectrograms, labels in dataloader:
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        # Forward pass
        outputs = model(spectrograms)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Train Loss: {running_loss / len(dataloader):.4f}, Train Accuracy: {accuracy:.2f}%")
    return running_loss / len(dataloader), accuracy

# Validation Function
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for spectrograms, labels in dataloader:
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            # Forward pass
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation Loss: {running_loss / len(dataloader):.4f}, Validation Accuracy: {accuracy:.2f}%")
    return running_loss / len(dataloader), accuracy

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Modified Validation Function with Detailed Results
def validate_with_detailed_results(model, dataloader, criterion, device, class_names):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for spectrograms, labels in dataloader:
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            # Forward pass
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            # Get predictions
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate loss
    avg_loss = running_loss / len(dataloader)

    # Classification report and confusion matrix
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    class_report = classification_report(all_labels, all_predictions, target_names=class_names)
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    print(f"Validation Loss: {avg_loss:.4f}")
    print("\nDetailed Results:")
    print(class_report)

    print("Confusion Matrix:")
    print(conf_matrix)

    # Class-wise accuracy
    class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    print("\nClass-Wise Accuracies:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {class_accuracies[i]:.2f}")

    return avg_loss, class_report, conf_matrix, class_accuracies

# Path to your dataset directory
data_dir = "H:\\Sound\\ICBHI_7s_fixed_length_processed"

# Load dataset using your existing function
train_loaders, test_loaders = load_datasets(data_dir)

# Training each task separately
num_classes_list = [2, len(set(train_loaders[1].dataset.tensors[1].numpy())),
                    2, 2]  # Adjust based on your tasks

class_names_list = [
        ["Normal", "Abnormal"],  # For Task 1
        ['Normal', 'Wheeze', 'Crackle'],
        ["Non-Wheeze", "Wheeze"],  # For Task 3
        ["Non-Crackle", "Crackle"]  # For Task 4
    ]

for task_id, (train_loader, test_loader) in enumerate(zip(train_loaders, test_loaders)):
    print(f"\nTraining Task {task_id + 1}")
    
    # Initialize model
    num_classes = num_classes_list[task_id]
    class_names = class_names_list[task_id]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN6(num_classes=num_classes).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 2
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    
    # Validate and print detailed results
    print(f"\nValidating Task {task_id + 1}")
    val_loss, class_report, conf_matrix, class_accuracies = validate_with_detailed_results(
        model, test_loader, criterion, device, class_names
    )