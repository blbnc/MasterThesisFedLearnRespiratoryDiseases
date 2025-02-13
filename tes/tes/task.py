"""tes: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

NUM_CLIENTS = 5
BATCH_SIZE = 32


import tes.ml_training as ml_training
def load_data(partition_id: int, num_partitions: int, client_type: int):
    #if train_loader is None:
    train_loader, test_loader = ml_training.load_datasets('H:\Sound\SPRSound\Classification\\train_classification_cycles', client_type)

    # Step 2: Partition the training dataset into `num_partitions`
    #client_datasets = ml_training.partition_dataset(train_loader.dataset, num_partitions)

    # Step 3: Select the dataset for the current partition
    #client_dataset = client_datasets[partition_id]

    # Step 4: Create DataLoader for the partitioned dataset
    #client_train_loader = DataLoader(client_dataset, batch_size=BATCH_SIZE, shuffle=True)

    client_train_loader = DataLoader(train_loader.dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Test loader is shared among all clients, so we return it as is
    return client_train_loader, test_loader


def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    try:
        net.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"Partial load due to mismatch: {e}")
        net.load_state_dict(state_dict, strict=False)
