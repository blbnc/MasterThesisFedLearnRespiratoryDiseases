"""tes: A Flower / PyTorch app."""

import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from tes.task import get_weights, load_data, set_weights, test, train
from tes.ml_models import CNN6, WeakCNN
import torch.nn as nn


# Define Flower Client and client_fn
class CustomClient(NumPyClient):
    def __init__(self, client_type, model, train_loader, test_loader):
        self.client_type = client_type
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_parameters(self):
        """Return model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        # Training logic for the client
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(1):  # Small epochs for demonstration
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        criterion = nn.CrossEntropyLoss()
        loss, correct = 0.0, 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss += criterion(outputs, labels).item()
                correct += (outputs.argmax(1) == labels).type(torch.float).sum().item()
        accuracy = correct / len(self.test_loader.dataset)
        return float(loss / len(self.test_loader)), len(self.test_loader.dataset), {"accuracy": accuracy}

current_client_index = 0

def client_fn_type1(context: Context) -> CustomClient:
    # Load model and data
    partition_id = context.node_config["partition-id"]

    # Logic to assign client type dynamically
    if partition_id % 4 == 0:
        client_type = 1
        model = WeakCNN(num_classes=2)
    elif partition_id % 4 == 1:
        client_type = 2
        model = WeakCNN(num_classes=2)#CNN6(num_classes=7)
    elif partition_id % 4 == 2:
        client_type = 3
        model = WeakCNN(num_classes=2)
    else:
        client_type = 4
        model = WeakCNN(num_classes=2)
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    train_loader, test_loader, label_encoder = load_data(partition_id, num_partitions, client_type)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return CustomClient(client_type, model, train_loader, test_loader).to_client()

# Flower ClientApp
app = ClientApp(
    client_fn_type1,
)