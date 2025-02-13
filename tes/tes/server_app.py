"""tes: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from tes.ml_models import CNN6

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def evaluate_metrics_aggregation_fn(metrics):
    """Aggregate evaluation metrics."""
    num_clients = len(metrics)
    
    # Extract losses and accuracies from the tuple structure
    total_loss = sum(m[0] for m in metrics)  # First element is loss
    total_accuracy = sum(m[1]["accuracy"] for m in metrics)  # Second element is a dictionary

    # Compute aggregated metrics
    aggregated_metrics = {
        "loss": total_loss / num_clients,
        "accuracy": total_accuracy / num_clients,
    }
    return aggregated_metrics

def fit_metrics_aggregation_fn(metrics):
    """Aggregate training (fit) metrics."""
    print("Fit Metrics received:", metrics)  # Debugging

    num_clients = len(metrics)
    
    # Extract losses and accuracies from the tuple structure
    total_loss = sum(m[0] for m in metrics)  # First element is loss
    total_accuracy = sum(m[1].get("accuracy", 0) for m in metrics)  # Second element is a dictionary

    # Compute aggregated metrics
    aggregated_metrics = {
        "loss": total_loss / num_clients,
        "accuracy": total_accuracy / num_clients,
    }
    return aggregated_metrics

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(CNN6())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
