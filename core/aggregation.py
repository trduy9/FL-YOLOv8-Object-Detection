import torch
from copy import deepcopy

def federated_average(state_dicts):
    """
    Performs federated averaging on a list of model state dictionaries.

    Args:
        state_dicts (list[dict]): A list of model state_dicts to be averaged.

    Returns:
        dict: A new state_dict with the averaged weights. Returns an empty dict if input is empty.
    """
    if not state_dicts:
        return {}

    # Use the first state_dict as a template for the aggregated weights
    avg_state_dict = deepcopy(state_dicts[0])

    # Zero out the template to ensure a clean slate
    for key in avg_state_dict:
        avg_state_dict[key] = torch.zeros_like(avg_state_dict[key], dtype=torch.float32)

    # Sum all the state_dicts
    for sd in state_dicts:
        for key in avg_state_dict:
            if key in sd:
                # Add a shape check for robustness
                if avg_state_dict[key].shape == sd[key].shape:
                    avg_state_dict[key] += sd[key]
                else:
                    print(f"Warning: Skipping key '{key}' due to shape mismatch during aggregation.")
            else:
                print(f"Warning: Skipping key '{key}' as it is missing in one of the client models.")

    # Divide by the number of clients to get the average
    num_clients = len(state_dicts)
    if num_clients > 0:
        for key in avg_state_dict:
            avg_state_dict[key] = torch.div(avg_state_dict[key], num_clients)

    return avg_state_dict 


def fedprox_aggregate(state_dicts, sample_counts=None):
    """
    Performs federated aggregation as defined in FedProx (Li et al., 2020).
    Aggregates model state dictionaries weighted by the number of samples from each client:
        w^{t+1} = sum_{k=1}^K (n_k / sum_j n_j) * w_k^{t+1}
    
    If sample_counts is not provided or all zero, falls back to uniform averaging (FedAvg).

    Args:
        state_dicts (list[dict]): A list of client model state_dicts to aggregate.
        sample_counts (list[int|float], optional): Number of training samples (or weights)
            for each participating client in the current round.

    Returns:
        dict: Aggregated global model state_dict.
    """
    if not state_dicts:
        return {}

    num_clients = len(state_dicts)
    if num_clients == 1:
        return deepcopy(state_dicts[0])

    # Calculate normalized weights p_k = n_k / sum(n_j)
    if sample_counts is not None and len(sample_counts) == num_clients:
        total_samples = float(sum(sample_counts))
        if total_samples > 0:
            weights = [float(s) / total_samples for s in sample_counts]
        else:
            weights = [1.0 / num_clients] * num_clients
    else:
        weights = [1.0 / num_clients] * num_clients

    # Template based on the first state_dict
    avg_state_dict = deepcopy(state_dicts[0])
    for key in avg_state_dict:
        avg_state_dict[key] = torch.zeros_like(avg_state_dict[key], dtype=torch.float32)

    # Weighted sum: sum_k p_k * w_k
    for sd, weight in zip(state_dicts, weights):
        for key in avg_state_dict:
            if key in sd:
                if avg_state_dict[key].shape == sd[key].shape:
                    avg_state_dict[key] += sd[key].to(torch.float32) * weight
                else:
                    print(f"Warning: Skipping key '{key}' due to shape mismatch during FedProx aggregation.")
            else:
                print(f"Warning: Skipping key '{key}' as it is missing in one of the client models.")

    return avg_state_dict


# Alias for convenience
federated_prox_average = fedprox_aggregate
