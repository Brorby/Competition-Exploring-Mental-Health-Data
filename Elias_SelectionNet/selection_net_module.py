import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Device configuration: use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class KnapsackLayerFunction(torch.autograd.Function):
    """
    A custom differentiable layer that selects the top-k elements from each input sample
    via a stochastic relaxation for gradient estimation.

    Forward pass:
        - Takes scores `c` and selects the k largest per sample to form a binary mask `b`.
    Backward pass:
        - Uses a simple finite-difference approach (with Gaussian noise) to approximate
          gradients w.r.t. the input scores.
    """
    @staticmethod
    def forward(ctx, c, k, epsilon, m):
        # c: tensor of shape (batch_size, n_items), raw scores per item
        # k: number of items to select per sample
        # epsilon: noise scale for gradient estimation
        # m: number of Monte Carlo samples for the finite-difference estimate

        # Initialize binary mask b with zeros
        b = torch.zeros_like(c)
        # Find indices of top-k scores along the item dimension
        _, topk_indices = torch.topk(c, k, dim=1)
        # Scatter ones into b at the selected indices
        b.scatter_(1, topk_indices, 1.0)

        # Save input for use in backward
        ctx.save_for_backward(c)
        ctx.k = k
        ctx.epsilon = epsilon
        ctx.m = m
        return b

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        c, = ctx.saved_tensors
        k, eps, m = ctx.k, ctx.epsilon, ctx.m
        batch_size, n_items = c.shape

        # Prepare an accumulator for the gradient estimate
        grad_est = torch.zeros_like(c)
        # Monte Carlo finite-difference estimation
        for _ in range(m):
            # Sample Gaussian noise
            z = torch.randn_like(c)
            # Perturb the scores
            c_pert = c + eps * z
            # Compute top-k mask on perturbed scores
            b_pert = torch.zeros_like(c)
            _, idx = torch.topk(c_pert, k, dim=1)
            b_pert.scatter_(1, idx, 1.0)
            # Accumulate the directional derivatives
            grad_est += b_pert * z
        # Average over samples
        grad_est /= m

        # Chain-rule: incoming gradient times our estimated Jacobian
        return grad_output * grad_est, None, None, None


def differentiable_knapsack_layer(c, k, epsilon=0.1, m=10):
    """
    Convenience wrapper to apply the KnapsackLayerFunction.

    Args:
        c (Tensor): Input scores of shape (batch_size, n_items).
        k (int): Number of items to select per sample.
        epsilon (float): Perturbation magnitude for gradient estimation.
        m (int): Number of Monte Carlo samples.

    Returns:
        In forward pass:
        Tensor: A binary mask of shape (batch_size, n_items) indicating selected items.

        In backward pass:
        Tensor: Gradient of the input scores with respect to the selection mask.
    """
    return KnapsackLayerFunction.apply(c, k, epsilon, m)


class SelectionNetwork(nn.Module):
    """
    A simple feedforward network that produces selection scores for each base learner.

    Architecture:
        - Linear layer from input features to 128 units.
        - ReLU activation.
        - Linear layer from 128 units to `num_learners` scores.
    """
    def __init__(self, input_size, num_learners):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, num_learners)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # Returns raw scores (logits) for each learner
        return self.fc2(x)


def ensemble_forward(net, X, base_preds, k, epsilon, m):
    """
    Forward pass for the differentiable ensemble.

    Steps:
        1. Use `net` to produce scores `c` for each learner.
        2. L2-normalize scores across learners.
        3. Apply differentiable knapsack to choose top-k learners.
        4. Mask and combine base predictions from selected learners.
        5. Return softmax over class logits.

    Args:
        net (nn.Module): SelectionNetwork instance.
        X (Tensor): Input features (batch_size, input_size).
        base_preds (Tensor): Base model probability predictions
                             of shape (batch_size, n_learners, n_classes).
        k (int): Number of learners to select per sample.
        epsilon (float): Noise scale for knapsack gradient.
        m (int): Monte Carlo samples for knapsack gradient.

    Returns:
        Tensor: Ensemble output probabilities (batch_size, n_classes).
    """
    # Compute raw selection scores
    c = net(X)
    # Normalize scores to unit norm (stabilizes gradient estimation)
    c_norm = F.normalize(c, p=2, dim=1)
    # Differentiable top-k selection mask
    mask = differentiable_knapsack_layer(c_norm, k, epsilon, m)
    # Apply mask to base predictions (broadcasting)
    masked = base_preds * mask.unsqueeze(-1)
    # Sum selected predictions across learners
    summed = masked.sum(dim=1)
    # Convert to probabilities
    return F.softmax(summed, dim=1)


def get_base_predictions(X, learners):
    """
    Query a dictionary of scikit-learn style learners for their class probabilities.

    Args:
        X (DataFrame or array): Input features for prediction.
        learners (dict): Mapping from model name to model instance with `predict_proba`.

    Returns:
        np.ndarray: Array of shape (n_samples, n_learners, n_classes).
    """
    preds = [model.predict_proba(X) for model in learners.values()]
    # Stack over learners
    arr = np.stack(preds, axis=0)     # (n_learners, n_samples, n_classes)
    return arr.transpose(1, 0, 2)     # (n_samples, n_learners, n_classes)


def train_selection_for_k(learners, X_train, y_train, X_val, y_val,
                          input_size, k, num_epochs=10, epsilon=0.1, m=10, batch_size=32):
    """
    Train the SelectionNetwork for a fixed k and evaluate on validation set.

    Args:
        learners (dict): Specialized base models.
        X_train (DataFrame): Training features.
        y_train (Series): Training labels.
        X_val (DataFrame): Validation features.
        y_val (Series): Validation labels.
        input_size (int): Dimensionality of input features.
        k (int): Number of learners to select.
        num_epochs (int): Training epochs.
        epsilon (float): Noise scale.
        m (int): MC samples for gradient.
        batch_size (int): Mini-batch size.

    Returns:
        best_acc (float): Highest validation accuracy achieved.
        best_state (dict): Model state dict at best accuracy.
        train_losses (list): Training loss history per epoch.
        val_accs (list): Validation accuracy history per epoch.
    """
    # Convert data to torch tensors
    X_tr = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train.values, dtype=torch.long).to(device)
    X_vl = torch.tensor(X_val.values, dtype=torch.float32).to(device)
    y_vl = torch.tensor(y_val.values, dtype=torch.long).to(device)
    bp_tr = torch.tensor(get_base_predictions(X_train, learners),
                         dtype=torch.float32).to(device)
    bp_vl = torch.tensor(get_base_predictions(X_val, learners),
                         dtype=torch.float32).to(device)

    # Initialize network and optimizer
    net = SelectionNetwork(input_size, len(learners)).to(device)
    opt = optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    train_losses, val_accs = [], []
    best_acc, best_state = 0.0, None

    # Training loop
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}...")
        net.train()
        perm = torch.randperm(X_tr.size(0))  # Shuffle indices
        epoch_loss = 0.0

        # Mini-batch updates
        for i in range(0, X_tr.size(0), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            out = ensemble_forward(net, X_tr[idx], bp_tr[idx], k, epsilon, m)
            loss = loss_fn(out, y_tr[idx])
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * idx.size(0)

        avg_loss = epoch_loss / X_tr.size(0)
        train_losses.append(avg_loss)

        # Validation evaluation
        net.eval()
        with torch.no_grad():
            preds = ensemble_forward(net, X_vl, bp_vl, k, epsilon, m).argmax(dim=1)
            acc = (preds == y_vl).float().mean().item()
        val_accs.append(acc)

        # Track best model
        if acc > best_acc:
            best_acc, best_state = acc, copy.deepcopy(net.state_dict())

    return best_acc, best_state, train_losses, val_accs


def tune_selection_net(learners, X_train, y_train, X_val, y_val,
                       ks, input_size, num_epochs=10,
                       epsilon=0.1, m=1000, batch_size=512,
                       plot=False):
    """
    Iterate over a list of k values to find the optimal number of learners to select.

    Args:
        learners (dict): Base models.
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        ks (list): List of k values to try.
        input_size (int): Feature dimension.
        num_epochs (int): Epochs per k.
        epsilon, m: Knapsack noise params.
        batch_size (int): Batch size.
        plot (bool): Whether to plot training loss and validation accuracy.

    Returns:
        best_k (int): k with highest validation accuracy.
        best_overall (float): Best validation accuracy.
        best_model (dict): State dict of the best network.
        results (dict): Mapping k -> best_val_accuracy.
        loss_results (dict): Mapping k -> training loss history.
        acc_results (dict): Mapping k -> validation accuracy history.
    """
    best_overall, best_k, best_model = 0.0, None, None
    results, loss_results, acc_results = {}, {}, {}

    print("Tuning selection network...")
    for k in ks:
        print(f"Training for k={k}...")
        acc, state, losses, accs = train_selection_for_k(
            learners, X_train, y_train, X_val, y_val,
            input_size, k, num_epochs, epsilon, m, batch_size
        )
        results[k] = acc
        loss_results[k] = losses
        acc_results[k] = accs
        # Update global best
        if acc > best_overall:
            best_overall, best_k, best_model = acc, k, state

    # Optional plotting of curves
    if plot:
        epochs = range(1, num_epochs + 1)
        plt.figure(figsize=(10, 6))
        for k in ks:
            plt.plot(epochs, loss_results[k], marker='o', label=f"Train Loss (k={k})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss for Different k")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 6))
        for k in ks:
            plt.plot(epochs, acc_results[k], marker='x', linestyle='--', label=f"Val Accuracy (k={k})")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")
        plt.title("Validation Accuracy for Different k")
        plt.legend()
        plt.grid(True)
        plt.show()

    return best_k, best_overall, best_model, results, loss_results, acc_results


if __name__ == "__main__":
    # Example usage:
    # 1. Load your training/validation data and specialized learner models.
    # 2. Define the list of k values to search over.
    # 3. Call `tune_selection_net` with `plot=True` to visualize results.

    ks = [1, 2, 3, 4, 5, 6, 7]
    feature_dim = X_train.shape[1]
    best_k, best_acc, best_state, results, losses, accs = tune_selection_net(
        specialized_learners, X_train, y_train, X_val, y_val,
        ks, feature_dim, num_epochs=5, epsilon=0.1, m=100,
        batch_size=512, plot=True
    )
    print(f"Best k={best_k} with val accuracy={best_acc:.4f}")

    # Save the best model state for later use
    torch.save(best_state, "best_selection_net.pth")
