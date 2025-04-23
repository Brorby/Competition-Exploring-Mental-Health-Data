import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Differentiable Knapsack Layer
# ------------------------------
class KnapsackLayerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, c, k, epsilon, m):
        b = torch.zeros_like(c)
        _, topk_indices = torch.topk(c, k, dim=1)
        b.scatter_(1, topk_indices, 1.0)
        ctx.save_for_backward(c)
        ctx.k = k
        ctx.epsilon = epsilon
        ctx.m = m
        return b

    @staticmethod
    def backward(ctx, grad_output):
        c, = ctx.saved_tensors
        k, eps, m = ctx.k, ctx.epsilon, ctx.m
        grad_est = torch.zeros_like(c)
        for _ in range(m):
            z = torch.randn_like(c)
            c_pert = c + eps * z
            b_pert = torch.zeros_like(c)
            _, idx = torch.topk(c_pert, k, dim=1)
            b_pert.scatter_(1, idx, 1.0)
            grad_est += b_pert * z
        grad_est /= m
        return grad_output * grad_est, None, None, None


def differentiable_knapsack_layer(c, k, epsilon=0.1, m=10):
    return KnapsackLayerFunction.apply(c, k, epsilon, m)

# ------------------------------
# Selection Network
# ------------------------------
class SelectionNetwork(nn.Module):
    def __init__(self, input_size, num_learners):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, num_learners)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ------------------------------
# Ensemble Forward
# ------------------------------
def ensemble_forward(net, X, base_preds, k, epsilon, m):
    c = net(X)
    c_norm = F.normalize(c, p=2, dim=1)
    mask = differentiable_knapsack_layer(c_norm, k, epsilon, m)
    masked = base_preds * mask.unsqueeze(-1)
    summed = masked.sum(dim=1)
    return F.softmax(summed, dim=1)

# ------------------------------
# Base Predictions
# ------------------------------
def get_base_predictions(X, learners):
    preds = [
        model.forward(X, 20) if "greedy" in name
        else model.predict_proba(X)
        for name, model in learners.items()
]
    arr = np.stack(preds, axis=0)
    return np.transpose(arr, (1,0,2))

# ------------------------------
# Training for a single k (records losses/accuracies)
# ------------------------------
def train_selection_for_k(learners, X_train, y_train, X_val, y_val,
                          input_size, k, num_epochs=10, epsilon=0.1, m=10, batch_size=32):
    # Prepare data tensors
    X_tr = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train.values, dtype=torch.long).to(device)
    X_vl = torch.tensor(X_val.values, dtype=torch.float32).to(device)
    y_vl = torch.tensor(y_val.values, dtype=torch.long).to(device)
    bp_tr = torch.tensor(get_base_predictions(X_train, learners), dtype=torch.float32).to(device)
    bp_vl = torch.tensor(get_base_predictions(X_val, learners), dtype=torch.float32).to(device)

    net = SelectionNetwork(input_size, len(learners)).to(device)
    opt = optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    train_losses, val_accs = [], []
    best_acc, best_state = 0.0, None

    for epoch in range(1, num_epochs+1):
        net.train()
        perm = torch.randperm(X_tr.size(0))
        epoch_loss = 0.0
        for i in range(0, X_tr.size(0), batch_size):
            idx = perm[i:i+batch_size]
            opt.zero_grad()
            out = ensemble_forward(net, X_tr[idx], bp_tr[idx], k, epsilon, m)
            loss = loss_fn(out, y_tr[idx])
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * idx.size(0)
        avg_loss = epoch_loss / X_tr.size(0)
        train_losses.append(avg_loss)

        # Validation
        net.eval()
        with torch.no_grad():
            preds = ensemble_forward(net, X_vl, bp_vl, k, epsilon, m).argmax(dim=1)
            acc = (preds == y_vl).float().mean().item()
        val_accs.append(acc)

        # Save best
        if acc > best_acc:
            best_acc, best_state = acc, copy.deepcopy(net.state_dict())

    return best_acc, best_state, train_losses, val_accs

# ------------------------------
# Tuning across ks with optional plotting
# ------------------------------
def tune_selection_net(learners, X_train, y_train, X_val, y_val,
                       ks, input_size, num_epochs=10,
                       epsilon=0.1, m=10, batch_size=32,
                       plot=False):
    best_overall, best_k, best_model = 0.0, None, None
    results, loss_results, acc_results = {}, {}, {}
    print("Tuning selection network...")
    for k in ks:
        acc, state, losses, accs = train_selection_for_k(
            learners, X_train, y_train, X_val, y_val,
            input_size, k, num_epochs, epsilon, m, batch_size
        )
        results[k] = acc
        loss_results[k] = losses
        acc_results[k] = accs
        if acc > best_overall:
            best_overall, best_k, best_model = acc, k, state

    if plot:
        epochs = range(1, num_epochs+1)
        plt.figure(figsize=(10,6))
        for k in ks:
            plt.plot(epochs, loss_results[k], marker='o', label=f"Train Loss (k={k})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss for Different k")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10,6))
        for k in ks:
            plt.plot(epochs, acc_results[k], marker='x', linestyle='--', label=f"Val Accuracy (k={k})")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")
        plt.title("Validation Accuracy for Different k")
        plt.legend()
        plt.grid(True)
        plt.show()

    return best_k, best_overall, best_model, results, loss_results, acc_results

# ------------------------------
# Example CLI Usage
# ------------------------------
if __name__ == "__main__":
    # Import data & learners, then:
    ks = [1,2,3,4,5,6,7]
    feature_dim = X_train.shape[1]
    best_k, best_acc, best_state, results, losses, accs = tune_selection_net(
        specialized_learners, X_train, y_train, X_val, y_val,
        ks, feature_dim, num_epochs=5, epsilon=0.1, m=100,
        batch_size=64, plot=True
    )
    print(f"Best k={best_k} with val accuracy={best_acc:.4f}")
    torch.save(best_state, "best_selection_net.pth")
