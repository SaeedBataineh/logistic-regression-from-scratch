# Logistic regression from scratch (NumPy only) on the Pima Indians diabetes dataset.
# Run: python logistic_regression.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGURES = os.path.join(HERE, "figures")
os.makedirs(FIGURES, exist_ok=True)

PRINT_EVERY_BATCH = False


def split_X_Y(df):
    X = df.drop(columns=["Outcome"]).values.astype(float)
    Y = df["Outcome"].values.astype(float)
    return X, Y


train_df = pd.read_csv(os.path.join(DATA, "diabetes_train.csv"))
dev_df = pd.read_csv(os.path.join(DATA, "diabetes_dev.csv"))
test_df = pd.read_csv(os.path.join(DATA, "diabetes_test.csv"))

train_X, train_Y = split_X_Y(train_df)
val_X, val_Y = split_X_Y(dev_df)
test_X, test_Y = split_X_Y(test_df)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def accuracy(X, Y, w, b):
    p = sigmoid(X @ w + b)
    predictions = (p >= 0.5).astype(float)
    return np.mean(predictions == Y)


def compute_loss(X, Y, w, b, lam=0):
    p = sigmoid(X @ w + b)
    p = np.clip(p, 1e-15, 1 - 1e-15)
    loss = -np.mean(Y * np.log(p) + (1 - Y) * np.log(1 - p))
    return loss + (lam / 2) * np.sum(w ** 2)


def is_unstable(X, Y, w, b):
    p = sigmoid(X @ w + b)
    real_loss = -np.mean(Y * np.log(p) + (1 - Y) * np.log(1 - p))
    return np.any(p == 0) or not np.isfinite(real_loss)


def train(X, Y, vX, vY, lr, iterations, batch_size=16, lam=0, shuffle=False, seed=0, verbose=False):
    if shuffle:
        order = np.random.default_rng(seed).permutation(len(Y))
        X, Y = X[order], Y[order]

    w = np.zeros(X.shape[1])
    b = 0.0
    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": [], "unstable": []}
    unstable = False
    start = 0

    for it in range(1, iterations + 1):
        Xb = X[start:start + batch_size]
        Yb = Y[start:start + batch_size]
        start = start + batch_size
        if start >= len(Y):
            start = 0

        p = sigmoid(Xb @ w + b)
        error = p - Yb
        dw = Xb.T @ error / len(Yb) + lam * w
        db = np.mean(error)
        w = w - lr * dw
        b = b - lr * db

        if is_unstable(Xb, Yb, w, b) or is_unstable(X, Y, w, b):
            unstable = True

        history["train_acc"].append(accuracy(X, Y, w, b))
        history["val_acc"].append(accuracy(vX, vY, w, b))
        history["train_loss"].append(compute_loss(X, Y, w, b, lam))
        history["val_loss"].append(compute_loss(vX, vY, w, b, lam))
        history["unstable"].append(unstable)

        if verbose or PRINT_EVERY_BATCH:
            batch_loss = compute_loss(Xb, Yb, w, b, lam)
            print(f"iteration {it}: batch loss = {batch_loss:.4f}, train accuracy = {history['train_acc'][-1] * 100:.2f}%")

    return w, b, history


def min_max_normalize(train_X, val_X, test_X):
    mn = train_X.min(axis=0)
    mx = train_X.max(axis=0)
    return (train_X - mn) / (mx - mn), (val_X - mn) / (mx - mn), (test_X - mn) / (mx - mn)


def learning_rate_tables(X, Y, vX, vY, title):
    learning_rates = [10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    iteration_counts = [10, 100, 1000, 10000]
    train_table = pd.DataFrame(index=learning_rates, columns=iteration_counts)
    val_table = pd.DataFrame(index=learning_rates, columns=iteration_counts)
    unstable_lrs = []

    for lr in learning_rates:
        w, b, h = train(X, Y, vX, vY, lr, 10000)
        for n in iteration_counts:
            star = " *" if h["unstable"][n - 1] else ""
            train_table.loc[lr, n] = f"{h['train_acc'][n - 1] * 100:.2f}{star}"
            val_table.loc[lr, n] = f"{h['val_acc'][n - 1] * 100:.2f}{star}"
        if h["unstable"][-1]:
            unstable_lrs.append(lr)

    print(f"\nTraining accuracy (%) - {title}   (* = overflow / unstable)")
    print(train_table)
    print(f"\nValidation accuracy (%) - {title}")
    print(val_table)
    print("Learning rates that overflowed:", unstable_lrs)


def plot_curves(histories, kind, title, filename):
    fig, axes = plt.subplots(1, len(histories), figsize=(16, 4.5))
    for ax, lr in zip(axes, histories):
        h = histories[lr]
        x = range(1, len(h["train_" + kind]) + 1)
        ax.plot(x, h["train_" + kind], label="Training")
        ax.plot(x, h["val_" + kind], label="Validation")
        ax.set_title(f"learning rate = {lr}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Accuracy" if kind == "acc" else "Average loss")
        if kind == "loss":
            ax.set_yscale("log")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, filename), dpi=150)
    plt.close(fig)


print("\n===== Learning rate sweep (raw features) =====")
learning_rate_tables(train_X, train_Y, val_X, val_Y, "raw features")


print("\n===== Training curves (raw features) =====")
raw_curves = {}
for lr in [0.1, 0.001, 0.00001]:
    w, b, h = train(train_X, train_Y, val_X, val_Y, lr, 1000)
    raw_curves[lr] = h
    print(f"lr={lr}: final train acc={h['train_acc'][-1] * 100:.2f}%, val acc={h['val_acc'][-1] * 100:.2f}%, "
          f"train loss={h['train_loss'][-1]:.3f}, val loss={h['val_loss'][-1]:.3f}")
plot_curves(raw_curves, "acc", "Accuracy vs iterations (raw features)", "accuracy_raw.png")
plot_curves(raw_curves, "loss", "Average loss vs iterations (raw features)", "loss_raw.png")


print("\n===== Best model on raw features =====")
w, b, h = train(train_X, train_Y, val_X, val_Y, 0.0001, 100, verbose=True)
print("Best lr = 0.0001 with 100 iterations")
print(f"Train accuracy = {accuracy(train_X, train_Y, w, b) * 100:.2f}%")
print(f"Validation accuracy = {accuracy(val_X, val_Y, w, b) * 100:.2f}%")
print(f"Test accuracy = {accuracy(test_X, test_Y, w, b) * 100:.2f}%")


print("\n===== Batch size =====")
batch_sizes = [4, 8, 16, 32, 64]
final_val = []
average_val = []
for bs in batch_sizes:
    w, b, h = train(train_X, train_Y, val_X, val_Y, 0.01, 1000, batch_size=bs)
    final_val.append(h["val_acc"][-1])
    average_val.append(np.mean(h["val_acc"][-100:]))
    print(f"batch size {bs}: final val acc = {final_val[-1] * 100:.2f}%, "
          f"average val acc of last 100 iterations = {average_val[-1] * 100:.2f}%")
plt.figure(figsize=(7, 4.5))
plt.plot(batch_sizes, final_val, "o-", label="Validation accuracy at iteration 1000")
plt.plot(batch_sizes, average_val, "s--", label="Average validation accuracy (last 100 iterations)")
plt.xscale("log", base=2)
plt.xticks(batch_sizes, batch_sizes)
plt.xlabel("Batch size")
plt.ylabel("Validation accuracy")
plt.title("Validation accuracy vs batch size (lr = 0.01, 1000 iterations)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "batch_size.png"), dpi=150)
plt.close()


print("\n===== L2 regularization =====")
lambdas = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10]
train_accs = []
val_accs = []
for lam in lambdas:
    w, b, h = train(train_X, train_Y, val_X, val_Y, 0.01, 1000, lam=lam)
    train_accs.append(h["train_acc"][-1])
    val_accs.append(h["val_acc"][-1])
    print(f"lambda {lam}: train acc = {train_accs[-1] * 100:.2f}%, val acc = {val_accs[-1] * 100:.2f}%, "
          f"test acc = {accuracy(test_X, test_Y, w, b) * 100:.2f}%")
plt.figure(figsize=(7, 4.5))
plt.plot(range(len(lambdas)), train_accs, "o-", label="Training accuracy")
plt.plot(range(len(lambdas)), val_accs, "s-", label="Validation accuracy")
plt.xticks(range(len(lambdas)), lambdas)
plt.xlabel("lambda")
plt.ylabel("Accuracy")
plt.title("Accuracy vs L2 lambda (lr = 0.01, 1000 iterations)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "l2_lambda.png"), dpi=150)
plt.close()


print("\n===== Min-max normalization =====")
norm_train_X, norm_val_X, norm_test_X = min_max_normalize(train_X, val_X, test_X)
learning_rate_tables(norm_train_X, train_Y, norm_val_X, val_Y, "normalized features")
norm_curves = {}
for lr in [10, 1, 0.1]:
    w, b, h = train(norm_train_X, train_Y, norm_val_X, val_Y, lr, 1000)
    norm_curves[lr] = h
plot_curves(norm_curves, "acc", "Accuracy vs iterations (normalized features)", "accuracy_normalized.png")
plot_curves(norm_curves, "loss", "Average loss vs iterations (normalized features)", "loss_normalized.png")
w, b, h = train(norm_train_X, train_Y, norm_val_X, val_Y, 1, 1000)
print(f"Normalized, lr=1, 1000 iterations: train acc = {accuracy(norm_train_X, train_Y, w, b) * 100:.2f}%, "
      f"val acc = {accuracy(norm_val_X, val_Y, w, b) * 100:.2f}%, test acc = {accuracy(norm_test_X, test_Y, w, b) * 100:.2f}%")


print("\n===== Shuffling =====")
w, b, h = train(train_X, train_Y, val_X, val_Y, 0.01, 1000)
print(f"No shuffle: val acc = {h['val_acc'][-1] * 100:.2f}%")
for seed in [1, 2, 3]:
    w, b, h = train(train_X, train_Y, val_X, val_Y, 0.01, 1000, shuffle=True, seed=seed)
    print(f"Shuffle run {seed}: train acc = {h['train_acc'][-1] * 100:.2f}%, val acc = {h['val_acc'][-1] * 100:.2f}%")
for seed in [1, 2, 3]:
    w, b, h = train(norm_train_X, train_Y, norm_val_X, val_Y, 0.01, 1000, shuffle=True, seed=seed)
    print(f"Shuffle run {seed} (normalized features): val acc = {h['val_acc'][-1] * 100:.2f}%")
