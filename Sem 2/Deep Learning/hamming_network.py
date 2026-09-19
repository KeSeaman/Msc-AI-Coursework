"""
Hamming Network — Competitive Network for Binary Pattern Recognition
=====================================================================

A Hamming Network classifies binary input patterns by measuring their
similarity (via Hamming distance) to a set of stored prototype patterns.

Architecture (two layers):
  1. Feed-forward layer  – computes the correlation (inner product) between
     the bipolar (±1) input and each prototype.  The neuron whose prototype
     is closest to the input will have the highest initial activation.
  2. Recurrent (MAXNET) layer – iteratively suppresses all neurons except
     the winner through lateral inhibition until only one neuron remains
     active.

Example:
  We store four 5×5 bipolar pixel patterns (letters T, L, I, C) and then
  classify noisy versions of each letter.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# 1.  Define prototype patterns (5×5 bipolar)
# ──────────────────────────────────────────────
#   +1 = black pixel,  -1 = white pixel

patterns = {
    "T": np.array([
        [+1, +1, +1, +1, +1],
        [-1, -1, +1, -1, -1],
        [-1, -1, +1, -1, -1],
        [-1, -1, +1, -1, -1],
        [-1, -1, +1, -1, -1],
    ]),
    "L": np.array([
        [+1, -1, -1, -1, -1],
        [+1, -1, -1, -1, -1],
        [+1, -1, -1, -1, -1],
        [+1, -1, -1, -1, -1],
        [+1, +1, +1, +1, +1],
    ]),
    "I": np.array([
        [+1, +1, +1, +1, +1],
        [-1, -1, +1, -1, -1],
        [-1, -1, +1, -1, -1],
        [-1, -1, +1, -1, -1],
        [+1, +1, +1, +1, +1],
    ]),
    "C": np.array([
        [+1, +1, +1, +1, +1],
        [+1, -1, -1, -1, -1],
        [+1, -1, -1, -1, -1],
        [+1, -1, -1, -1, -1],
        [+1, +1, +1, +1, +1],
    ]),
}

# Flatten prototypes into row vectors (each row is one prototype)
proto_names = list(patterns.keys())
W = np.array([patterns[k].flatten() for k in proto_names], dtype=float)  # shape (4, 25)
n_protos, n_inputs = W.shape


# ──────────────────────────────────────────────
# 2.  Hamming-network functions
# ──────────────────────────────────────────────
def feedforward_layer(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Compute initial activations as the correlation between the input
    and each prototype.  For bipolar vectors the inner product equals
    n - 2·d_H, where d_H is the Hamming distance, so maximising the
    inner product minimises the Hamming distance.

    Returns initial activation vector of shape (n_prototypes,).
    """
    return W @ x  # shape (n_protos,)


def maxnet_layer(activations: np.ndarray, epsilon: float = 0.04,
                 max_iters: int = 100) -> np.ndarray:
    """
    Recurrent MAXNET: iteratively applies lateral inhibition so that
    the strongest neuron suppresses all others.

    Update rule:  a_i(t+1) = max(0,  a_i(t) - ε · Σ_{j≠i} a_j(t))

    Returns the final activation vector (only the winner is > 0).
    """
    a = activations.copy().astype(float)
    n = len(a)

    for iteration in range(max_iters):
        a_new = np.zeros(n)
        for i in range(n):
            inhibition = epsilon * (np.sum(a) - a[i])
            a_new[i] = max(0.0, a[i] - inhibition)
        # Stop if only one neuron is still active
        if np.count_nonzero(a_new) <= 1:
            a = a_new
            break
        a = a_new
    return a


def classify(x: np.ndarray, W: np.ndarray, proto_names: list) -> str:
    """Run the full Hamming network and return the winning class name."""
    initial = feedforward_layer(x, W)
    final = maxnet_layer(initial)
    winner = np.argmax(final)
    return proto_names[winner], initial, final


# ──────────────────────────────────────────────
# 3.  Add noise to a pattern
# ──────────────────────────────────────────────
def add_noise(pattern: np.ndarray, noise_frac: float = 0.2,
              rng: np.random.Generator = None) -> np.ndarray:
    """Flip a fraction of bits in a bipolar pattern."""
    if rng is None:
        rng = np.random.default_rng()
    noisy = pattern.flatten().copy()
    n_flip = int(noise_frac * len(noisy))
    flip_idx = rng.choice(len(noisy), size=n_flip, replace=False)
    noisy[flip_idx] *= -1
    return noisy


# ──────────────────────────────────────────────
# 4.  Run experiments and visualise
# ──────────────────────────────────────────────
rng = np.random.default_rng(42)
noise_level = 0.20  # flip 20 % of pixels
results = []  # store per-pattern results for reuse in the summary table

fig, axes = plt.subplots(len(proto_names), 3, figsize=(10, 12))
fig.suptitle("Hamming Network — Noisy Pattern Classification",
             fontsize=15, fontweight="bold", y=0.98)

for row, name in enumerate(proto_names):
    original = patterns[name]
    noisy = add_noise(original, noise_frac=noise_level, rng=rng)
    predicted, init_act, final_act = classify(noisy, W, proto_names)
    results.append((name, predicted, init_act))

    # --- original pattern ---
    axes[row, 0].imshow(original, cmap="gray_r", vmin=-1, vmax=1)
    axes[row, 0].set_title(f"Original '{name}'", fontsize=11)
    axes[row, 0].axis("off")

    # --- noisy pattern ---
    axes[row, 1].imshow(noisy.reshape(5, 5), cmap="gray_r", vmin=-1, vmax=1)
    axes[row, 1].set_title(f"Noisy ({int(noise_level*100)}% flipped)", fontsize=11)
    axes[row, 1].axis("off")

    # --- activations bar chart ---
    colors = ["#2ecc71" if n == predicted else "#bdc3c7" for n in proto_names]
    axes[row, 2].barh(proto_names, init_act, color=colors, edgecolor="black")
    axes[row, 2].set_title(f"Predicted → '{predicted}'", fontsize=11,
                            fontweight="bold",
                            color="#2ecc71" if predicted == name else "#e74c3c")
    axes[row, 2].set_xlabel("Correlation score")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("/home/captc/Devwork/Msc/DL/hamming_network_results.png", dpi=150)
plt.show()

# ──────────────────────────────────────────────
# 5.  Print summary table
# ──────────────────────────────────────────────
print("\n" + "=" * 55)
print("  Hamming Network Classification Results")
print("=" * 55)
print(f"  {'True':^6} {'Predicted':^10} {'Correct':^8} {'Scores'}")
print("-" * 55)

for name, predicted, init_act in results:
    status = "✓" if predicted == name else "✗"
    scores = "  ".join(f"{proto_names[i]}:{init_act[i]:+.0f}"
                       for i in range(len(proto_names)))
    print(f"  {name:^6} {predicted:^10} {status:^8} {scores}")

print("=" * 55)
