# datasets.py
import numpy as np
import jax.numpy as jnp

def generate_xor_data(n=200, test_split=0.3, seed=42):
    """
    Generate XOR dataset (nonlinear).
    n: total samples
    test_split: fraction for test set
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1, 1, (n, 2))
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)

    # Shuffle & split
    idx = np.arange(n)
    rng.shuffle(idx)
    split = int(n * (1 - test_split))
    train_idx = idx[:split]
    test_idx  = idx[split:]

    return (jnp.array(X[train_idx]), jnp.array(y[train_idx]),
            jnp.array(X[test_idx]),  jnp.array(y[test_idx]))

def generate_circle_data(n=200, test_split=0.3, seed=42):
    """
    Generate two concentric circles.
    """
    rng = np.random.RandomState(seed)

    # Inner circle (radius ~0..1)
    r = np.sqrt(rng.uniform(0, 1, n//2))
    theta = rng.uniform(0, 2*np.pi, n//2)
    inner = np.c_[r*np.cos(theta), r*np.sin(theta)]
    labels_inner = np.zeros(n//2)

    # Outer circle (radius ~1..2)
    r = 1 + np.sqrt(rng.uniform(0, 1, n//2))
    theta = rng.uniform(0, 2*np.pi, n//2)
    outer = np.c_[r*np.cos(theta), r*np.sin(theta)]
    labels_outer = np.ones(n//2)

    X_ = np.vstack([inner, outer])
    y_ = np.hstack([labels_inner, labels_outer])

    # Shuffle & split
    idx = np.arange(n)
    rng.shuffle(idx)
    split = int(n * (1 - test_split))
    train_idx = idx[:split]
    test_idx  = idx[split:]

    return (jnp.array(X_[train_idx]), jnp.array(y_[train_idx]),
            jnp.array(X_[test_idx]),  jnp.array(y_[test_idx]))

def generate_spiral_data(n=200, test_split=0.3, noise=0.1, seed=42):
    """
    Generate two intertwined spirals.
    """
    rng = np.random.RandomState(seed)
    n_half = n // 2

    theta = np.linspace(0, 4*np.pi, n_half)
    r = np.linspace(0, 2, n_half)
    x1 = r * np.cos(theta) + rng.normal(0, noise, n_half)
    y1 = r * np.sin(theta) + rng.normal(0, noise, n_half)

    x2 = -r * np.cos(theta) + rng.normal(0, noise, n_half)
    y2 = -r * np.sin(theta) + rng.normal(0, noise, n_half)

    X_ = np.vstack([np.c_[x1, y1], np.c_[x2, y2]])
    y_ = np.hstack([np.zeros(n_half), np.ones(n_half)])

    # Shuffle & split
    idx = np.arange(n)
    rng.shuffle(idx)
    split = int(n * (1 - test_split))
    train_idx = idx[:split]
    test_idx  = idx[split:]

    return (jnp.array(X_[train_idx]), jnp.array(y_[train_idx]),
            jnp.array(X_[test_idx]),  jnp.array(y_[test_idx]))
