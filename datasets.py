import numpy as np
import jax.numpy as jnp

def polynomial_expand(X):
    """Expand 2D input X into multiple polynomial/trigonometric features."""
    x1 = X[:, 0]
    x2 = X[:, 1]
    x1_sq = x1**2
    x2_sq = x2**2
    x1x2  = x1 * x2
    sin_x1 = jnp.sin(x1)
    sin_x2 = jnp.sin(x2)
    cos_x1 = jnp.cos(x1)
    cos_x2 = jnp.cos(x2)

    # Stack them all into a single feature matrix
    X_aug = jnp.column_stack([
        x1,
        x2,
        x1_sq,
        x2_sq,
        x1x2,
        sin_x1,
        sin_x2,
        cos_x1,
        cos_x2
    ])
    return X_aug

def generate_xor_data(n=200, test_split=0.3, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1, 1, (n, 2))
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)

    idx = np.arange(n)
    rng.shuffle(idx)
    split = int(n * (1 - test_split))
    return (jnp.array(X[idx[:split]]), jnp.array(y[idx[:split]]),
            jnp.array(X[idx[split:]]), jnp.array(y[idx[split:]]))

def generate_circle_data(n=200, test_split=0.3, seed=42):
    rng = np.random.RandomState(seed)
    r = np.sqrt(rng.uniform(0, 1, n//2))
    theta = rng.uniform(0, 2*np.pi, n//2)
    inner = np.c_[r*np.cos(theta), r*np.sin(theta)]
    labels_inner = np.zeros(n//2)

    r = 1 + np.sqrt(rng.uniform(0, 1, n//2))
    theta = rng.uniform(0, 2*np.pi, n//2)
    outer = np.c_[r*np.cos(theta), r*np.sin(theta)]
    labels_outer = np.ones(n//2)

    X_ = np.vstack([inner, outer])
    y_ = np.hstack([labels_inner, labels_outer])

    idx = np.arange(n)
    rng.shuffle(idx)
    split = int(n * (1 - test_split))
    return (jnp.array(X_[idx[:split]]), jnp.array(y_[idx[:split]]),
            jnp.array(X_[idx[split:]]), jnp.array(y_[idx[split:]]))


def generate_spiral_data(n=200, test_split=0.3, noise=0.1, seed=42, expand_features=False):
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

    idx = np.arange(n)
    rng.shuffle(idx)
    split = int(n * (1 - test_split))

    # Convert to JAX arrays
    X_train = jnp.array(X_[idx[:split]])
    y_train = jnp.array(y_[idx[:split]])
    X_test  = jnp.array(X_[idx[split:]])
    y_test  = jnp.array(y_[idx[split:]])

    # If requested, expand each feature vector to include polynomial/trig terms
    if expand_features:
        X_train = polynomial_expand(X_train)
        X_test  = polynomial_expand(X_test)

    return X_train, y_train, X_test, y_test
