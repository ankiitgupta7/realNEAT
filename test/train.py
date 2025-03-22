# train.py
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.linen import initializers as init
from neat_genome import Genome

class FeedForwardNN(nn.Module):
    input_size: int
    hidden_size: int
    output_size: int

    def setup(self):
        self.dense1 = nn.Dense(self.hidden_size, kernel_init=init.xavier_uniform())
        self.dense2 = nn.Dense(self.output_size, kernel_init=init.xavier_uniform())

    def __call__(self, x):
        x = nn.relu(self.dense1(x))
        x = nn.sigmoid(self.dense2(x))  # binary classification
        return x

def genome_to_nn(genome):
    # Force at least 4 hidden neurons to handle XOR, Spiral, etc.
    input_size = genome.num_inputs
    output_size = genome.num_outputs
    hidden_size = max(4, len([n for n in genome.nodes.values() if n.node_type == 'hidden']))
    return FeedForwardNN(input_size, hidden_size, output_size)

def loss_fn(params, model, X, y):
    logits = model.apply(params, X).squeeze()
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, y))

def train_step(params, X, y, opt_state, optimizer, model):
    loss, grads = jax.value_and_grad(loss_fn)(params, model, X, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

train_step_jit = jax.jit(train_step, static_argnums=(4, 5))

def train_model(genome, X, y, learning_rate=0.01, epochs=300):
    model = genome_to_nn(genome)
    rng = jax.random.PRNGKey(np.random.randint(1e9))
    params = model.init(rng, jnp.ones((1, genome.num_inputs)))
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    for _ in range(epochs):
        params, opt_state, loss = train_step_jit(params, X, y, opt_state, optimizer, model)
    return params, model

def evaluate_model(params, model, X, y):
    logits = model.apply(params, X).squeeze()
    preds = jnp.round(logits)
    accuracy = jnp.mean(preds == y)
    return float(accuracy)