"""
backprop_neat_jax.py

Backprop NEAT in JAX, with:
- Hidden node addition
- Weight/bias mutation
- Simple feed-forward building
- Complexity penalty
"""

import jax
import jax.numpy as jnp
import numpy as np
import random
from typing import List, Tuple

# -----------------------------------
# Hyperparameters & Defaults
# -----------------------------------
POPULATION_SIZE = 20
WEIGHT_INIT_SCALE = 0.1
MUTATION_RATE = 0.8
ADD_NODE_RATE = 0.03
ADD_CONN_RATE = 0.05
CROSSOVER_RATE = 0.3

# Weighted penalty for # of hidden nodes or # of connections
# Larger -> simpler networks are favored
COMPLEXITY_PENALTY = 0.0001

# -----------------------------------
# Genome Class
# -----------------------------------
class Genome:
    """
    A genome that holds:
    - node_ids: Unique IDs for input, hidden, output
    - connections: (in_node_id, out_node_id, weight, enabled)
    - activation: (string) specifying activation function for hidden nodes
    """
    def __init__(self, node_ids, connections, activation="relu"):
        self.node_ids = list(node_ids)  # keep as list
        self.connections = list(connections)
        self.activation = activation

    def copy(self):
        return Genome(
            node_ids=self.node_ids[:],
            connections=[(c[0], c[1], c[2], c[3]) for c in self.connections],
            activation=self.activation
        )

# -----------------------------------
# Activation
# -----------------------------------
def get_activation_fn(name):
    if name == "relu":
        return jax.nn.relu
    elif name == "tanh":
        return jnp.tanh
    elif name == "sigmoid":
        return jax.nn.sigmoid
    else:
        return jax.nn.relu

# -----------------------------------
# Build Feedforward
# -----------------------------------
def build_feedforward(genome: Genome, input_size, output_size):
    act_fn = get_activation_fn(genome.activation)

    input_nodes = [n for n in genome.node_ids if n.startswith("in")]
    output_nodes = [n for n in genome.node_ids if n.startswith("out")]
    hidden_nodes = [n for n in genome.node_ids if n.startswith("h")]

    adjacency = {n: [] for n in genome.node_ids}
    for (in_id, out_id, w, en) in genome.connections:
        if en:
            adjacency[out_id].append((in_id, w))

    def forward_fn(_, x):
        node_values = {n: jnp.zeros((x.shape[0],)) for n in genome.node_ids}  # Initialize all nodes

        for i, in_node in enumerate(input_nodes):
            node_values[in_node] = x[:, i]

        for h in hidden_nodes:
            if h in adjacency:
                inputs_sum = jnp.zeros((x.shape[0],))
                for (in_id, w) in adjacency[h]:
                    inputs_sum += node_values.get(in_id, 0) * w  # Use .get() to avoid KeyError
                node_values[h] = act_fn(inputs_sum)

        outputs = []
        for out_node in output_nodes:
            inputs_sum = jnp.zeros((x.shape[0],))
            for (in_id, w) in adjacency[out_node]:
                inputs_sum += node_values.get(in_id, 0) * w  # Use .get() to avoid KeyError
            outputs.append(inputs_sum)

        return jnp.stack(outputs, axis=-1)

    return forward_fn

# -----------------------------------
# Backprop Training
# -----------------------------------
def train_genome_backprop(genome: Genome,
                          X: jnp.ndarray,
                          y: jnp.ndarray,
                          steps=100,
                          learning_rate=1e-2):
    """
    Convert the connection list to trainable params, run gradient updates, store back.
    """

    def param_to_connections(param_vector, connections):
        idx = 0
        new_connections = []
        for (i_node, o_node, w, en) in connections:
            new_connections.append((i_node, o_node, param_vector[idx], en))
            idx += 1
        return new_connections

    init_params = []
    for c in genome.connections:
        init_params.append(c[2])
    init_params = jnp.array(init_params)

    def loss_fn(param_vector, X, y):
        # Rebuild
        updated_connections = param_to_connections(param_vector, genome.connections)
        temp_genome = genome.copy()
        temp_genome.connections = updated_connections
        f = build_feedforward(temp_genome, X.shape[1], 1)

        # forward
        logits = f(None, X)
        # apply sigmoid for binary classification
        preds = jax.nn.sigmoid(logits)  # shape (batch_size, 1)
        # BCE loss
        y_ = y.reshape(-1, 1)
        bce = -y_ * jnp.log(preds + 1e-7) - (1 - y_) * jnp.log(1 - preds + 1e-7)
        return jnp.mean(bce)

    grad_fn = jax.grad(loss_fn)

    param_vector = init_params
    for _ in range(steps):
        grads = grad_fn(param_vector, X, y)
        param_vector = param_vector - learning_rate * grads

    # Update the genome's connections
    final_connections = param_to_connections(param_vector, genome.connections)
    genome.connections = final_connections

# -----------------------------------
# Mutation
# -----------------------------------
def mutate(genome: Genome):
    """
    - Mutates connection weights
    - Adds a hidden node with sequential naming (h1, h2, ...)
    - Adds a new connection if needed
    """
    new_connections = []
    for (i_node, o_node, w, en) in genome.connections:
        if random.random() < MUTATION_RATE:
            w += np.random.normal(scale=0.1)
        new_connections.append((i_node, o_node, w, en))
    genome.connections = new_connections

    # Add a new node with a simple incremental name (h1, h2, ...)
    if random.random() < ADD_NODE_RATE and len(genome.connections) > 0:
        conn_idx = np.random.randint(len(genome.connections))
        in_id, out_id, w, en = genome.connections[conn_idx]
        if en:
            genome.connections[conn_idx] = (in_id, out_id, w, False)

            # Find the next available hidden node name
            existing_hidden_nodes = [n for n in genome.node_ids if n.startswith("h")]
            new_hid_id = f"h{len(existing_hidden_nodes) + 1}"  # Sequential naming

            genome.node_ids.append(new_hid_id)
            genome.connections.append((in_id, new_hid_id, 1.0, True))
            genome.connections.append((new_hid_id, out_id, w, True))

    # Add a new connection if needed
    if random.random() < ADD_CONN_RATE:
        possible_nodes = genome.node_ids[:]
        n1, n2 = random.sample(possible_nodes, 2)
        if n1 != n2 and not any((c[0] == n1 and c[1] == n2) for c in genome.connections):
            genome.connections.append((n1, n2, np.random.normal(), True))

# -----------------------------------
# Crossover
# -----------------------------------
def crossover(g1: Genome, g2: Genome):
    """
    Simple uniform crossover for demonstration.
    (Full NEAT does more advanced matching of connection genes.)
    We assume g1, g2 have the same node_ids for simplicity.
    """
    child = g1.copy()
    # length check
    min_len = min(len(g1.connections), len(g2.connections))
    for i in range(min_len):
        if random.random() < 0.5:
            child.connections[i] = g2.connections[i]
    return child

# -----------------------------------
# Evaluate with Complexity Penalty
# -----------------------------------
def evaluate_genome(genome: Genome, X: jnp.ndarray, y: jnp.ndarray):
    """
    Evaluate classification accuracy, then subtract a complexity penalty
    based on number of hidden nodes & connections.
    """
    # forward
    forward_fn = build_feedforward(genome, X.shape[1], 1)
    logits = forward_fn(None, X)
    preds = jax.nn.sigmoid(logits)
    pred_binary = (preds > 0.5).astype(jnp.float32)
    acc = jnp.mean((pred_binary == y.reshape(-1,1)).astype(jnp.float32))

    # Complexity penalty: # hidden nodes + # connections
    hidden_count = sum(1 for n in genome.node_ids if n.startswith("h"))
    connection_count = sum(1 for c in genome.connections if c[3] == True)
    penalty = COMPLEXITY_PENALTY * (hidden_count + connection_count)
    fitness = float(acc) - penalty
    return fitness

def raw_accuracy(genome: Genome, X: jnp.ndarray, y: jnp.ndarray):
    """Return pure accuracy (no penalty) for final reporting."""
    forward_fn = build_feedforward(genome, X.shape[1], 1)
    logits = forward_fn(None, X)
    preds = jax.nn.sigmoid(logits)
    pred_binary = (preds > 0.5).astype(jnp.float32)
    acc = jnp.mean((pred_binary == y.reshape(-1,1)).astype(jnp.float32))
    return float(acc)
