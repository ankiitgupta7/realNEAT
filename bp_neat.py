"""
backprop_neat_jax.py

A minimal NEAT + JAX backprop implementation.

Key Classes and Functions:
- Genome: Represents a single neural architecture (nodes + connections).
- Population: Manages a group of Genomes, applies NEAT operators (mutation, crossover).
- build_feedforward: Turns a Genome into a JAX function for forward pass.
- train_genome_backprop: Trains the weights of a Genome with JAX-based gradient descent.
- evaluate_genome: Evaluates a Genome's performance on a dataset.
"""

import jax
import jax.numpy as jnp
import numpy as np
import functools
from typing import List, Tuple
import random

# -----------------------------
# Hyperparameters for NEAT
# -----------------------------
POPULATION_SIZE = 50
MUTATION_RATE = 0.8
ADD_NODE_RATE = 0.03
ADD_CONN_RATE = 0.05
CROSSOVER_RATE = 0.3
WEIGHT_INIT_SCALE = 0.1

# -----------------------------
# Genome Class
# -----------------------------
class Genome:
    """
    Genome holds:
    - node_ids: list of node IDs (input, hidden, output)
    - connections: list of (in_node_id, out_node_id, weight, enabled)
    - activation: function used by these nodes (can be per-connection or per-layer)
      For simplicity, we store a single activation name and apply it to all hidden nodes.
    """
    def __init__(self, node_ids, connections, activation="relu"):
        self.node_ids = node_ids  # e.g. ["in1","in2",...,"out1","out2",...,"h1","h2"]
        self.connections = connections  # list of tuples: (in_id, out_id, weight, enabled)
        self.activation = activation

    def copy(self):
        return Genome(
            node_ids=self.node_ids[:],
            connections=[(c[0], c[1], c[2], c[3]) for c in self.connections],
            activation=self.activation
        )

# -----------------------------
# Activation Helpers
# -----------------------------
def get_activation_fn(name):
    if name == "relu":
        return jax.nn.relu
    elif name == "tanh":
        return jnp.tanh
    elif name == "sigmoid":
        return jax.nn.sigmoid
    else:
        return jax.nn.relu  # default

# -----------------------------
# Feedforward Network Builder
# -----------------------------
def build_feedforward(genome: Genome, input_size, output_size):
    """
    Build a feedforward function from the genome.
    For simplicity, we do a topological sort of the nodes and pass data
    from inputs to outputs ignoring any cycles.
    """
    act_fn = get_activation_fn(genome.activation)
    # 1) Identify input, hidden, output nodes
    input_nodes = [n for n in genome.node_ids if n.startswith("in")]
    output_nodes = [n for n in genome.node_ids if n.startswith("out")]
    hidden_nodes = [n for n in genome.node_ids if n.startswith("h")]

    # Let's create a dictionary that maps node -> function to compute its value
    # We'll store the adjacency in a dict: node -> list of (in_node, weight)
    adjacency = {n: [] for n in genome.node_ids}
    for (in_id, out_id, w, enabled) in genome.connections:
        if enabled:
            adjacency[out_id].append((in_id, w))

    # 2) We define a function that, given x (batch of inputs),
    # returns the output by forward-propagating through adjacency.

    def forward_fn(params, x):
        """
        params: We might store weights in the genome directly, but let's just rely on the
        connection list's weights for now. This is a minimal example.
        x: (batch_size, input_size)
        """
        # node_values is a dict from node_id -> (batch_size,) values
        node_values = {}

        # Assign input node values directly
        for i, in_node in enumerate(input_nodes):
            node_values[in_node] = x[:, i]

        # For hidden & output nodes, compute sum of inputs * weights, then apply activation
        # We'll do a simple topological ordering: input_nodes -> hidden_nodes -> output_nodes
        # (We assume no extremely complex connectivity for demonstration.)
        for h in hidden_nodes:
            sum_in = jnp.zeros(x.shape[0])
            for (in_id, w) in adjacency[h]:
                sum_in = sum_in + node_values[in_id] * w
            node_values[h] = act_fn(sum_in)

        # For output nodes, we’ll just do a linear transform + sigmoid or so
        # but let's apply the same activation for demonstration
        outs = []
        for out_node in output_nodes:
            sum_in = jnp.zeros(x.shape[0])
            for (in_id, w) in adjacency[out_node]:
                sum_in = sum_in + node_values[in_id] * w
            # We might do something like logistic regression for classification:
            # or we can keep the same activation
            # For classification, let's do e.g. "sigmoid" if it's a single output
            # or "softmax" if multiple. We'll keep it simple:
            outs.append(sum_in)  # no final activation here for example
        # shape = (batch_size, len(output_nodes))
        return jnp.stack(outs, axis=-1)

        # In practice, you'd refine for multi-output tasks, normalization, etc.

    return forward_fn

# -----------------------------
# Backprop Training
# -----------------------------
def train_genome_backprop(genome: Genome,
                          X_train: jnp.ndarray,
                          y_train: jnp.ndarray,
                          steps=100,
                          learning_rate=1e-2):
    """
    We'll treat the connection weights in the genome as trainable parameters.
    A simple approach: collect them in an array, do gradient descent, then put them back.
    """

    # 1) Convert the connection weights into a single 1D parameter vector
    # We'll store (index_in_connections -> weight)
    # Then we'll define a forward pass that reconstructs the weights from this param vector.
    def param_to_connections(param_vector, connections):
        idx = 0
        new_connections = []
        for (i_node, o_node, w, en) in connections:
            new_connections.append((i_node, o_node, param_vector[idx], en))
            idx += 1
        return new_connections

    init_params = []
    for c in genome.connections:
        init_params.append(c[2])  # weight
    init_params = jnp.array(init_params)

    def loss_fn(param_vector, X, y):
        # rebuild the connection list
        new_conns = param_to_connections(param_vector, genome.connections)
        # temporary genome
        temp_genome = Genome(node_ids=genome.node_ids, connections=new_conns, activation=genome.activation)
        f = build_feedforward(temp_genome, X.shape[1], 1)
        preds = f(None, X)
        # For a binary classification, let's use a logistic cross-entropy
        # preds shape: (batch_size, 1)
        # We'll apply sigmoid
        preds_sigmoid = jax.nn.sigmoid(preds)
        # y is (batch_size,1) or (batch_size,)
        y = y.reshape(-1, 1)
        bce = -y * jnp.log(preds_sigmoid + 1e-7) - (1 - y) * jnp.log(1 - preds_sigmoid + 1e-7)
        return jnp.mean(bce)

    grad_loss_fn = jax.grad(loss_fn)

    param_vector = init_params
    for _ in range(steps):
        grads = grad_loss_fn(param_vector, X_train, y_train)
        param_vector = param_vector - learning_rate * grads

    # put the final params back into the genome
    new_conns = param_to_connections(param_vector, genome.connections)
    genome.connections = new_conns

# -----------------------------
# NEAT Operators
# -----------------------------
def mutate(genome: Genome):
    """
    Mutate connection weights, possibly add new node or connection.
    """
    # Weight mutation
    new_connections = []
    for (i_node, o_node, w, en) in genome.connections:
        if random.random() < MUTATION_RATE:
            # random slight perturbation
            w += np.random.normal(scale=0.1)
        new_connections.append((i_node, o_node, w, en))
    genome.connections = new_connections

    # Add new node
    if random.random() < ADD_NODE_RATE and len(genome.connections) > 0:
        conn_idx = np.random.randint(len(genome.connections))
        in_id, out_id, w, en = genome.connections[conn_idx]
        if en:
            # disable old conn
            genome.connections[conn_idx] = (in_id, out_id, w, False)
            # create new node
            new_id = f"h{random.randint(0,999999)}"
            genome.node_ids.append(new_id)
            # connect in_id -> new_id with weight 1
            # connect new_id -> out_id with weight w
            genome.connections.append((in_id, new_id, 1.0, True))
            genome.connections.append((new_id, out_id, w, True))

    # Add new connection
    if random.random() < ADD_CONN_RATE:
        # pick two random nodes
        possible_nodes = genome.node_ids[:]
        n1 = random.choice(possible_nodes)
        n2 = random.choice(possible_nodes)
        if n1 != n2:
            # check if no existing connection
            if not any((c[0] == n1 and c[1] == n2) for c in genome.connections):
                genome.connections.append((n1, n2, np.random.normal(), True))

def crossover(genome1: Genome, genome2: Genome):
    """
    Simple 1-point or uniform crossover for demonstration.
    We assume both have the same node_ids for simplicity.
    In full NEAT, matching/mismatching genes are more carefully handled.
    """
    child = genome1.copy()
    for i in range(len(child.connections)):
        if i < len(genome2.connections) and random.random() < 0.5:
            child.connections[i] = genome2.connections[i]
    return child

# -----------------------------
# Population Class
# -----------------------------
class Population:
    def __init__(self, input_size, output_size, population_size=POPULATION_SIZE):
        self.input_size = input_size
        self.output_size = output_size
        self.genomes = []
        # Initialize population
        for _ in range(population_size):
            node_ids = [f"in{i}" for i in range(input_size)] + \
                       [f"out{o}" for o in range(output_size)]
            # connect inputs to outputs w/ random weights
            connections = []
            for i_in in range(input_size):
                for i_out in range(output_size):
                    connections.append((f"in{i_in}", f"out{i_out}",
                                        np.random.normal(scale=WEIGHT_INIT_SCALE), True))
            g = Genome(node_ids=node_ids, connections=connections, activation="relu")
            self.genomes.append(g)

# -----------------------------
# Evaluation / Fitness
# -----------------------------
def evaluate_genome(genome: Genome, X: jnp.ndarray, y: jnp.ndarray):
    """
    Evaluate the trained genome. We'll compute accuracy for demonstration.
    """
    f = build_feedforward(genome, X.shape[1], 1)
    logits = f(None, X)
    preds = jax.nn.sigmoid(logits)
    pred_binary = (preds > 0.5).astype(jnp.float32)
    acc = jnp.mean((pred_binary == y.reshape(-1,1)).astype(jnp.float32))
    return float(acc)
