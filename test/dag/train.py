# train.py
import jax
import jax.numpy as jnp
import numpy as np
import networkx as nx
import optax
from flax import linen as nn
from flax.linen import initializers as init
from flax import struct  # optional if you want a frozen dataclass
from neat_genome import Genome

# class FeedForwardNN(nn.Module):
#     input_size: int
#     hidden_size: int
#     output_size: int

#     def setup(self):
#         self.dense1 = nn.Dense(self.hidden_size, kernel_init=init.xavier_uniform())
#         self.dense2 = nn.Dense(self.output_size, kernel_init=init.xavier_uniform())

#     def __call__(self, x):
#         x = nn.relu(self.dense1(x))
#         x = nn.sigmoid(self.dense2(x))  # binary classification
#         return x

def extract_params_from_genome(genome):
    """
    Build a dict of {(in_node, out_node): weight_value} for all enabled connections.
    Returns a dictionary of jnp arrays that JAX can track/optimize.
    """
    param_dict = {}
    for conn in genome.connections:
        if conn.enabled:
            key = (conn.in_node, conn.out_node)
            param_dict[key] = jnp.array(conn.weight, dtype=jnp.float32)
    return param_dict

def dag_forward(params, X, genome):
    """
    params: dict {(in_node, out_node): weight_value}
    X: shape (batch, num_inputs)
    genome: for node info and topological sort
    Returns: jnp.array of shape (batch, genome.num_outputs)
    """
    # 1) Build a directed graph from enabled connections
    G = nx.DiGraph()
    for node_id in genome.nodes:
        G.add_node(node_id)
    for (in_n, out_n), w in params.items():
        G.add_edge(in_n, out_n)

    # 2) Remove edges that violate feed-forward ordering (heuristic: in_node >= out_node)
    edges_to_remove = [(u, v) for u, v in G.edges() if u >= v]
    G.remove_edges_from(edges_to_remove)

    # 3) If cycles still exist, remove one edge per cycle until acyclic.
    while not nx.is_directed_acyclic_graph(G):
        cycle = nx.find_cycle(G)
        G.remove_edge(cycle[0][0], cycle[0][1])

    topo_order = list(nx.topological_sort(G))

    # 4) Compute node outputs in topological order
    node_output = {}

    # Initialize input node outputs from X
    for i in range(genome.num_inputs):
        node_output[i] = X[:, i:i+1]  # each input node i gets its corresponding feature

    # For each non-input node, sum contributions from its predecessors
    for node_id in topo_order:
        if node_id < genome.num_inputs:
            continue  # already initialized

        total_in = jnp.zeros((X.shape[0], 1), dtype=jnp.float32)
        # Iterate only over the predecessors of this node in G
        for in_n in G.predecessors(node_id):
            # If the predecessor's output is available, add its contribution
            if in_n in node_output:
                total_in += node_output[in_n] * params[(in_n, node_id)]
        # Apply activation based on node type
        node_type = genome.nodes[node_id].node_type
        if node_type == 'hidden':
            node_output[node_id] = jnp.maximum(total_in, 0)  # ReLU
        elif node_type == 'output':
            node_output[node_id] = 1 / (1 + jnp.exp(-total_in))  # Sigmoid
        else:
            node_output[node_id] = total_in

    # Gather outputs for the output nodes
    outs = []
    for out_id in range(genome.num_inputs, genome.num_inputs + genome.num_outputs):
        outs.append(node_output[out_id])
    return jnp.concatenate(outs, axis=1)

# def genome_to_nn(genome):
#     # Force at least 4 hidden neurons to handle XOR, Spiral, etc.
#     input_size = genome.num_inputs
#     output_size = genome.num_outputs
#     hidden_size = max(4, len([n for n in genome.nodes.values() if n.node_type == 'hidden']))
#     return FeedForwardNN(input_size, hidden_size, output_size)

def loss_fn(params, genome, X, y):
    logits = dag_forward(params, X, genome).squeeze()  # shape (batch,)
    # Use small epsilon to avoid log(0)
    eps = 1e-7
    bce = - (y * jnp.log(logits + eps) + (1 - y) * jnp.log(1 - logits + eps))
    return jnp.mean(bce)

def train_step(params, X, y, opt_state, optimizer, model):
    loss, grads = jax.value_and_grad(loss_fn)(params, model, X, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

train_step_jit = jax.jit(train_step, static_argnums=(4, 5))

def train_model(genome, X, y, learning_rate=0.01, epochs=300):
    # 1) Extract param_dict from genome
    params = extract_params_from_genome(genome)

    # 2) Setup optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # 3) Training step
    def train_step(params, opt_state, X, y):
        l, grads = jax.value_and_grad(loss_fn)(params, genome, X, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, l

    # 4) Training loop
    for _ in range(epochs):
        params, opt_state, loss_val = train_step(params, opt_state, X, y)

    return params  # return the trained parameters

def update_genome_weights(genome, params):
    for c in genome.connections:
        if c.enabled:
            key = (c.in_node, c.out_node)
            if key in params:
                c.weight = float(params[key])


def evaluate_model(params, genome, X, y):
    logits = dag_forward(params, X, genome).squeeze()
    preds = jnp.round(logits)
    accuracy = jnp.mean(preds == y)
    return float(accuracy)

