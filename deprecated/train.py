"""
train.py

Usage:
  python train.py --task circle --generations 10 --pop_size 50

Features:
- Trains a population via NEAT with JAX-based backprop for weights
- TQDM progress bar
- Complexity penalty
- Visualization each generation
- GIF creation for best-network evolution
"""

import argparse
import os
import numpy as np
import jax.numpy as jnp
import networkx as nx
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
import random

from bp_neat import (
    Genome, mutate, crossover, train_genome_backprop,
    evaluate_genome, raw_accuracy,
    POPULATION_SIZE
)

# -----------------------------------
# Data Generators
# -----------------------------------
def generate_circle_data(n=200, radius=1.0):
    X = np.random.uniform(-1.5, 1.5, size=(n, 2))
    y = np.array([1 if (x[0]**2 + x[1]**2) < radius**2 else 0 for x in X])
    return jnp.array(X, dtype=jnp.float32), jnp.array(y, dtype=jnp.float32)

def generate_xor_data(n=200):
    X = np.random.uniform(0, 1, size=(n, 2))
    y = np.array([1 if ((x[0] > 0.5) ^ (x[1] > 0.5)) else 0 for x in X])
    return jnp.array(X, dtype=jnp.float32), jnp.array(y, dtype=jnp.float32)

def generate_spiral_data(n=200, turns=1):
    n_class = n // 2
    theta = np.sqrt(np.random.rand(n_class,1)) * 2 * np.pi * turns
    r_a = 2*theta + np.pi
    data_a = np.concatenate([r_a*np.cos(theta), r_a*np.sin(theta)], axis=1)
    # swirl in the opposite direction
    theta_b = np.sqrt(np.random.rand(n_class,1)) * 2 * np.pi * turns
    r_b = -2*theta_b - np.pi
    data_b = np.concatenate([r_b*np.cos(theta_b), r_b*np.sin(theta_b)], axis=1)

    X = np.vstack([data_a, data_b])
    y = np.array([0]*n_class + [1]*n_class)
    idx = np.arange(n)
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]
    return jnp.array(X, dtype=jnp.float32), jnp.array(y, dtype=jnp.float32)

# -----------------------------------
# Population
# -----------------------------------
class Population:
    def __init__(self, input_size, output_size, population_size=50):
        self.input_size = input_size
        self.output_size = output_size
        self.genomes = []
        for _ in range(population_size):
            node_ids = [f"in{i}" for i in range(input_size)] + \
                       [f"out{o}" for o in range(output_size)]
            connections = []
            for i_in in range(input_size):
                for i_out in range(output_size):
                    w = np.random.normal(scale=0.1)
                    connections.append((f"in{i_in}", f"out{i_out}", w, True))
            g = Genome(node_ids, connections, activation="relu")
            self.genomes.append(g)

def visualize_genome(genome: Genome, save_path=None):
    """
    Draw the genome as a directed graph using networkx.
    """
    G = nx.DiGraph()
    # add nodes
    for n in genome.node_ids:
        G.add_node(n)
    # add edges
    for (i_node, o_node, w, en) in genome.connections:
        if en:
            G.add_edge(i_node, o_node, weight=w)

    # layout
    # separate layers: input at top, hidden middle, output bottom
    input_nodes = [n for n in genome.node_ids if n.startswith("in")]
    hidden_nodes = [n for n in genome.node_ids if n.startswith("h")]
    output_nodes = [n for n in genome.node_ids if n.startswith("out")]

    # positions for each node in a layered manner
    pos = {}
    # input layer
    for i, n in enumerate(input_nodes):
        pos[n] = (i, 2)
    # hidden
    for i, n in enumerate(hidden_nodes):
        pos[n] = (i, 1)
    # output
    for i, n in enumerate(output_nodes):
        pos[n] = (i, 0)

    plt.figure(figsize=(8,6))
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color='lightblue', arrows=True)
    # edge labels
    edge_labels = {(u, v): f"{d['weight']:.2f}" for (u, v, d) in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Genome Visualization")
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()

def main(args):
    # 1) Create data
    if args.task == 'circle':
        X, y = generate_circle_data(n=400, radius=1.0)
        print("Circle data generated")
    elif args.task == 'xor':
        X, y = generate_xor_data(n=400)
        print("XOR data generated")
    elif args.task == 'spiral':
        X, y = generate_spiral_data(n=400, turns=1)
        print("Spiral data generated")
    else:
        raise ValueError("Unknown task")

    # 2) Initialize population
    pop = Population(input_size=2, output_size=1, population_size=args.pop_size)

    # For GIF frames
    os.makedirs("viz_frames", exist_ok=True)
    frames = []

    # 3) Evolve
    for gen in tqdm(range(args.generations), desc="Evolving Generations"):
        # Evaluate + train each genome
        fitnesses = []
        for i, genome in enumerate(pop.genomes):
            # Train genome weights via backprop
            train_genome_backprop(genome, X, y, steps=args.bp_steps, learning_rate=args.lr)
            # Evaluate with complexity penalty
            fit = evaluate_genome(genome, X, y)
            fitnesses.append(fit)

        # track best
        sorted_indices = np.argsort(fitnesses)[::-1]  # descending
        best_idx = sorted_indices[0]
        best_fitness = fitnesses[best_idx]
        best_genome = pop.genomes[best_idx]
        # measure raw accuracy
        best_acc = raw_accuracy(best_genome, X, y)

        print(f"[Gen {gen}] Best Fit (with penalty) = {best_fitness:.4f}, Raw Acc = {best_acc:.4f}")

        # Visualization of best genome
        frame_path = f"viz_frames/gen_{gen}.png"
        visualize_genome(best_genome, save_path=frame_path)

        # GA-like next generation
        survivors_count = len(pop.genomes)//2
        new_genomes = []
        for rank_idx in range(survivors_count):
            idx = sorted_indices[rank_idx]
            new_genomes.append(pop.genomes[idx])

        # Fill the rest with offspring
        while len(new_genomes) < len(pop.genomes):
            parent1 = pop.genomes[random.choice(sorted_indices[:survivors_count])]
            parent2 = pop.genomes[random.choice(sorted_indices[:survivors_count])]
            child = crossover(parent1, parent2)
            mutate(child)
            new_genomes.append(child)

        pop.genomes = new_genomes

    # Final results
    final_fitnesses = []
    for g in pop.genomes:
        final_fitnesses.append(evaluate_genome(g, X, y))
    best_idx = np.argmax(final_fitnesses)
    final_best_genome = pop.genomes[best_idx]
    final_best_acc = raw_accuracy(final_best_genome, X, y)
    print(f"Final Best Raw Accuracy: {final_best_acc:.4f}")

    # Create GIF from frames
    try:
        images = []
        for gen in range(args.generations):
            frame_path = f"viz_frames/gen_{gen}.png"
            img = imageio.imread(frame_path)
            images.append(img)
        imageio.mimsave(f"best_genomes_{args.task}.gif", images, fps=1)
        print("GIF saved to best_genomes.gif")
    except Exception as e:
        print("Could not create GIF:", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="xor", help="circle|xor|spiral")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--pop_size", type=int, default=20)
    parser.add_argument("--bp_steps", type=int, default=100, help="Backprop steps per genome evaluation")
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()
    main(args)
