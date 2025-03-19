"""
train_backprop_neat_jax.py

Example usage of the Backprop NEAT in JAX. Demonstrates:
- Generating toy classification datasets (Circle, XOR, Spiral).
- Evolving a population of networks via NEAT.
- Training each genome's weights using JAX backprop for a few epochs.
- Evaluating and selecting the best genome.
- (Optional) Saving or visualizing results.

Run:
  python train_backprop_neat_jax.py --task circle --generations 10
"""

import argparse
import numpy as np
import jax.numpy as jnp
from jax import random

from bp_neat import (Population,
                               train_genome_backprop,
                               evaluate_genome,
                               mutate,
                               crossover)

# -----------------------------
# Data Generators
# -----------------------------
def generate_circle_data(n=200, radius=1.0):
    X = np.random.uniform(-1.5, 1.5, size=(n, 2))
    y = np.array([1 if (x[0]**2 + x[1]**2) < radius**2 else 0 for x in X])
    return jnp.array(X, dtype=jnp.float32), jnp.array(y, dtype=jnp.float32)

def generate_xor_data(n=200):
    X = np.random.uniform(0, 1, size=(n, 2))
    y = np.array([1 if ((x[0] > 0.5) ^ (x[1] > 0.5)) else 0 for x in X])
    return jnp.array(X, dtype=jnp.float32), jnp.array(y, dtype=jnp.float32)

def generate_spiral_data(n=200, turns=1):
    """
    Two-spiral dataset (simplified).
    We'll generate points for two spirals, label them 0 or 1.
    """
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
    # shuffle
    idx = np.arange(n)
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]
    return jnp.array(X, dtype=jnp.float32), jnp.array(y, dtype=jnp.float32)

# -----------------------------
# Main training loop
# -----------------------------
def main(args):
    # 1) Generate data
    if args.task == "circle":
        X, y = generate_circle_data(n=400, radius=1.0)
    elif args.task == "xor":
        X, y = generate_xor_data(n=400)
    elif args.task == "spiral":
        X, y = generate_spiral_data(n=400, turns=1)
    else:
        raise ValueError("Unknown task: {}".format(args.task))

    input_size = X.shape[1]
    output_size = 1  # binary classification

    # 2) Initialize population
    population = Population(input_size, output_size, population_size=args.pop_size)

    # 3) Evolve for N generations
    for gen in range(args.generations):
        fitnesses = []
        # For each genome: train it w/ backprop, then evaluate
        for genome in population.genomes:
            # Train the genome
            train_genome_backprop(genome, X, y, steps=args.bp_steps, learning_rate=args.lr)
            # Evaluate
            acc = evaluate_genome(genome, X, y)
            fitnesses.append(acc)

        # Sort by fitness
        sorted_indices = np.argsort(fitnesses)[::-1]  # descending
        best_index = sorted_indices[0]
        best_acc = fitnesses[best_index]
        print(f"Generation {gen} | Best Accuracy: {best_acc:.3f}")

        # Breed new population (simple approach: top half survive, produce offspring)
        num_survivors = len(population.genomes)//2
        new_genomes = []
        for rank_idx in range(num_survivors):
            idx = sorted_indices[rank_idx]
            new_genomes.append(population.genomes[idx])
        # Reproduce
        while len(new_genomes) < len(population.genomes):
            parent1 = population.genomes[random.choice(random.PRNGKey(np.random.randint(100000)),
                                                       jnp.array(sorted_indices[:num_survivors]))]
            parent2 = population.genomes[random.choice(random.PRNGKey(np.random.randint(100000)),
                                                       jnp.array(sorted_indices[:num_survivors]))]
            child = crossover(parent1, parent2)
            mutate(child)
            new_genomes.append(child)
        population.genomes = new_genomes

    # Final best
    # Evaluate or visualize final best genome
    final_fitnesses = []
    for genome in population.genomes:
        acc = evaluate_genome(genome, X, y)
        final_fitnesses.append(acc)
    best_final_idx = np.argmax(final_fitnesses)
    print(f"Final Best Accuracy: {final_fitnesses[best_final_idx]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="circle", help="circle, xor, spiral")
    parser.add_argument("--generations", type=int, default=5, help="Number of NEAT generations")
    parser.add_argument("--pop_size", type=int, default=50, help="Population size")
    parser.add_argument("--bp_steps", type=int, default=100, help="Steps of backprop for each evaluation")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for backprop")
    args = parser.parse_args()

    main(args)
