import os
import random
import copy
import numpy as np
from tqdm import trange
from train import train_model, evaluate_model
from neat_genome import Genome
from visualize import visualize_genome, create_genome_evolution_gif

COMPLEXITY_PENALTY = 0.005  
MUTATION_RATE = 0.9
ELITE_SIZE = 5

def fitness_function(params, model, X, y, genome):
    """Accuracy minus complexity penalty."""
    acc = evaluate_model(params, model, X, y)
    num_conn = sum(c.enabled for c in genome.connections)
    penalty = num_conn * COMPLEXITY_PENALTY
    return acc - penalty

def evaluate_genome(genome, X, y, epochs=300):
    """Train + evaluate a genome, returning fitness, trained_params, trained_model."""
    trained_params, trained_model = train_model(genome, X, y, epochs=epochs)
    fit = fitness_function(trained_params, trained_model, X, y, genome)
    return fit, trained_params, trained_model

def reproduce_and_mutate(elites, pop_size):
    """Create a new population by mutating the best genomes."""
    new_population = []
    while len(new_population) < pop_size:
        parent = random.choice(elites)
        child = copy.deepcopy(parent)
        if random.random() < MUTATION_RATE:
            child.mutate()
        new_population.append(child)
    return new_population

def evolve_population(X, y, pop_size=20, generations=10, epochs=300, task_name="experiment"):
    """
    1) Initialize population
    2) Train and evolve over multiple generations
    3) Save genome images per generation for animation
    4) Return best genome & fitness history
    """
    # Create a folder to store genome images per generation
    genome_evo_folder = os.path.join("plots", task_name.lower(), "genome_evolution")
    os.makedirs(genome_evo_folder, exist_ok=True)

    # Initialize population
    population = [Genome(num_inputs=X.shape[1], num_outputs=1) for _ in range(pop_size)]
    best_genome = None
    best_fitness = -999

    # Lists to store metrics per generation
    best_fitnesses = []
    avg_fitnesses = []
    best_complexities = []

    for gen in trange(generations, desc=f"Evolving {task_name}", ncols=80):
        # Evaluate each genome
        fits = []
        for genome in population:
            f, p, m = evaluate_genome(genome, X, y, epochs=epochs)
            fits.append((f, genome, p, m))

        # Sort by fitness descending
        fits.sort(key=lambda x: x[0], reverse=True)

        # Best of this generation
        top_f = fits[0][0]
        top_genome = fits[0][1]

        # Save an image of the best genome in this generation
        genome_path = os.path.join(genome_evo_folder, f"gen_{gen}.png")
        visualize_genome(top_genome, save_path=genome_path)

        # If this generation's best is better than the overall best, update
        if top_f > best_fitness:
            best_fitness = top_f
            best_genome = top_genome

        # Compute average fitness
        avg_f = np.mean([f_[0] for f_ in fits])

        # Complexity measure = number of enabled connections in the top genome
        complexity = sum(c.enabled for c in top_genome.connections)

        # Store metrics
        best_fitnesses.append(top_f)
        avg_fitnesses.append(avg_f)
        best_complexities.append(complexity)

        # # Print summary for this generation
        # print(f"\nGeneration {gen}: Best Fitness = {top_f:.4f}, Avg Fitness = {avg_f:.4f}, Best Complexity = {complexity}")
        # print("Population details:")
        # for i, (f_val, genome, _, _) in enumerate(fits):
        #     comp = sum(c.enabled for c in genome.connections)
        #     print(f"   Genome {i}: Fitness = {f_val:.4f}, Complexity = {comp}")


        # Create next generation
        elites = [f_[1] for f_ in fits[:ELITE_SIZE]]
        population = reproduce_and_mutate(elites, pop_size)

    # Generate GIF of genome evolution
    gif_path = os.path.join("plots", task_name.lower(), "genome_evolution.gif")
    create_genome_evolution_gif(genome_evo_folder, gif_path)
    print(f"🎬 Genome evolution GIF saved: {gif_path}")

    # Return the best genome plus the fitness/complexity history
    return best_genome, best_fitnesses, avg_fitnesses, best_complexities
