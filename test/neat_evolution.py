from tqdm import trange
from train import train_model, evaluate_model
from neat_genome import Genome
from visualize import visualize_genome, create_genome_evolution_gif  # ✅ Import GIF function
import random
import copy
import os

COMPLEXITY_PENALTY = 0.0005  
MUTATION_RATE = 0.9
ELITE_SIZE = 5

def fitness_function(params, model, X, y, genome):
    """Accuracy - complexity penalty."""
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
            child.mutate()  # Assuming Genome has a mutate() method
        new_population.append(child)
    return new_population

def evolve_population(X, y, pop_size=20, generations=10, epochs=300, task_name="experiment"):
    """
    1) Initialize population
    2) Train and evolve over multiple generations
    3) Save genome images per generation for animation
    4) Return best genome & fitness history
    """
    # ✅ Ensure directory for genome evolution images
    genome_evo_folder = os.path.join("plots", task_name.lower(), "genome_evolution")
    os.makedirs(genome_evo_folder, exist_ok=True)

    population = [Genome(num_inputs=X.shape[1], num_outputs=1) for _ in range(pop_size)]
    best_genome = None
    best_fitness = -999
    fitness_curve = []

    for gen in trange(generations, desc=f"Evolving {task_name}", ncols=80):
        fits = []
        for genome in population:
            f, p, m = evaluate_genome(genome, X, y, epochs=epochs)
            fits.append((f, genome, p, m))

        fits.sort(key=lambda x: x[0], reverse=True)
        top_f = fits[0][0]
        top_genome = fits[0][1]

        # ✅ Save best genome of this generation as an image
        genome_path = os.path.join(genome_evo_folder, f"gen_{gen}.png")
        visualize_genome(top_genome, save_path=genome_path)

        if top_f > best_fitness:
            best_fitness = top_f
            best_genome = top_genome

        fitness_curve.append(best_fitness)

        elites = [f[1] for f in fits[:ELITE_SIZE]]
        population = reproduce_and_mutate(elites, pop_size)

    # ✅ Generate GIF of genome evolution
    gif_path = os.path.join("plots", task_name.lower(), "genome_evolution.gif")
    create_genome_evolution_gif(genome_evo_folder, gif_path)

    print(f"🎬 Genome evolution GIF saved: {gif_path}")
    return best_genome, fitness_curve