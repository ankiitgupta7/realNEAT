# neat_evolution.py
import random
from tqdm import trange
from train import train_model, evaluate_model

COMPLEXITY_PENALTY = 0.0005  # tweak as you like
MUTATION_RATE = 0.9
ELITE_SIZE = 5

def fitness_function(params, model, X, y, genome):
    """Accuracy - complexity penalty."""
    acc = evaluate_model(params, model, X, y)
    # Count # of enabled connections
    num_conn = sum(c.enabled for c in genome.connections)
    penalty = num_conn * COMPLEXITY_PENALTY
    return acc - penalty

def evaluate_genome(genome, X, y, epochs=300):
    """Train + evaluate a genome, returning fitness, trained_params, trained_model."""
    trained_params, trained_model = train_model(genome, X, y, epochs=epochs)
    fit = fitness_function(trained_params, trained_model, X, y, genome)
    return fit, trained_params, trained_model

def reproduce_and_mutate(parents, pop_size):
    """Clones elites, mutates them, and refills population."""
    new_pop = []
    while len(new_pop) < pop_size:
        parent = random.choice(parents)
        # shallow copy
        child = clone_genome(parent)
        if random.random() < MUTATION_RATE:
            child.mutate()
        new_pop.append(child)
    return new_pop

def clone_genome(genome):
    from neat_genome import Genome
    g = Genome(genome.num_inputs, genome.num_outputs)
    g.nodes = dict(genome.nodes)
    g.connections = [c for c in genome.connections]
    g.innovation_history = [i for i in genome.innovation_history]
    g.next_node_id = genome.next_node_id
    return g

def evolve_population(X, y, pop_size=20, generations=10, epochs=300):
    """
    1) Init population
    2) For each generation:
       - Evaluate each genome (backprop)
       - Sort by fitness
       - Keep top elites
       - Reproduce & mutate
    Returns best genome + fitness curve
    """
    from neat_genome import Genome

    population = [Genome(num_inputs=X.shape[1], num_outputs=1) for _ in range(pop_size)]
    best_fit = -999
    best_genome = None
    fitness_curve = []

    for gen in trange(generations, desc="Evolving", ncols=80):
        fits = []
        for genome in population:
            f, p, m = evaluate_genome(genome, X, y, epochs=epochs)
            fits.append((f, genome, p, m))

        fits.sort(key=lambda x: x[0], reverse=True)
        top_f = fits[0][0]
        if top_f > best_fit:
            best_fit = top_f
            best_genome = fits[0][1]

        fitness_curve.append(best_fit)

        # Elitism
        elites = [f[1] for f in fits[:ELITE_SIZE]]
        population = reproduce_and_mutate(elites, pop_size)

    return best_genome, fitness_curve
