# neat_evolution.py
import os
import random
import copy
import numpy as np
from tqdm import trange
from neat_genome import Genome
from train import train_model, evaluate_model, update_genome_weights
from visualize import visualize_genome, create_genome_evolution_gif

COMPLEXITY_PENALTY = 0.0005  
MUTATION_RATE = 0.9
ELITE_SIZE = 5

def fitness_function(params, model, X_train, y_train, genome):
    """Compute fitness as training accuracy minus a penalty on complexity."""
    acc = evaluate_model(params, model, X_train, y_train)
    num_conn = sum(c.enabled for c in genome.connections)
    penalty = num_conn * COMPLEXITY_PENALTY
    return acc - penalty

def evaluate_genome(genome, X_train, y_train, X_test, y_test, epochs=300):
    """
    Train the genome and evaluate its performance on both train and test sets.
    Returns a dictionary with metrics.
    """
    trained_params, trained_model = train_model(genome, X_train, y_train, epochs=epochs)
    update_genome_weights(genome, trained_params)
    train_acc = evaluate_model(trained_params, trained_model, X_train, y_train)
    test_acc  = evaluate_model(trained_params, trained_model, X_test, y_test)
    fit = fitness_function(trained_params, trained_model, X_train, y_train, genome)
    complexity = sum(c.enabled for c in genome.connections)
    
    return {
        "fitness": fit,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "complexity": complexity,
        # We do not include trained_params and trained_model in the logs
        # "trained_params": trained_params,
        # "trained_model": trained_model
    }

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

def sanitize_metrics(metrics):
    """
    Remove non-serializable entries (e.g. trained_params, trained_model) from a metrics dict.
    In our current evaluate_genome, only serializable keys are present.
    This function exists in case additional keys are added later.
    """
    serializable = {k: v for k, v in metrics.items() if k in ["fitness", "train_acc", "test_acc", "complexity"]}
    return serializable

def evolve_population(X_train, y_train, X_test, y_test, pop_size=20, generations=10, epochs=300, task_name="experiment"):
    """
    1) Initialize population.
    2) For each generation, evaluate every genome on train and test sets.
       Log per-genome metrics (sanitized) and compute aggregated metrics.
    3) Save genome images per generation.
    4) Return:
         - overall best genome,
         - evolution_log: full sanitized metrics per generation and a final best summary,
         - aggregated_metrics: aggregated metrics per generation.
    """
    genome_evo_folder = os.path.join("plots", task_name.lower(), "genome_evolution")
    os.makedirs(genome_evo_folder, exist_ok=True)

    population = [Genome(num_inputs=X_train.shape[1], num_outputs=1) for _ in range(pop_size)]
    best_genome = None
    overall_best_fitness = -999

    evolution_log = []       # list of dicts: each with generation index and list of per-genome metrics
    aggregated_metrics = []  # list of dicts with aggregated metrics per generation

    for gen in trange(generations, desc=f"Evolving {task_name}", ncols=80):
        gen_log = []  # list to store sanitized metrics for each genome in this generation
        for genome in population:
            metrics = evaluate_genome(genome, X_train, y_train, X_test, y_test, epochs=epochs)
            gen_log.append(sanitize_metrics(metrics))
        evolution_log.append({"generation": gen, "genomes": gen_log})

        fitnesses = [m["fitness"] for m in gen_log]
        train_accs = [m["train_acc"] for m in gen_log]
        test_accs  = [m["test_acc"] for m in gen_log]
        complexities = [m["complexity"] for m in gen_log]

        agg = {
            "generation": gen,
            "avg_fitness": np.mean(fitnesses),
            "best_fitness": np.max(fitnesses),
            "avg_train_acc": np.mean(train_accs),
            "best_train_acc": np.max(train_accs),
            "avg_test_acc": np.mean(test_accs),
            "best_test_acc": np.max(test_accs),
            "avg_complexity": np.mean(complexities),
            "best_complexity": np.max(complexities)
        }
        aggregated_metrics.append(agg)

        # Determine best genome of this generation
        gen_best_metrics = max(gen_log, key=lambda m: m["fitness"])
        if gen_best_metrics["fitness"] > overall_best_fitness:
            overall_best_fitness = gen_best_metrics["fitness"]
            # For simplicity, we assume that the best genome is the one with matching complexity.
            best_genome = [g for g in population if sum(c.enabled for c in g.connections) == gen_best_metrics["complexity"]][0]

        # Save image of best genome of this generation
        genome_path = os.path.join(genome_evo_folder, f"gen_{gen}.png")
        visualize_genome(best_genome, save_path=genome_path)

        # Create next generation using elites based on fitness.
        population_sorted = sorted(population, key=lambda g: evaluate_genome(g, X_train, y_train, X_test, y_test, epochs=epochs)["fitness"], reverse=True)
        elites_genomes = population_sorted[:ELITE_SIZE]
        population = reproduce_and_mutate(elites_genomes, pop_size)

    # Append a final summary entry for the overall best genome.
    final_summary = {"final_best_genome": sanitize_metrics(evaluate_genome(best_genome, X_train, y_train, X_test, y_test, epochs=epochs))}
    evolution_log.append(final_summary)

    gif_path = os.path.join("plots", task_name.lower(), "genome_evolution.gif")
    create_genome_evolution_gif(genome_evo_folder, gif_path)

    return best_genome, evolution_log, aggregated_metrics
