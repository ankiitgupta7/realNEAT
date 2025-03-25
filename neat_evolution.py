from tqdm import trange
from train import train_model, evaluate_model
from neat_genome import Genome
from visualize import visualize_genome, create_genome_evolution_gif
import random
import copy
import os

COMPLEXITY_PENALTY = 0.001  
MUTATION_RATE = 0.9

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


def tournament_selection(fits, tournament_size=3):
    """
    Select a parent genome using tournament selection.
    'fits' is a list of tuples (fitness, genome, params, model).
    """
    tournament = random.sample(fits, tournament_size)
    # Sort tournament participants by fitness in descending order and select the best one
    tournament.sort(key=lambda x: x[0], reverse=True)
    return tournament[0][1]  # return the genome of the best participant

def reproduce_and_mutate_tournament(fits, pop_size, tournament_size=3):
    """
    Create a new population using tournament selection and mutation.
    'fits' is a list of tuples (fitness, genome, params, model).
    """
    new_population = []
    while len(new_population) < pop_size:
        parent = tournament_selection(fits, tournament_size)
        child = copy.deepcopy(parent)
        if random.random() < MUTATION_RATE:
            child.mutate()  # apply mutation
        new_population.append(child)
    return new_population

def evolve_population(X_train, y_train, X_test, y_test, pop_size, generations, epochs, task_name="experiment"):
    """
    1) Initialize population
    2) Train and evolve over multiple generations
    3) Save genome images per generation for animation
    4) Return best genome & fitness history
    """
    # Ensure directory for genome evolution images
    genome_evo_folder = os.path.join("plots", task_name.lower(), "genome_evolution")
    os.makedirs(genome_evo_folder, exist_ok=True)

    population = [Genome(num_inputs=X_train.shape[1], num_outputs=1) for _ in range(pop_size)]
    best_genome = None
    best_fitness = -999
    fitness_curve = []  # to log best fitness per generation
    test_acc_curve = []  # to log test accuracy per generation for best fitness genome
    train_acc_curve = []  # to log train accuracy per generation for best fitness genome
    connections_history = []  # to log number of connections per generation for best fitness genome

    for gen in trange(generations, desc=f"Evolving {task_name}", ncols=80):
        fits = []
        for genome in population:
            f, p, m = evaluate_genome(genome, X_train, y_train, epochs=epochs)
            fits.append((f, genome, p, m))

        # fits.sort(key=lambda x: x[0], reverse=True)
        # top_f = fits[0][0]
        # top_genome = fits[0][1]

        fits.sort(key=lambda x: x[0], reverse=True)
        top_f, top_genome, top_params, top_model = fits[0]

        # Save best genome of this generation as an image
        genome_path = os.path.join(genome_evo_folder, f"gen_{gen}.png")
        visualize_genome(top_genome, save_path=genome_path)

        if top_f > best_fitness:
            best_fitness = top_f
            best_genome = top_genome

            # also update the best params and model for this genome
            best_params = top_params
            best_model = top_model

        # Log fitness for top genome of this generation
        fitness_curve.append(top_f)

        num_connections = sum(c.enabled for c in top_genome.connections)
        connections_history.append(num_connections)
        

        # Log test accuracy for this generation
        test_acc = evaluate_model(top_params, top_model, X_test, y_test)
        train_acc = evaluate_model(top_params, top_model, X_train, y_train)
        print(f"Generation {gen}: Best Fitness = {top_f:.4f}")
        print(f"Train Accuracy = {train_acc:.4f}")
        print(f"Test Accuracy = {test_acc:.4f}")
        print(f"Number of Connections in Top Genome in this Gen = {num_connections}")

        test_acc_curve.append(test_acc)
        train_acc_curve.append(train_acc)

        population = reproduce_and_mutate_tournament(fits, pop_size, tournament_size=3)


    # Generate GIF of genome evolution
    gif_path = os.path.join("plots", task_name.lower(), "genome_evolution.gif")
    create_genome_evolution_gif(genome_evo_folder, gif_path)

    print(f"Genome evolution GIF saved: {gif_path}")
    return best_genome, best_params, best_model, fitness_curve, train_acc_curve, test_acc_curve, connections_history