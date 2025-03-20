# bp-neat.py

import csv
import os
from datasets import generate_xor_data, generate_circle_data, generate_spiral_data
from neat_evolution import evolve_population
from train import train_model, evaluate_model
from visualize import visualize_dataset, visualize_decision_boundary, visualize_genome, visualize_fitness_history

def run_experiment(name, data_func, data_range=(-1,1), pop_size=50, generations=50, epochs=500):
    """Runs a full NEAT experiment: data generation, evolution, training, and visualization."""
    print(f"\n=== Running {name} Experiment ===")

    # 1) Generate dataset
    X_train, y_train, X_test, y_test = data_func(n=400)
    print("Dataset Generated. Shape:", X_train.shape, y_train.shape)

    # Visualize dataset with both train and test data
    dataset_plot_path = f"plots/{name.lower()}/{name}_dataset.png"
    visualize_dataset(X_train, y_train, X_test, y_test, save_path=dataset_plot_path)
    print(f"Dataset visualization saved to {dataset_plot_path}")
    

    # 2) Evolve population to get the best genome
    best_genome, best_fitnesses, avg_fitnesses, best_complexities = evolve_population(
        X_train, y_train, pop_size=pop_size, generations=generations, epochs=epochs, task_name=name
    )

    # 3) Visualize best genome architecture
    visualize_genome(best_genome, save_path=f"plots/{name.lower()}/{name}_best_genome.png")

    # 4) Save fitness & complexity history to CSV
    csv_path = f"plots/{name.lower()}/{name}_fitness_history.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Generation", "BestFitness", "AvgFitness", "BestComplexity"])
        for i in range(len(best_fitnesses)):
            writer.writerow([i, best_fitnesses[i], avg_fitnesses[i], best_complexities[i]])
    print(f"Saved fitness history to {csv_path}")

    # 5) Plot the fitness & complexity history
    fitness_plot_path = f"plots/{name.lower()}/{name}_fitness_history.png"
    visualize_fitness_history(
        best_fitnesses, avg_fitnesses, best_complexities,
        save_path=fitness_plot_path
    )
    print(f"Saved fitness plot to {fitness_plot_path}")

    # 6) Train the best genome with backpropagation
    trained_params, trained_model = train_model(best_genome, X_train, y_train, epochs=epochs)

    # 7) Plot decision boundary
    visualize_decision_boundary(
        trained_params, trained_model, X_train, y_train, X_test, y_test,
        data_range=data_range, resolution=200,
        save_path=f"plots/{name.lower()}/{name}_decision_boundary.png"
    )

    # 8) Evaluate model and print final accuracy
    final_acc = evaluate_model(trained_params, trained_model, X_test, y_test)
    print(f"{name} Final Test Accuracy: {final_acc:.4f}")

if __name__ == "__main__":
    run_experiment("SPIRAL", generate_spiral_data, (-3, 3))
    run_experiment("CIRCLE", generate_circle_data, (-2, 2))
    run_experiment("XOR", generate_xor_data, (-1, 1))
