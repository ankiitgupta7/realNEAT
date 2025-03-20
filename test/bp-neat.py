# bp-neat.py
import csv
import os
import json
from datasets import generate_xor_data, generate_circle_data, generate_spiral_data
from neat_evolution import evolve_population
from train import train_model, evaluate_model
from visualize import visualize_dataset, visualize_decision_boundary, visualize_genome, visualize_fitness_history

def run_experiment(name, data_func, data_range=(-1,1), pop_size=10, generations=5, epochs=100):
    """Runs a full NEAT experiment: data generation, evolution, training, and visualization."""
    print(f"\n=== Running {name} Experiment ===")

    # 1) Generate dataset
    X_train, y_train, X_test, y_test = data_func(n=400)
    print("Dataset Generated. Shape:", X_train.shape, y_train.shape)

    # Visualize dataset
    dataset_plot_path = f"plots/{name.lower()}/{name}_dataset.png"
    visualize_dataset(X_train, y_train, X_test, y_test, save_path=dataset_plot_path)
    print(f"Dataset visualization saved to {dataset_plot_path}")
    
    # 2) Evolve population with full logging
    best_genome, evolution_log, aggregated_metrics = evolve_population(
        X_train, y_train, X_test, y_test, pop_size=pop_size, generations=generations, epochs=epochs, task_name=name
    )

    # 3) Visualize best genome architecture
    visualize_genome(best_genome, save_path=f"plots/{name.lower()}/{name}_best_genome.png")

    # 4) Save aggregated metrics to a single CSV
    csv_path = f"plots/{name.lower()}/{name}_aggregated_metrics.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Generation", "AvgFitness", "BestFitness", "AvgTrainAcc", "BestTrainAcc", "AvgTestAcc", "BestTestAcc", "AvgComplexity", "BestComplexity"])
        for agg in aggregated_metrics:
            writer.writerow([
                agg["generation"],
                agg["avg_fitness"],
                agg["best_fitness"],
                agg["avg_train_acc"],
                agg["best_train_acc"],
                agg["avg_test_acc"],
                agg["best_test_acc"],
                agg["avg_complexity"],
                agg["best_complexity"]
            ])
    print(f"Saved aggregated metrics to {csv_path}")

    # 5) Save full evolution log (without non-serializable entries) to JSON
    json_path = f"plots/{name.lower()}/{name}_evolution_log.json"
    with open(json_path, "w") as f:
        json.dump(evolution_log, f, indent=4)
    print(f"Saved evolution log to {json_path}")

    # 6) Plot fitness & complexity history using aggregated metrics
    fitness_plot_path = f"plots/{name.lower()}/{name}_fitness_history.png"
    best_fitnesses = [agg["best_fitness"] for agg in aggregated_metrics]
    avg_fitnesses = [agg["avg_fitness"] for agg in aggregated_metrics]
    best_complexities = [agg["best_complexity"] for agg in aggregated_metrics]
    visualize_fitness_history(best_fitnesses, avg_fitnesses, best_complexities, save_path=fitness_plot_path)
    print(f"Saved fitness plot to {fitness_plot_path}")

    # 7) Train the best genome with backpropagation on the training set
    trained_params, trained_model = train_model(best_genome, X_train, y_train, epochs=epochs)

    # 8) Plot decision boundary
    visualize_decision_boundary(
        trained_params, trained_model, X_train, y_train, X_test, y_test,
        data_range=data_range, resolution=200,
        save_path=f"plots/{name.lower()}/{name}_decision_boundary.png"
    )

    # 9) Evaluate final test accuracy
    final_acc = evaluate_model(trained_params, trained_model, X_test, y_test)
    print(f"{name} Final Test Accuracy: {final_acc:.4f}")

if __name__ == "__main__":
    run_experiment("SPIRAL", generate_spiral_data, (-3, 3))
    run_experiment("CIRCLE", generate_circle_data, (-2, 2))
    run_experiment("XOR", generate_xor_data, (-1, 1))
