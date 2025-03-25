from datasets import generate_xor_data, generate_circle_data, generate_spiral_data
from neat_evolution import evolve_population
from train import train_model, evaluate_model
from visualize import visualize_decision_boundary, visualize_genome
import pandas as pd
import matplotlib.pyplot as plt

from datasets import polynomial_expand  # at the top


def run_experiment(name, data_func, data_range=(-1,1), pop_size=10, generations=10, epochs=300):
    """Runs a full NEAT experiment: data generation, evolution, training, and visualization."""
    print(f"\n=== Running {name} Experiment ===")
    
    # 1) Generate dataset
    X_train, y_train, X_test, y_test = data_func(n=200)
    print("Dataset Generated. Shape:", X_train.shape, y_train.shape)

    # 2) Evolve population to get the best genome
    best_genome, best_params, best_model, fitness_curve, train_acc_history, test_acc_history, connections_history = evolve_population(
        X_train, y_train, X_test, y_test, pop_size=pop_size, generations=generations, epochs=epochs, task_name=name
    )

    # 3) Visualize best genome architecture
    visualize_genome(best_genome, save_path=f"plots/{name.lower()}/{name}_best_genome.png")

    # 5) Plot decision boundary
    # visualize_decision_boundary(
    #     best_params, best_model, X_train, y_train, X_test, y_test,
    #     data_range=data_range, resolution=200, save_path=f"plots/{name.lower()}/{name}_decision_boundary.png"
    # )

    visualize_decision_boundary(
        best_params, best_model,
        X_train, y_train, X_test, y_test,
        data_range=data_range,
        resolution=200,
        save_path=f"plots/{name.lower()}/{name}_decision_boundary.png",
        feature_expand_fn=polynomial_expand if name.upper() == "SPIRAL" else None
    )

    # 6) Evaluate model and print final accuracy
    final_acc = evaluate_model(best_params, best_model, X_test, y_test)
    final_train_acc = evaluate_model(best_params, best_model, X_train, y_train)
    print(f"{name} Train Accuracy for Best Genome: {final_train_acc:.4f}")
    print(f"{name} Test Accuracy for Best Genmome: {final_acc:.4f}")

    # 7) Plot fitness curve
    plt.figure(figsize=(10, 4))
    
    # Plot fitness curve
    plt.subplot(1, 2, 1)
    plt.plot(connections_history, label="# of Connections in Top Genome")
    plt.xlabel("Generation")
    plt.ylabel("# of Connections")
    plt.title(f"{name} # of Connections Curve")
    plt.legend()
    plt.grid(True)

    # Plot accuracy histories
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label="Train Accuracy")
    plt.plot(test_acc_history, label="Test Accuracy")
    plt.plot(fitness_curve, label="Top Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Score")
    plt.title(f"{name} Genome Metrics History")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"plots/{name.lower()}/{name}_evolution_history.png")
    plt.close()

    # 8) Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Generation': range(len(fitness_curve)),
        'Fitness': fitness_curve,
        'Train_Accuracy': train_acc_history,
        'Test_Accuracy': test_acc_history,
        'Connections': connections_history
    })
    metrics_df.to_csv(f"plots/{name.lower()}/{name}_metrics.csv", index=False)


if __name__ == "__main__":
    # run_experiment("CIRCLE", generate_circle_data, (-2, 2))
    # run_experiment("XOR", generate_xor_data, (-1, 1))
    # Example: run experiment with expanded spiral features
    run_experiment(
        name="SPIRAL",
        data_func=lambda **kwargs: generate_spiral_data(expand_features=True, **kwargs),
        data_range=(-3, 3),
        pop_size=20,
        generations=100,
        epochs=300
    )

