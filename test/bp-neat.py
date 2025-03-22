from datasets import generate_xor_data, generate_circle_data, generate_spiral_data
from neat_evolution import evolve_population
from train import train_model, evaluate_model
from visualize import visualize_decision_boundary, visualize_genome
import matplotlib.pyplot as plt

def run_experiment(name, data_func, data_range=(-1,1), pop_size=10, generations=10, epochs=1):
    """Runs a full NEAT experiment: data generation, evolution, training, and visualization."""
    print(f"\n=== Running {name} Experiment ===")
    
    # 1) Generate dataset
    X_train, y_train, X_test, y_test = data_func(n=200)
    print("Dataset Generated. Shape:", X_train.shape, y_train.shape)

    # 2) Evolve population to get the best genome
    best_genome, fitness_curve, train_acc_history, test_acc_history, connections_history = evolve_population(
        X_train, y_train, X_test, y_test, pop_size=pop_size, generations=generations, epochs=epochs, task_name=name
    )

    # 3) Visualize best genome architecture
    visualize_genome(best_genome, save_path=f"plots/{name.lower()}/{name}_best_genome.png")

    # 4) Train the best genome with backpropagation
    trained_params, trained_model = train_model(best_genome, X_train, y_train, epochs=epochs)

    # 5) Plot decision boundary
    visualize_decision_boundary(
        trained_params, trained_model, X_train, y_train, X_test, y_test,
        data_range=data_range, resolution=200, save_path=f"plots/{name.lower()}/{name}_decision_boundary.png"
    )

    # 6) Evaluate model and print final accuracy
    final_acc = evaluate_model(trained_params, trained_model, X_test, y_test)
    final_train_acc = evaluate_model(trained_params, trained_model, X_train, y_train)
    print(f"{name} Final Train Accuracy: {final_train_acc:.4f}")
    print(f"{name} Final Test Accuracy: {final_acc:.4f}")

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


if __name__ == "__main__":
    run_experiment("SPIRAL", generate_spiral_data, (-3, 3))
    run_experiment("CIRCLE", generate_circle_data, (-2, 2))
    run_experiment("XOR", generate_xor_data, (-1, 1))
