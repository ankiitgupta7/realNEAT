# bp_neat_demo.py
import os
import jax
import jax.numpy as jnp
import numpy as np

from datasets import (
    generate_xor_data,
    generate_circle_data,
    generate_spiral_data
)
from neat_evolution import evolve_population
from train import train_model, evaluate_model
from visualize import (
    visualize_dataset,
    visualize_genome,
    visualize_decision_boundary
)

def run_experiment(
    name, 
    data_func, 
    data_range=(-1,1), 
    pop_size=20, 
    generations=10, 
    epochs=300
):
    """
    Full pipeline:
    1) Generate dataset
    2) Visualize dataset
    3) Evolve NEAT
    4) Visualize best genome
    5) Train best genome
    6) Visualize decision boundary (with train/test data + accuracies)
    """
    print(f"\n=== {name.upper()} EXPERIMENT ===")

    # 1) Generate Data (train/test split)
    X_train, y_train, X_test, y_test = data_func(n=200)
    print("Dataset shapes:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # Create a subfolder for each experiment
    exp_folder = os.path.join("plots", name.lower())
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder, exist_ok=True)

    # 2) Visualize dataset (train + test combined for a quick view)
    X_all = jnp.concatenate([X_train, X_test], axis=0)
    y_all = jnp.concatenate([y_train, y_test], axis=0)
    visualize_dataset(
        X_all, y_all, 
        save_path=os.path.join(exp_folder, f"{name}_dataset.png")
    )

    # 3) Evolve population (Backprop NEAT)
    best_genome, fitness_curve = evolve_population(
        X_train, y_train,
        pop_size=pop_size,
        generations=generations,
        epochs=epochs
    )
    print(f"Best genome after {generations} gens:", best_genome)
    print("Fitness curve:", fitness_curve)

    # 4) Visualize best genome architecture
    visualize_genome(
        best_genome,
        save_path=os.path.join(exp_folder, f"{name}_best_genome.png")
    )

    # 5) Convert best genome -> model and train with backprop
    trained_params, trained_model = train_model(
        best_genome, X_train, y_train, 
        learning_rate=0.01, 
        epochs=epochs
    )

    # 6) Visualize final decision boundary (train vs. test)
    visualize_decision_boundary(
        trained_params,
        trained_model,
        X_train, y_train,
        X_test, y_test,
        data_range=data_range,
        resolution=200,
        save_path=os.path.join(exp_folder, f"{name}_decision_boundary.png")
    )

    # Final train/test accuracy
    train_acc = evaluate_model(trained_params, trained_model, X_train, y_train)
    test_acc  = evaluate_model(trained_params, trained_model, X_test, y_test)
    print(f"{name} final train accuracy: {train_acc*100:.2f}%")
    print(f"{name} final test accuracy:  {test_acc*100:.2f}%")

if __name__ == "__main__":
    # XOR
    run_experiment(
        name="XOR",
        data_func=generate_xor_data,
        data_range=(-1, 1),
        pop_size=20,
        generations=10,
        epochs=300
    )

    # Circle
    run_experiment(
        name="CIRCLE",
        data_func=generate_circle_data,
        data_range=(-2, 2),
        pop_size=20,
        generations=10,
        epochs=300
    )

    # Spiral
    run_experiment(
        name="SPIRAL",
        data_func=generate_spiral_data,
        data_range=(-3, 3),  # or -2..2, up to you
        pop_size=20,
        generations=10,
        epochs=500
    )
