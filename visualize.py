# visualize.py
import os
import jax.numpy as jnp
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # disable interactive GUI
import matplotlib.pyplot as plt

def _ensure_dir(path):
    if path and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)

def visualize_dataset(X, y, save_path=None):
    """
    Plot the dataset points (X,y). Saves a PNG (no display).
    """
    plt.figure(figsize=(5, 5))
    plt.scatter(X[:,0], X[:,1], c=y, edgecolors='k', cmap='bwr')
    plt.title("Dataset")
    plt.xlabel("x1")
    plt.ylabel("x2")
    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=300)
    plt.close()

def visualize_genome(genome, save_path=None):
    """
    Show NEAT genome via NetworkX, color-coded by node type.
    Saves to file (no display).
    """
    plt.figure(figsize=(5, 5))
    G = nx.DiGraph()

    # Add nodes
    for nid, node_gene in genome.nodes.items():
        G.add_node(nid, type=node_gene.node_type)

    # Add edges
    for c in genome.connections:
        if c.enabled:
            G.add_edge(c.in_node, c.out_node, weight=c.weight)

    pos = nx.spring_layout(G, seed=42)
    node_colors = []
    for node_id in G.nodes():
        t = G.nodes[node_id]['type']
        if t == 'input':
            node_colors.append('lightblue')
        elif t == 'output':
            node_colors.append('lightgreen')
        else:
            node_colors.append('lightgray')

    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=800, font_size=10)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    edge_labels_rounded = {k: f"{v:.2f}" for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_rounded, font_color='red')

    plt.title(f"Genome (nodes={len(G.nodes())}, edges={len(G.edges())})")

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=300)
    plt.close()


def visualize_decision_boundary(
    params, model, X_train, y_train, X_test, y_test, 
    data_range=(-1,1), resolution=200, save_path=None
):
    """
    Fancy contour-based decision boundary + data overlay + train/test accuracy annotation.
    """
    x_min, x_max = data_range
    # We'll define y_min, y_max = data_range, assuming we want the same min/max for x and y.
    y_min, y_max = data_range

    xs = jnp.linspace(x_min, x_max, resolution)
    ys = jnp.linspace(y_min, y_max, resolution)

    xx, yy = jnp.meshgrid(xs, ys)
    grid = jnp.column_stack([xx.ravel(), yy.ravel()])  # shape (resolution^2, 2)

    logits = model.apply(params, grid).squeeze()  # shape (res^2,)
    preds = (logits > 0.5).astype(int)

    # Reshape for contour plotting
    logits_np = logits.reshape((resolution, resolution))

    import matplotlib
    matplotlib.use('Agg')  # No GUI
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(6, 6))
    # Contourf for raw logits
    levels = 50  
    plt.contourf(xx, yy, logits_np, levels=levels, cmap="bwr", alpha=0.4)
    # Decision boundary (logits = 0.0, i.e. probability=0.5)
    plt.contour(xx, yy, logits_np, levels=[0.0], colors='black', linewidths=1)

    # Plot training data
    plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap='bwr', edgecolors='k', alpha=0.8, label='Train')

    # Plot test data
    plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap='bwr', edgecolors='gray', alpha=0.4, marker='s', label='Test')

    # Compute train/test accuracies
    train_logits = model.apply(params, X_train).squeeze()
    train_preds = (train_logits > 0.5).astype(int)
    train_acc = jnp.mean(train_preds == y_train) * 100

    test_logits = model.apply(params, X_test).squeeze()
    test_preds = (test_logits > 0.5).astype(int)
    test_acc = jnp.mean(test_preds == y_test) * 100

    # Annotate
    plt.text(
        x_min + 0.2, 
        y_max - 0.3,
        f"Train Acc = {train_acc:.1f}%\nTest Acc = {test_acc:.1f}%",
        color="black",
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.6)
    )

    plt.title("Decision Boundary")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(loc="upper right")

    if save_path:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()
