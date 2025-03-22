import os
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Disable GUI for batch processing
import matplotlib.pyplot as plt
import jax.numpy as jnp
from PIL import Image
import matplotlib.patches as mpatches


def _ensure_dir(path):
    """Ensure the directory exists for saving plots."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

def layered_layout(genome):
    G = nx.DiGraph()
    # replicate adding nodes/edges if needed, or do it externally
    # separate node ids by type
    input_nodes = [n for n, node_gene in genome.nodes.items() if node_gene.node_type == 'input']
    hidden_nodes = [n for n, node_gene in genome.nodes.items() if node_gene.node_type == 'hidden']
    output_nodes = [n for n, node_gene in genome.nodes.items() if node_gene.node_type == 'output']

    pos = {}
    # Place input nodes at x=0
    for i, n in enumerate(input_nodes):
        pos[n] = (0, i)
    # Place hidden nodes at x=1
    for i, n in enumerate(hidden_nodes):
        pos[n] = (1, i)
    # Place output nodes at x=2
    for i, n in enumerate(output_nodes):
        pos[n] = (2, i)
    return pos

def visualize_genome(
    genome,
    save_path=None,
    generation=None,
    train_acc=None,
    test_acc=None,
    activation_info="ReLU -> Sigmoid"
):
    """
    Render a NEAT genome using NetworkX, and optionally annotate:
      - generation index
      - train/test accuracy
      - activation function info
    """
    plt.figure(figsize=(6, 6))
    G = nx.DiGraph()

    # 1) Build graph from genome
    for node_id, node_gene in genome.nodes.items():
        G.add_node(node_id, type=node_gene.node_type)

    for c in genome.connections:
        if c.enabled:
            G.add_edge(c.in_node, c.out_node, weight=c.weight)

    # 2) Decide layout
    pos = layered_layout(genome)  # or nx.spring_layout(G, seed=42)

    # 3) Assign colors by node type
    node_colors = {'input': 'lightblue', 'hidden': 'lightgray', 'output': 'lightgreen'}
    node_list = list(G.nodes())
    node_color_map = [node_colors[G.nodes[n]['type']] for n in node_list]

    # 4) Draw the graph
    nx.draw(G, pos, with_labels=True, node_color=node_color_map, node_size=800, font_size=10)

    # 5) Draw edge labels (rounded to 2 decimals)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    edge_labels_rounded = {k: f"{v:.2f}" for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_rounded, font_color='red')

    # 6) Add a legend for node types
    legend_handles = [
        mpatches.Patch(color='lightblue', label='Input'),
        mpatches.Patch(color='lightgray', label='Hidden'),
        mpatches.Patch(color='lightgreen', label='Output')
    ]
    plt.legend(handles=legend_handles, title="Node Types", loc="upper left")

    # 7) Title & Additional Text
    # Basic title showing #nodes and #edges
    plt.title(f"Genome (Nodes: {len(G.nodes())}, Edges: {len(G.edges())})", pad=20)

    # Show activation info near the top center
    plt.text(0.5, 1.04, f"Activations: {activation_info}",
             transform=plt.gca().transAxes,
             ha='center', va='bottom', fontsize=9)

    # Generation + Accuracies in bottom-left corner
    text_lines = []
    if generation is not None:
        text_lines.append(f"Gen: {generation}")
    if train_acc is not None:
        text_lines.append(f"Train Acc: {train_acc*100:.1f}%")
    if test_acc is not None:
        text_lines.append(f"Test Acc: {test_acc*100:.1f}%")
    if text_lines:
        info_str = "\n".join(text_lines)
        plt.text(0.02, 0.02, info_str, transform=plt.gca().transAxes,
                 ha='left', va='bottom', fontsize=9,
                 bbox=dict(facecolor='white', alpha=0.6, boxstyle="round"))

    # 8) Save the figure
    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=300)
    plt.close()

def visualize_decision_boundary(
    params, model, X_train, y_train, X_test, y_test, 
    data_range=(-1,1), resolution=200, save_path=None
):
    """Plot decision boundary with train/test data points."""
    x_min, x_max = data_range
    y_min, y_max = data_range  # Ensure y-range matches x-range

    xs = jnp.linspace(x_min, x_max, resolution)
    ys = jnp.linspace(y_min, y_max, resolution)

    xx, yy = jnp.meshgrid(xs, ys)
    grid = jnp.column_stack([xx.ravel(), yy.ravel()])  # shape (resolution^2, 2)

    logits = model.apply(params, grid).squeeze()  # shape (res^2,)
    preds = (logits > 0.5).astype(int)

    logits_np = logits.reshape((resolution, resolution))

    plt.figure(figsize=(6, 6))
    plt.contourf(xx, yy, logits_np, levels=50, cmap="bwr", alpha=0.4)
    plt.contour(xx, yy, logits_np, levels=[0.0], colors='black', linewidths=1)

    plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap='bwr', edgecolors='k', alpha=0.8, label='Train')
    plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap='bwr', edgecolors='gray', alpha=0.4, marker='s', label='Test')

    train_logits = model.apply(params, X_train).squeeze()
    train_preds = (train_logits > 0.5).astype(int)
    train_acc = jnp.mean(train_preds == y_train) * 100

    test_logits = model.apply(params, X_test).squeeze()
    test_preds = (test_logits > 0.5).astype(int)
    test_acc = jnp.mean(test_preds == y_test) * 100

    plt.text(x_min + 0.2, y_max - 0.3, 
             f"Train Acc = {train_acc:.1f}%\nTest Acc = {test_acc:.1f}%",
             color="black", fontsize=10, bbox=dict(facecolor='white', alpha=0.6))

    plt.title("Decision Boundary")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(loc="upper right")

    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=300)
    plt.close()


def create_genome_evolution_gif(image_folder, output_gif, duration=500):
    """Generate a GIF from saved genome images per generation."""
    images = []
    
    # ✅ Ensure images are sorted correctly (gen_0, gen_1, ..., gen_n)
    for filename in sorted(os.listdir(image_folder), key=lambda x: int(x.split('_')[-1].split('.')[0])):
        img_path = os.path.join(image_folder, filename)
        images.append(Image.open(img_path))

    if images:
        images[0].save(output_gif, save_all=True, append_images=images[1:], duration=duration, loop=0)
        print(f"✅ Genome evolution GIF saved: {output_gif}")
    else:
        print("⚠️ No images found for genome evolution GIF.")