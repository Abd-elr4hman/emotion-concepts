"""
Analyze extracted emotion vectors.

Run after extract_vectors.py:
    python analyze_vectors.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

VECTORS_DIR = Path("data/vectors")
PLOTS_DIR = Path("data/plots")

# -----------------------------------------------------------------------------
# Load vectors
# -----------------------------------------------------------------------------

def load_vectors():
    """Load cleaned emotion vectors."""
    data = np.load(VECTORS_DIR / "emotion_vectors_cleaned.npz")
    with open(VECTORS_DIR / "metadata.json") as f:
        metadata = json.load(f)

    emotions = metadata["emotions"]
    vectors = {e: data[f"vec_{e}"] for e in emotions}

    return vectors, metadata

# -----------------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------------

def compute_cosine_similarity_matrix(vectors):
    """Compute pairwise cosine similarities."""
    emotions = list(vectors.keys())
    n = len(emotions)

    # Stack into matrix
    V = np.stack([vectors[e] for e in emotions])  # (n_emotions, hidden_dim)

    # Normalize rows
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    V_norm = V / norms

    # Cosine similarity matrix
    sim_matrix = V_norm @ V_norm.T  # (n_emotions, n_emotions)

    return sim_matrix, emotions

def plot_similarity_heatmap(sim_matrix, emotions, save_path=None):
    """Plot cosine similarity heatmap with hierarchical clustering like Anthropic's figure."""
    import pandas as pd
    from scipy.cluster.hierarchy import linkage, leaves_list

    # Create DataFrame for clustermap
    df = pd.DataFrame(sim_matrix, index=emotions, columns=emotions)

    # Get clustering order without showing dendrograms
    linkage_matrix = linkage(sim_matrix, method='average')
    order = leaves_list(linkage_matrix)
    ordered_emotions = [emotions[i] for i in order]

    # Reorder dataframe
    df_ordered = df.loc[ordered_emotions, ordered_emotions]

    # Plot clean heatmap (no dendrograms)
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        df_ordered,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        square=True,
        annot_kws={"size": 9},
        cbar_kws={
            "label": "Cosine Similarity",
            "shrink": 0.8,
        },
        ax=ax,
        linewidths=0.5,
        linecolor='white',
    )

    ax.set_title("Emotion Vector Similarity\n(hierarchically clustered)", fontsize=14, pad=20)
    plt.xticks(rotation=45, ha="right", fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    return fig

def plot_pca_2d(vectors, save_path=None):
    """Project emotion vectors to 2D via PCA (should show valence/arousal axes)."""
    from sklearn.decomposition import PCA

    emotions = list(vectors.keys())
    V = np.stack([vectors[e] for e in emotions])

    pca = PCA(n_components=2)
    V_2d = pca.fit_transform(V)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(V_2d[:, 0], V_2d[:, 1], s=100, c='steelblue')

    for i, emotion in enumerate(emotions):
        ax.annotate(
            emotion,
            (V_2d[i, 0], V_2d[i, 1]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=12
        )

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title("Emotion Vectors in 2D (PCA)\nExpected: PC1=valence, PC2=arousal")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    plt.show()
    return fig

def print_vector_stats(vectors):
    """Print basic stats about the vectors."""
    print("\nVector Statistics:")
    print("-" * 40)

    for emotion, vec in vectors.items():
        norm = np.linalg.norm(vec)
        print(f"  {emotion:12}: norm = {norm:.4f}")

    # Overall stats
    all_vecs = np.stack(list(vectors.values()))
    print(f"\n  Mean norm:   {np.linalg.norm(all_vecs, axis=1).mean():.4f}")
    print(f"  Std norm:    {np.linalg.norm(all_vecs, axis=1).std():.4f}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading vectors...")
    vectors, metadata = load_vectors()
    print(f"Loaded {len(vectors)} emotion vectors")
    print(f"Model: {metadata['model_name']}")
    print(f"Layer: {metadata['layer']}")
    print(f"Hidden dim: {metadata['hidden_dim']}")

    # Stats
    print_vector_stats(vectors)

    # Cosine similarity heatmap
    print("\n" + "=" * 50)
    print("Cosine Similarity Heatmap")
    print("=" * 50)
    sim_matrix, emotions = compute_cosine_similarity_matrix(vectors)
    plot_similarity_heatmap(sim_matrix, emotions, PLOTS_DIR / "similarity_heatmap.png")

    # PCA projection
    print("\n" + "=" * 50)
    print("PCA 2D Projection")
    print("=" * 50)
    plot_pca_2d(vectors, PLOTS_DIR / "pca_2d.png")

    print("\nDone! Plots saved to data/plots/")

if __name__ == "__main__":
    main()
