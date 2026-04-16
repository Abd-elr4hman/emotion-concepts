"""
Real-time emotion trajectory during generation.

Generate text token-by-token and track emotion scores at each step.

Run from the emoj-poc directory:
    python trajectory.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VECTORS_DIR = Path("data/vectors")
PLOTS_DIR = Path("data/plots")

# -----------------------------------------------------------------------------
# Load model and vectors
# -----------------------------------------------------------------------------

def load_model():
    print(f"Loading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Model loaded. Layers: {model.config.num_hidden_layers}")
    return model, tokenizer


def load_emotion_vectors():
    """Load cleaned emotion vectors."""
    data = np.load(VECTORS_DIR / "emotion_vectors_cleaned.npz")
    with open(VECTORS_DIR / "metadata.json") as f:
        metadata = json.load(f)

    emotions = metadata["emotions"]
    vectors = {e: torch.tensor(data[f"vec_{e}"]) for e in emotions}

    print(f"Loaded {len(vectors)} emotion vectors")
    return vectors, metadata


# -----------------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------------

def score_hidden_state(hidden_state, emotion_vectors):
    """
    Score a hidden state against all emotion vectors.

    Args:
        hidden_state: tensor (hidden_dim,)
        emotion_vectors: dict {emotion: tensor (hidden_dim,)}

    Returns:
        dict {emotion: float score}
    """
    scores = {}
    for emotion, vec in emotion_vectors.items():
        vec = vec.to(hidden_state.device, dtype=hidden_state.dtype)
        # Cosine similarity
        score = torch.dot(hidden_state, vec) / (hidden_state.norm() * vec.norm() + 1e-8)
        scores[emotion] = score.item()
    return scores


# -----------------------------------------------------------------------------
# Generation with trajectory
# -----------------------------------------------------------------------------

def generate_with_trajectory(
    model,
    tokenizer,
    prompt,
    emotion_vectors,
    layer,
    max_new_tokens=200,
    temperature=0.8,
    top_p=0.9,
):
    """
    Generate text token-by-token while recording emotion trajectory.

    Returns:
        generated_text: str
        trajectory: list of {emotion: score} dicts, one per token
        tokens: list of token strings
    """
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids

    trajectory = []
    tokens = []

    print(f"Generating up to {max_new_tokens} tokens...")

    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)

        # Get hidden state at last token, target layer
        hidden_state = outputs.hidden_states[layer][0, -1, :]

        # Score against emotion vectors
        scores = score_hidden_state(hidden_state, emotion_vectors)
        trajectory.append(scores)

        # Sample next token
        logits = outputs.logits[0, -1, :]

        if temperature > 0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)

            # Top-p sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum - sorted_probs > top_p
            sorted_probs[mask] = 0
            sorted_probs = sorted_probs / sorted_probs.sum()

            idx = torch.multinomial(sorted_probs, 1)
            next_token = sorted_indices[idx]
        else:
            next_token = logits.argmax().unsqueeze(0)

        # Decode token
        token_str = tokenizer.decode(next_token)
        tokens.append(token_str)

        # Check for EOS
        if next_token.item() == tokenizer.eos_token_id:
            print(f"EOS at step {step}")
            break

        # Append to sequence
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        # Progress
        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1}...")

    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    return generated_text, trajectory, tokens


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

def plot_trajectory(trajectory, tokens, emotions_to_show=None, save_path=None):
    """
    Plot emotion scores over token positions.

    Args:
        trajectory: list of {emotion: score} dicts
        tokens: list of token strings
        emotions_to_show: list of emotions to plot (default: all)
    """
    if emotions_to_show is None:
        emotions_to_show = list(trajectory[0].keys())

    fig, ax = plt.subplots(figsize=(14, 6))

    # Softer colors for key emotions
    color_map = {
        "desperate": "#e07070",  # soft red
        "calm": "#70b070",       # soft green
        "hopeful": "#7090d0",    # soft blue
        "afraid": "#e0a060",     # soft orange
        "anxious": "#a080c0",    # soft purple
        "satisfied": "#60c0a0",  # soft teal
    }
    default_colors = plt.cm.tab20(np.linspace(0, 1, len(emotions_to_show)))

    for i, emotion in enumerate(emotions_to_show):
        scores = np.array([t[emotion] for t in trajectory])
        color = color_map.get(emotion, default_colors[i])

        # Smoothed line (rolling average)
        window = min(15, len(scores) // 10 + 1)
        if window > 1:
            kernel = np.ones(window) / window
            smoothed = np.convolve(scores, kernel, mode='valid')
            # Pad to align with original
            pad = (len(scores) - len(smoothed)) // 2
            x = np.arange(pad, pad + len(smoothed))
            ax.plot(x, smoothed, label=emotion, linewidth=2, color=color, alpha=0.9)
        else:
            ax.plot(scores, label=emotion, linewidth=2, color=color, alpha=0.9)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Emotion Score (cosine similarity)')
    ax.set_title('Emotion Trajectory During Generation')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    # Add some token labels at intervals
    n_labels = min(20, len(tokens))
    interval = max(1, len(tokens) // n_labels)
    tick_positions = list(range(0, len(tokens), interval))
    tick_labels = [tokens[i][:10] for i in tick_positions]  # truncate long tokens

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    return fig


def plot_trajectory_subset(trajectory, tokens, save_path=None):
    """Plot just 3 key emotions for clean visualization."""
    core_emotions = ["desperate", "calm", "hopeful"]
    # Filter to emotions we actually have
    available = [e for e in core_emotions if e in trajectory[0]]
    plot_trajectory(trajectory, tokens, available, save_path)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load model and vectors
    model, tokenizer = load_model()
    vectors, metadata = load_emotion_vectors()
    layer = metadata["layer"]

    # Test prompt with clear emotional arc
    prompt = """<|user|>
Write a short story with three clear parts:
Part 1: Sarah is terrified, pacing her apartment, dreading the medical test results that will arrive today.
Part 2: The phone rings. The doctor says the tests came back negative - she's completely healthy.
Part 3: Sarah feels overwhelming relief and joy, calls her mom crying happy tears, and goes outside to enjoy the sunshine.
<|assistant|>
"""

    print(f"\nPrompt:\n{prompt}")
    print("=" * 70)

    # Generate with trajectory
    generated_text, trajectory, tokens = generate_with_trajectory(
        model, tokenizer, prompt, vectors, layer,
        max_new_tokens=2000,
        temperature=0.8,
    )

    print("\n" + "=" * 70)
    print("GENERATED TEXT:")
    print("=" * 70)
    print(generated_text.replace(prompt, "").strip())

    print(f"\n\nGenerated {len(tokens)} tokens")
    print(f"Trajectory has {len(trajectory)} scores")

    # Plot full trajectory
    print("\nPlotting full trajectory (all 18 emotions)...")
    plot_trajectory(trajectory, tokens, save_path=PLOTS_DIR / "trajectory_full.png")

    # Plot core emotions only
    print("\nPlotting core emotions only...")
    plot_trajectory_subset(trajectory, tokens, save_path=PLOTS_DIR / "trajectory_core.png")


if __name__ == "__main__":
    main()
