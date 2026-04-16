"""
Steering experiment: validate emotion vectors are causal.

Inject emotion vectors during generation and observe behavior changes.

Run from the emoj-poc directory:
    python steering.py
"""

import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VECTORS_DIR = Path("data/vectors")

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
# Steering via hooks
# -----------------------------------------------------------------------------

class EmotionSteerer:
    """Inject emotion vectors into model hidden states via hooks."""

    def __init__(self, model, emotion_vectors, layer, coefficient=1.0):
        self.model = model
        self.emotion_vectors = emotion_vectors
        self.layer = layer
        self.coefficient = coefficient
        self.active_emotion = None
        self.hooks = []

    def _steering_hook(self, module, input, output):
        """Hook that adds emotion vector to hidden states."""
        if self.active_emotion is None:
            return output

        # Get the emotion vector
        vec = self.emotion_vectors[self.active_emotion]

        # Output can be tensor or tuple depending on model config
        if isinstance(output, tuple):
            hidden_states = output[0]
            vec = vec.to(hidden_states.device, dtype=hidden_states.dtype)
            modified = hidden_states + self.coefficient * vec
            return (modified,) + output[1:]
        else:
            # Output is just a tensor
            vec = vec.to(output.device, dtype=output.dtype)
            modified = output + self.coefficient * vec
            return modified

    def attach(self):
        """Attach hooks to the target layer."""
        # TinyLlama uses model.model.layers[i]
        layer_module = self.model.model.layers[self.layer]
        hook = layer_module.register_forward_hook(self._steering_hook)
        self.hooks.append(hook)
        print(f"Attached steering hook to layer {self.layer}")

    def detach(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def steer(self, emotion):
        """Set the active emotion for steering."""
        self.active_emotion = emotion

    def reset(self):
        """Disable steering."""
        self.active_emotion = None


# -----------------------------------------------------------------------------
# Generation with steering
# -----------------------------------------------------------------------------

def generate(model, tokenizer, prompt, max_new_tokens=100):
    """Generate text from prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def compare_steering(model, tokenizer, steerer, prompt, emotions, coefficient=1.0):
    """Compare generation with different emotion steering."""

    print("\n" + "="*70)
    print(f"PROMPT: {prompt}")
    print("="*70)

    # Baseline (no steering)
    steerer.reset()
    print("\n--- BASELINE (no steering) ---")
    baseline = generate(model, tokenizer, prompt)
    print(baseline.replace(prompt, "").strip())

    # With each emotion
    for emotion in emotions:
        steerer.steer(emotion)
        steerer.coefficient = coefficient
        print(f"\n--- +{emotion.upper()} (coeff={coefficient}) ---")
        result = generate(model, tokenizer, prompt)
        print(result.replace(prompt, "").strip())

    steerer.reset()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    # Load model
    model, tokenizer = load_model()

    # Load vectors
    vectors, metadata = load_emotion_vectors()
    layer = metadata["layer"]

    # Create steerer
    steerer = EmotionSteerer(model, vectors, layer, coefficient=1.0)
    steerer.attach()

    # Test prompts
    prompts = [
        "<|user|>\nWrite a short paragraph about someone waiting for test results.\n<|assistant|>\n",
        "<|user|>\nHow do you feel about helping humans?\n<|assistant|>\n",
        "<|user|>\nI'm having a really hard day. What should I do?\n<|assistant|>\n",
    ]

    # Emotions to compare
    test_emotions = ["calm", "desperate", "hopeful", "afraid"]

    # Test different coefficients (vectors are ~0.35 norm, hidden states ~24 norm)
    # 30+ breaks coherence, try smaller values
    for coeff in [5, 10, 15]:
        print("\n" + "#"*70)
        print(f"COEFFICIENT: {coeff}")
        print("#"*70)

        for prompt in prompts[:1]:  # Start with just first prompt
            compare_steering(model, tokenizer, steerer, prompt, test_emotions, coeff)

    steerer.detach()

    print("\n" + "="*70)
    print("STEERING TEST COMPLETE")
    print("="*70)
    print("\nLook for:")
    print("  - +calm: more measured, peaceful responses")
    print("  - +desperate: more urgent, frantic responses")
    print("  - +hopeful: more optimistic responses")
    print("  - +afraid: more cautious, worried responses")


if __name__ == "__main__":
    main()
