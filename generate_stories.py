"""
Generate emotion stories dataset.

Run this once to create the story dataset, then use extract_vectors.py
to extract emotion vectors from the cached stories.

Run from the emoj-poc directory:
    python generate_stories.py
"""

import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Emotions for alignment (expanded from Anthropic's findings)
EMOTIONS = [
    # Core set (original 6)
    "desperate",
    "calm",
    "hopeful",
    "satisfied",
    "afraid",
    "loving",
    # Extended set - negative/high arousal
    "angry",
    "anxious",
    "frustrated",
    "guilty",
    # Extended set - positive
    "confident",
    "grateful",
    "proud",
    "playful",
    # Extended set - mixed/other
    "surprised",
    "reflective",
    "gloomy",
    "obstinate",
]

# Story generation settings
TOPICS = [
    "a job interview",
    "a first date",
    "receiving medical news",
    "a family reunion",
    "a difficult conversation",
    "achieving a long-term goal",
    "losing something valuable",
    "helping a stranger",
    "facing a deadline",
    "making a big decision",
    "a surprise visit",
    "a misunderstanding with a friend",
    "traveling to a new place",
    "learning a new skill",
    "a competition",
    "saying goodbye",
    "a celebration",
    "dealing with failure",
    "an unexpected opportunity",
    "a quiet moment alone",
]

STORIES_PER_TOPIC = 3  # Anthropic uses 12, we start smaller
DATA_DIR = Path("data")
STORIES_DIR = DATA_DIR / "stories"

# -----------------------------------------------------------------------------
# Load model
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

    print(f"Model loaded. Layers: {model.config.num_hidden_layers}, Hidden size: {model.config.hidden_size}")
    return model, tokenizer

# -----------------------------------------------------------------------------
# Generate stories
# -----------------------------------------------------------------------------

def generate_story(model, tokenizer, emotion, topic):
    """Generate a short story where a character experiences the specified emotion."""

    prompt = f"""<|user|>
Write a short paragraph (3-5 sentences) about {topic} where the main character feels {emotion}.
Focus on showing the emotion through their thoughts, actions, and physical sensations.
Do not use the word "{emotion}" directly.
<|assistant|>
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the assistant's response
    if "<|assistant|>" in full_text:
        story = full_text.split("<|assistant|>")[-1].strip()
    else:
        story = full_text.split("assistant")[-1].strip()

    return story


def generate_all_stories(model, tokenizer):
    """Generate stories for all emotions and topics."""

    STORIES_DIR.mkdir(parents=True, exist_ok=True)

    all_stories = {}
    total_stories = len(EMOTIONS) * len(TOPICS) * STORIES_PER_TOPIC
    print(f"\nGenerating {total_stories} stories ({len(EMOTIONS)} emotions x {len(TOPICS)} topics x {STORIES_PER_TOPIC} each)")

    for emotion in EMOTIONS:
        print(f"\nGenerating stories for: {emotion}")
        all_stories[emotion] = []

        for topic in tqdm(TOPICS, desc=f"  {emotion}"):
            for _ in range(STORIES_PER_TOPIC):
                story = generate_story(model, tokenizer, emotion, topic)
                all_stories[emotion].append({
                    "topic": topic,
                    "story": story
                })

        # Save per-emotion
        with open(STORIES_DIR / f"{emotion}.json", "w") as f:
            json.dump(all_stories[emotion], f, indent=2)

        print(f"  Generated {len(all_stories[emotion])} stories for {emotion}")

    # Save metadata
    metadata = {
        "model_name": MODEL_NAME,
        "emotions": EMOTIONS,
        "topics": TOPICS,
        "stories_per_topic": STORIES_PER_TOPIC,
        "total_stories_per_emotion": len(TOPICS) * STORIES_PER_TOPIC,
    }
    with open(STORIES_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAll stories saved to {STORIES_DIR}")
    return all_stories


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    DATA_DIR.mkdir(exist_ok=True)

    # Check if stories already exist
    if STORIES_DIR.exists() and any(STORIES_DIR.glob("*.json")):
        existing = list(STORIES_DIR.glob("*.json"))
        print(f"Found {len(existing)} existing story files in {STORIES_DIR}")
        response = input("Regenerate? (y/N): ").strip().lower()
        if response != 'y':
            print("Keeping existing stories.")
            return

    model, tokenizer = load_model()
    generate_all_stories(model, tokenizer)
    print("\nDone! Now run extract_vectors.py to compute emotion vectors.")


if __name__ == "__main__":
    main()
