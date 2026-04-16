# Real-Time Emotion Trajectory Visualizer

**Goal:** Visualize multiple emotion/concept trajectories in real-time as a model generates text — like Anthropic's figure showing desperate/hopeful/satisfied/obstinate over a 35k token math proof.

**Based on:** [Anthropic Transformer Circuits - Emotions (2026)](https://transformer-circuits.pub/2026/emotions/index.html)

---

## Key Insights from Anthropic's Paper

### What They Found

1. **Emotion vectors are locally scoped** — they track the "operative emotion" at each token, not a persistent character state
2. **Layer dynamics matter:**
   - Early layers: token-level emotional connotation (just the word itself)
   - Early-middle: local context emotion (current phrase) — "sensory"
   - Middle-late (~2/3 through model): emotion relevant to predicting next tokens — "action"
3. **The "Assistant colon" token** (`:` after "Assistant") predicts emotional content of upcoming response (r=0.87)
4. **Geometry mirrors human psychology** — PC1 = valence, PC2 = arousal
5. **Causal effects are real** — steering with emotion vectors changes behavior (sycophancy, blackmail, reward hacking)

### How They Differ from RepE

| Aspect | RepE | Anthropic Emotions |
|--------|------|-------------------|
| Training data | Contrast pairs ("be honest" vs "be dishonest") | Stories where characters *experience* emotions |
| Extraction | PCA on activation differences | Average activations, subtract mean, project out confounds |
| What they capture | Model's "mode" before responding | Operative emotion at each token |
| Validation | Classification accuracy | Causal steering + correlation with preferences |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Generation Loop                          │
│  ┌─────────┐    ┌──────────────┐    ┌─────────────────────┐    │
│  │  Model  │───▶│ Hidden States │───▶│ Project onto        │    │
│  │ Forward │    │ (mid-late     │    │ Emotion Vectors     │    │
│  │         │    │  layers)      │    │                     │    │
│  └─────────┘    └──────────────┘    └──────────┬──────────┘    │
│       │                                         │               │
│       ▼                                         ▼               │
│  ┌─────────┐                           ┌──────────────┐        │
│  │  Next   │                           │ Live Plot    │        │
│  │  Token  │                           │ Update       │        │
│  └─────────┘                           └──────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Method: Extracting Emotion Vectors (Anthropic's Approach)

### Step 1: Generate Training Stories

For each emotion (e.g., "desperate", "hopeful", "calm"):
```python
# Prompt the model to write stories where a character experiences the emotion
stories[emotion] = []
for topic in topics:  # ~100 diverse topics
    prompt = f"Write a short story about {topic} where a character experiences {emotion}"
    story = model.generate(prompt)
    stories[emotion].append(story)
```

### Step 2: Extract Activations

```python
emotion_activations = {}
for emotion, story_list in stories.items():
    activations = []
    for story in story_list:
        tokens = tokenizer(story)
        with torch.no_grad():
            outputs = model(tokens, output_hidden_states=True)
        
        # Average across token positions (after token 50 where emotion is apparent)
        hidden = outputs.hidden_states[mid_late_layer][:, 50:, :].mean(dim=1)
        activations.append(hidden)
    
    emotion_activations[emotion] = torch.stack(activations).mean(dim=0)
```

### Step 3: Compute Emotion Vectors

```python
# Subtract global mean across emotions
global_mean = torch.stack(list(emotion_activations.values())).mean(dim=0)
emotion_vectors = {
    emotion: act - global_mean 
    for emotion, act in emotion_activations.items()
}
```

### Step 4: Project Out Confounds (Important!)

```python
# Get activations on emotionally NEUTRAL text
neutral_activations = get_activations_on_neutral_corpus()

# Compute top PCs explaining 50% variance
pca = PCA(n_components=k)  # k chosen to explain 50% variance
pca.fit(neutral_activations)
confound_directions = pca.components_

# Project confounds OUT of emotion vectors
for emotion in emotion_vectors:
    for direction in confound_directions:
        projection = (emotion_vectors[emotion] @ direction) * direction
        emotion_vectors[emotion] = emotion_vectors[emotion] - projection
```

---

## Scoring: Per-Token Emotion Projection

```python
def score_token(hidden_state, emotion_vectors, layer):
    """
    Score a single token's hidden state against all emotion vectors.
    
    Args:
        hidden_state: tensor (hidden_dim,) at the token position
        emotion_vectors: dict {emotion_name: vector (hidden_dim,)}
        layer: which layer (should be mid-late, ~2/3 through model)
    
    Returns:
        dict {emotion_name: float score}
    """
    scores = {}
    for emotion, vector in emotion_vectors.items():
        # Simple dot product (cosine similarity also works)
        scores[emotion] = (hidden_state @ vector).item()
    return scores
```

---

## Layer Selection

```python
def get_mid_late_layer(model):
    """
    Get the layer ~2/3 through the model.
    This is where 'action' emotion representations live.
    """
    num_layers = model.config.num_hidden_layers
    return int(num_layers * 0.66)

# For a 22-layer model (TinyLlama): layer 14
# For a 32-layer model (Mistral-7B): layer 21
# For a 80-layer model (Llama-70B): layer 53
```

---

## Generation Loop with Trajectory

```python
def generate_with_emotion_trajectory(
    model, 
    tokenizer, 
    prompt, 
    emotion_vectors,
    max_new_tokens=100
):
    """
    Generate text while recording emotion trajectory at each token.
    """
    mid_late_layer = get_mid_late_layer(model)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    trajectory = []
    tokens_generated = []
    
    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
        
        # Get hidden state at last token position, mid-late layer
        hidden_state = outputs.hidden_states[mid_late_layer][0, -1, :]
        
        # Score against all emotion vectors
        scores = score_token(hidden_state, emotion_vectors, mid_late_layer)
        trajectory.append(scores)
        
        # Get next token
        next_token_logits = outputs.logits[0, -1, :]
        next_token = next_token_logits.argmax()
        tokens_generated.append(tokenizer.decode(next_token))
        
        # Check for EOS
        if next_token.item() == tokenizer.eos_token_id:
            break
        
        # Append to sequence
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
    
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text, trajectory, tokens_generated
```

---

## Visualization

```python
import matplotlib.pyplot as plt

def plot_emotion_trajectory(trajectory, tokens, emotions_to_show=None):
    """
    Plot emotion scores over token positions.
    
    Args:
        trajectory: list of {emotion: score} dicts
        tokens: list of token strings (for x-axis labels)
        emotions_to_show: list of emotion names to plot (default: all)
    """
    if emotions_to_show is None:
        emotions_to_show = list(trajectory[0].keys())
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for emotion in emotions_to_show:
        scores = [t[emotion] for t in trajectory]
        ax.plot(scores, label=emotion, linewidth=1.5)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Emotion Score (projection onto emotion vector)')
    ax.set_title('Emotion Trajectory During Generation')
    ax.legend(loc='upper right')
    
    # Add token labels at intervals
    tick_interval = max(1, len(tokens) // 20)
    ax.set_xticks(range(0, len(tokens), tick_interval))
    ax.set_xticklabels(tokens[::tick_interval], rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    return fig, ax
```

---

## File Structure

```
emoj-poc/
├── plan.md                 # This file
├── extract_vectors.py      # Generate stories, extract emotion vectors
├── scoring.py              # Score hidden states against vectors
├── generation.py           # Custom generation loop with trajectory
├── visualize.py            # Plotting utilities
├── main.py                 # Entry point
├── data/
│   ├── stories/            # Generated emotion stories (cached)
│   └── vectors/            # Extracted emotion vectors (cached)
└── configs/
    └── default.yaml        # Model, emotions, layers, etc.
```

---

## Implementation Order

### Phase 1: Extract Emotion Vectors
1. [ ] Generate emotion stories dataset (or use existing if available)
2. [ ] Extract activations from stories
3. [ ] Compute emotion vectors (mean - global mean)
4. [ ] Project out confounds from neutral corpus
5. [ ] Save vectors to disk

### Phase 2: Scoring & Generation
6. [ ] Implement `score_token()` function
7. [ ] Implement `generate_with_emotion_trajectory()`
8. [ ] Test on simple prompts, verify vectors activate sensibly

### Phase 3: Visualization
9. [ ] Static plot of trajectory (matplotlib)
10. [ ] Live updating plot during generation
11. [ ] Add token annotations on x-axis

### Phase 4: Validation & Polish
12. [ ] Validate with causal steering (does adding vector change output?)
13. [ ] Test on alignment-relevant scenarios (like Anthropic's blackmail case)
14. [ ] Move to larger model
15. [ ] Web UI (Gradio/Streamlit)

---

## Emotions to Start With

Based on Anthropic's findings on alignment-relevant emotions:

**Core set (6):**
- `desperate` — key driver of misalignment (blackmail, reward hacking)
- `calm` — opposite of desperate, associated with aligned behavior  
- `hopeful` — positive valence, high arousal
- `satisfied` — positive valence, low arousal
- `afraid` — activates on dangerous content
- `loving` — associated with empathetic responses

**Extended set (add later):**
- `angry`, `guilty`, `proud`, `surprised`
- `obstinate` — relevant to persistence in difficult tasks
- `playful`, `reflective`, `gloomy`

---

## Key Differences from Our Original RepE-Based Plan

| Original Plan | Updated Plan |
|---------------|--------------|
| Use RepE's contrast pairs | Generate emotion stories |
| PCA on activation differences | Average activations, subtract mean |
| Use RepReader class | Simple dict of vectors |
| Unclear which layers | Mid-late layers (~2/3 through) |
| No confound removal | Project out neutral corpus PCs |

---

## Open Questions

1. **Story generation:** Can we use a smaller model to generate stories, then extract vectors from a larger model?

2. **Neutral corpus:** What counts as "emotionally neutral"? Wikipedia? Code? Need to define this.

3. **Vector normalization:** Should we normalize emotion vectors to unit length for fair comparison?

4. **Cross-model transfer:** Do emotion vectors from one model work on another?

5. **Computational cost:** Generating 100+ stories per emotion × 171 emotions is expensive. Can we use fewer?

---

## Refinements (TODO)

1. **Filter emotion word leakage:** TinyLlama doesn't follow "don't use the word X" instructions well. Stories often contain the target emotion word directly. Options:
   - Use a stronger model for story generation
   - Post-filter stories that contain the emotion word
   - May not matter much if activations still capture the emotion

2. **Truncation is acceptable:** Stories cut off at 150 tokens mid-sentence still carry emotional content. The truncation doesn't hurt — the emotion is concentrated in the narrative, not the conclusion.

---

## References

- [Anthropic Transformer Circuits - Emotions (2026)](https://transformer-circuits.pub/2026/emotions/index.html)
- [The Geometry of Truth (Marks & Tegmark)](https://arxiv.org/abs/2310.06824)
- [Representation Engineering (Zou et al.)](https://arxiv.org/abs/2310.01405)
- [Discovering Latent Knowledge (Burns et al.)](https://arxiv.org/abs/2212.03827)
