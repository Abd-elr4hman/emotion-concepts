# Implementation Decisions: Mapping to Anthropic's Paper

This document traces each implementation decision in our code back to the specific sections and quotes from Anthropic's "Emotions" paper (2026).

---

## 1. Story Generation (`generate_stories.py`)

### Decision: Generate stories where characters experience emotions

**Paper Reference (Part 1: Finding emotion vectors):**
> "To extract vectors corresponding to specific emotion concepts ('emotion vectors'), we first prompted Sonnet 4.5 to write short (roughly one paragraph) stories on diverse topics in which a character experiences a specified emotion (100 topics, 12 stories per topic per emotion)"

**Our Implementation:**
```python
TOPICS = [...]  # 20 topics (Anthropic used 100)
STORIES_PER_TOPIC = 3  # (Anthropic used 12)
```

**Tradeoff:** We use fewer topics (20) and stories per topic (3) for faster iteration. Total: 60 stories/emotion vs Anthropic's 1200.

---

### Decision: Prompt structure asking for emotional portrayal

**Paper Reference:**
> "This provides labeled text where emotional content is clearly present, and which is explicitly associated with what the model views as being related to the emotion"

**Our Implementation:**
```python
prompt = f"""<|user|>
Write a short paragraph (3-5 sentences) about {topic} where the main character feels {emotion}.
Focus on showing the emotion through their thoughts, actions, and physical sensations.
Do not use the word "{emotion}" directly.
<|assistant|>
"""
```

**Note:** We added "Do not use the word directly" to encourage implicit emotional content. Anthropic doesn't mention this explicitly, but they validated stories through "manual inspection of a random subsample."

---

### Decision: Story length ~150 tokens

**Paper Reference:**
> "short (roughly one paragraph) stories"

**Our Implementation:**
```python
max_new_tokens=150
```

---

## 2. Activation Extraction (`extract_vectors.py`)

### Decision: Use mid-late layer (~2/3 through model)

**Paper Reference (Part 1):**
> "We extracted residual stream activations at each layer, averaging across all token positions within each story"

> "Except where otherwise noted, we show results using activations and emotion vectors from a particular model layer about two-thirds of the way through the model"

**Paper Reference (Part 2: What do emotion vectors represent?):**
> "Middle-late layers encode emotions relevant to predicting upcoming tokens ('action' representations)"

> "Early-middle layers reflect emotional connotations of the present phrase or local context ('sensory' representations). Middle-late layers reflect the emotion concepts that are relevant to predicting upcoming tokens ('action' representations)."

**Anthropic's Approach:** They extract at ALL layers (for multi-layer analysis in Part 2), but use the mid-late layer (~2/3 through) for their main results and steering experiments.

**Our Implementation:**
```python
def get_mid_late_layer(model):
    num_layers = model.config.num_hidden_layers
    return int(num_layers * 0.66)  # ~2/3 through
```

For TinyLlama (22 layers): layer 14.

**Simplification:** We only extract at the mid-late layer, skipping multi-layer analysis. This gives us the same final vectors but without the ability to compare layer dynamics.

---

### Decision: Average across token positions, skip early tokens

**Paper Reference:**
> "We extracted residual stream activations at each layer, averaging across all token positions within each story, beginning with the 50th token (at which point the emotional content should be apparent)"

**Our Implementation:**
```python
# Anthropic skips first 50, but our stories are shorter
start_token = min(10, hidden_states.shape[1] // 2)
mean_activation = hidden_states[0, start_token:, :].mean(dim=0)
```

**Tradeoff:** Our stories are ~150 tokens, so we skip 10 tokens instead of 50. The reasoning is the same: skip setup/context, capture emotional content.

---

### Decision: Average across stories per emotion

**Paper Reference:**
> "We obtained emotion vectors by averaging these activations across stories corresponding to a given emotion"

**Our Implementation:**
```python
stacked = torch.stack(activations)  # (n_stories, hidden_dim)
emotion_activations[emotion] = stacked.mean(dim=0)  # (hidden_dim,)
```

---

## 3. Vector Computation (`extract_vectors.py`)

### Decision: Subtract global mean across emotions

**Paper Reference:**
> "We obtained emotion vectors by averaging these activations across stories corresponding to a given emotion, and subtracting off the mean activation across different emotions."

**Our Implementation:**
```python
def compute_emotion_vectors(emotion_activations):
    all_activations = torch.stack(list(emotion_activations.values()))
    global_mean = all_activations.mean(dim=0)
    
    emotion_vectors = {}
    for emotion, activation in emotion_activations.items():
        emotion_vectors[emotion] = activation - global_mean
    
    return emotion_vectors, global_mean
```

**Why:** Subtracting the global mean removes activation patterns common to all emotions (general "story" features), leaving only emotion-specific directions.

---

### Decision: Project out confounds from neutral corpus

**Paper Reference:**
> "We found that the model's activation along these vectors could sometimes be influenced by confounds unrelated to emotion. To mitigate this, we obtained model activations on a set of emotionally neutral transcripts and computed the top principal components of the activations on this dataset (enough to explain 50% of the variance). We then projected out these components from our emotion vectors."

**Our Implementation:**
```python
def project_out_confounds(emotion_vectors, neutral_activations, variance_threshold=0.5):
    # Fit PCA on neutral activations
    pca_full = PCA().fit(neutral_np)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.argmax(cumvar >= variance_threshold) + 1
    
    # Project confounds out of each emotion vector
    for direction in confound_directions:
        projection = (cleaned @ direction) * direction
        cleaned = cleaned - projection
```

**Why:** Neutral text has no emotional content, so its principal components capture non-emotional variance (syntax, topic, style). Removing these from emotion vectors yields cleaner emotional signal.

---

### Decision: 50% variance threshold for confound removal

**Paper Reference:**
> "enough to explain 50% of the variance"

**Our Implementation:**
```python
variance_threshold=0.5
```

---

## 4. Analysis (`analyze_vectors.py`)

### Decision: Cosine similarity heatmap with hierarchical clustering

**Paper Reference (Part 2: The geometry of emotion space):**
> "We first examined the pairwise cosine similarities between emotion vectors... Emotion concepts that we would expect to be similar show high cosine similarity: fear and anxiety cluster together, as do joy and excitement"

**Paper Reference (Figure 5):**
> "Pairwise cosine similarity between all emotion probes, ordered by hierarchical clustering."

**Our Implementation:**
```python
def plot_similarity_heatmap(sim_matrix, emotions, save_path=None):
    linkage_matrix = linkage(sim_matrix, method='average')
    order = leaves_list(linkage_matrix)
    ordered_emotions = [emotions[i] for i in order]
    # ... plot heatmap with reordered emotions
```

---

### Decision: PCA to find valence/arousal axes

**Paper Reference:**
> "We performed PCA on the set of emotion vectors to identify components along which the model's emotion representations are organized. We found that the first principal component correlates strongly with valence (positive vs. negative affect)... We also observed another dominant factor (occupying a mix of the second and third PCs) corresponding to arousal."

> "PC1 (26% variance) orders emotions from fear/panic to joy/optimism, while PC2 (15% variance) separates serene/reflective states from angry/playful arousal."

**Our Implementation:**
```python
def plot_pca_2d(vectors, save_path=None):
    pca = PCA(n_components=2)
    V_2d = pca.fit_transform(V)
    # Plot with axis labels showing variance explained
```

**Our Results:** PC1 = 34.7% variance (valence), PC2 = 16.4% variance. Matches Anthropic's finding that PC1 captures valence.

---

## 5. Steering (`steering.py`)

### Decision: Add emotion vectors to hidden states via hooks

**Paper Reference (Part 1: Emotion vectors reflect and influence preferences):**
> "To test if the emotion vectors are causally important... we performed a steering experiment... Each emotion vector was applied at strength 0.5 across the same middle layers where we previously measured activations."

**Paper Reference (Appendix: Causal effects):**
> "steering with emotion vectors causes the model to produce text in line with the corresponding emotion concept"

**Our Implementation:**
```python
def _steering_hook(self, module, input, output):
    vec = self.emotion_vectors[self.active_emotion]
    vec = vec.to(hidden_states.device, dtype=hidden_states.dtype)
    modified = hidden_states + self.coefficient * vec
    return modified
```

---

### Decision: Steering coefficient calibration

**Paper Reference:**
> "Each emotion vector was applied at strength 0.5"

**Our Finding:**
Anthropic's vectors have different magnitude relative to hidden states. Our vectors have norm ~0.35 while hidden states have norm ~24. We found:
- coeff < 5: no visible effect
- coeff 5-15: noticeable behavioral change
- coeff > 30: breaks coherence

**Note:** The coefficient is model-specific. Anthropic's 0.5 works for their normalized setup; ours needs higher values due to magnitude differences.

---

## 6. Trajectory Scoring (`trajectory.py`)

### Decision: Score each token against emotion vectors via cosine similarity

**Paper Reference (Part 1):**
> "computed the model's activations on these documents and their projection onto the emotion vectors"

> "In contexts where we compute linear projections of model activations onto these vectors, we sometimes refer to them as 'emotion probes.'"

**Our Implementation:**
```python
def score_hidden_state(hidden_state, emotion_vectors):
    scores = {}
    for emotion, vec in emotion_vectors.items():
        score = torch.dot(hidden_state, vec) / (hidden_state.norm() * vec.norm())
        scores[emotion] = score.item()
    return scores
```

---

### Decision: Track emotions at each token during generation

**Paper Reference (Part 2: Emotion vectors encode locally operative emotion concepts):**
> "The emotion vectors we have identified represent the operative emotion concept at a point in time, which is relevant to encoding the local context and predicting the upcoming text"

> "these representations are primarily 'local,' tracking the operative emotion concept most relevant to predicting upcoming tokens"

**Our Implementation:**
```python
for step in range(max_new_tokens):
    outputs = model(input_ids, output_hidden_states=True)
    hidden_state = outputs.hidden_states[layer][0, -1, :]  # last token
    scores = score_hidden_state(hidden_state, emotion_vectors)
    trajectory.append(scores)
```

---

## 7. Emotion Selection

### Decision: Start with alignment-relevant emotions

**Paper Reference (Part 3: Emotion vectors in the wild):**
> "We observe that emotion vectors corresponding to desperation, and lack of calm, play an important and causal role in agentic misalignment"

> "Similarly, desperation vector activation (and calm vector suppression) play a causal role in instances of reward hacking"

**Our Implementation:**
```python
EMOTIONS = [
    # Core set - alignment relevant
    "desperate",  # key driver of misalignment
    "calm",       # opposite of desperate, associated with aligned behavior
    "hopeful",
    "satisfied",
    "afraid",     # activates on dangerous content
    "loving",     # associated with empathetic responses
    # Extended set...
]
```

