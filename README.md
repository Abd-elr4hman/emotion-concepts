# Emotion Concepts

Replication of Anthropic's [Emotion Concepts and their Function in a Large Language Model](https://transformer-circuits.pub/2026/emotions/index.html) research.

Extracts emotion vectors from TinyLlama (1.1B) by comparing hidden state activations between emotionally-charged and neutral text. Validates via steering and trajectory visualization.

![Similarity Heatmap](data/plots/similarity_heatmap.png)

## Requirements

- Python 3.11+
- ~2GB VRAM (TinyLlama FP16)

## Setup

```bash
pip install uv
uv sync
```

## Usage

```bash
python generate_stories.py   # Generate emotion story dataset
python extract_vectors.py    # Extract emotion vectors
python analyze_vectors.py    # Heatmap + PCA visualization
python steering.py           # Validate via causal steering
python trajectory.py         # Generate with emotion trajectory
```

## Todo

- [x] Generate emotion stories dataset
- [x] Extract emotion vectors (mean - global mean, project out confounds)
- [x] Cosine similarity heatmap with clustering
- [x] PCA 2D projection (valence/arousal axes)
- [x] Validate with causal steering
- [x] Trajectory visualization
- [ ] Support larger models
- [ ] Multi-layer analysis
- [ ] Manual story curation

## Limitations

- TinyLlama doesn't follow "don't use the word X" well — stories leak target emotion words, which means vectors may encode the literal word rather than the underlying concept
- No manual curation (Anthropic hand-picked stories for quality)
- Single layer extraction (layer 16 of 22 — ~70% depth where semantic concepts typically emerge) vs multi-layer analysis in paper

## More Information

For implementation details and methodology decisions, see [METHODOLOGY.md](METHODOLOGY.md).
