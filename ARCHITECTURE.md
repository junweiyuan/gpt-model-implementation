# GPT Model Architecture Documentation

## Overview

This document provides a detailed explanation of the GPT (Generative Pre-trained Transformer) model architecture implemented in this project.

## Architecture Diagram

```
Input Text
    ↓
[Tokenization]
    ↓
Token IDs: [t1, t2, t3, ..., tn]
    ↓
┌─────────────────────────────────────────┐
│         Token Embedding Layer           │
│  Maps token IDs to dense vectors        │
│  Shape: (vocab_size, d_model)           │
└─────────────────────────────────────────┘
    ↓
    + (Element-wise addition)
    ↓
┌─────────────────────────────────────────┐
│      Positional Encoding Layer          │
│  Adds position information              │
│  Uses sine/cosine functions             │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│          Dropout Layer                  │
└─────────────────────────────────────────┘
    ↓
┌═════════════════════════════════════════┐
║   Transformer Block 1                   ║
║  ┌───────────────────────────────────┐  ║
║  │   Multi-Head Self-Attention       │  ║
║  │   - Query, Key, Value projections │  ║
║  │   - Scaled dot-product attention  │  ║
║  │   - Causal masking applied        │  ║
║  │   - num_heads parallel attention  │  ║
║  └───────────────────────────────────┘  ║
║         ↓                                ║
║  [Add & Norm] (Residual + LayerNorm)    ║
║         ↓                                ║
║  ┌───────────────────────────────────┐  ║
║  │   Feed-Forward Network            │  ║
║  │   - Linear (d_model → d_ff)       │  ║
║  │   - GELU activation               │  ║
║  │   - Dropout                        │  ║
║  │   - Linear (d_ff → d_model)       │  ║
║  └───────────────────────────────────┘  ║
║         ↓                                ║
║  [Add & Norm] (Residual + LayerNorm)    ║
╚═════════════════════════════════════════╝
    ↓
┌═════════════════════════════════════════┐
║   Transformer Block 2                   ║
║   (Same structure as Block 1)           ║
╚═════════════════════════════════════════╝
    ↓
    ...
    ↓
┌═════════════════════════════════════════┐
║   Transformer Block N                   ║
║   (Same structure as Block 1)           ║
╚═════════════════════════════════════════╝
    ↓
┌─────────────────────────────────────────┐
│       Final Layer Normalization         │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│      Output Projection Layer            │
│  Linear: d_model → vocab_size           │
└─────────────────────────────────────────┘
    ↓
Logits: [vocab_size probabilities for each position]
    ↓
[Softmax] (during generation)
    ↓
Next Token Prediction
```

## Component Details

### 1. Token Embedding

Converts discrete token IDs into continuous vector representations.

**Input**: Token IDs `[batch_size, seq_length]`
**Output**: Embeddings `[batch_size, seq_length, d_model]`

```python
embedding = TokenEmbedding(vocab_size, d_model)
x = embedding(token_ids) * sqrt(d_model)  # Scaled by sqrt(d_model)
```

### 2. Positional Encoding

Injects positional information into the model since transformers are position-agnostic.

**Formula**:
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Properties**:
- Fixed (not learned)
- Allows model to learn relative positions
- Different frequencies for different dimensions

### 3. Multi-Head Attention

The core mechanism that allows the model to focus on different parts of the input.

**Architecture**:
```
Input: x [batch_size, seq_length, d_model]
    ↓
┌────────────────────────────────────────────┐
│  Linear Projections (W_q, W_k, W_v)        │
│  Q = x @ W_q                               │
│  K = x @ W_k                               │
│  V = x @ W_v                               │
└────────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────────┐
│  Split into num_heads                      │
│  Shape: [batch, heads, seq_len, d_k]       │
│  where d_k = d_model / num_heads           │
└────────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────────┐
│  Scaled Dot-Product Attention              │
│  scores = (Q @ K^T) / sqrt(d_k)            │
│  scores = scores.masked_fill(mask, -inf)   │
│  attn = softmax(scores, dim=-1)            │
│  output = attn @ V                         │
└────────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────────┐
│  Concatenate heads & project               │
│  output = concat(heads) @ W_o              │
└────────────────────────────────────────────┘
    ↓
Output: [batch_size, seq_length, d_model]
```

**Causal Mask**:
```
For seq_length = 4:
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]

Prevents position i from attending to positions > i
```

### 4. Feed-Forward Network

Two-layer MLP with GELU activation.

**Structure**:
```
x → Linear(d_model → d_ff) → GELU → Dropout → Linear(d_ff → d_model)
```

**Typical ratio**: `d_ff = 4 * d_model`

**GELU Activation**: Gaussian Error Linear Unit
```
GELU(x) = x * Φ(x)
where Φ(x) is the cumulative distribution function of standard normal distribution
```

### 5. Layer Normalization

Normalizes activations across the feature dimension.

**Formula**:
```
LayerNorm(x) = γ * (x - μ) / sqrt(σ² + ε) + β

where:
- μ = mean(x) across features
- σ² = variance(x) across features
- γ, β are learned parameters
- ε = small constant for numerical stability
```

### 6. Residual Connections

Skip connections that add the input to the output of each sub-layer.

**Pattern**:
```
x = x + SubLayer(x)
x = LayerNorm(x)
```

**Benefits**:
- Enables training of very deep networks
- Helps gradient flow during backpropagation
- Provides identity mapping pathway

## Training Process

### Loss Computation

Cross-entropy loss for next-token prediction:

```python
# Forward pass
logits, loss = model(input_ids, target_ids)

# Loss calculation
loss = CrossEntropy(
    logits.view(-1, vocab_size),
    targets.view(-1)
)
```

### Optimization

**Optimizer**: AdamW (Adam with weight decay)

**Parameters**:
- Learning rate: 3e-4
- Betas: (0.9, 0.95)
- Weight decay: 0.01

**Learning Rate Schedule**:
1. Linear warmup (first N steps)
2. Cosine annealing decay

```
     Learning Rate
         ^
         |     ╱────╲
         |    ╱      ╲
         |   ╱        ╲___
         |  ╱
         | ╱
         |╱________________> Steps
         Warmup   Cosine Decay
```

### Gradient Clipping

Clips gradient norm to prevent exploding gradients:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## Text Generation

### Autoregressive Generation

Generates text one token at a time:

```
1. Start with prompt: [t1, t2, t3]
2. Predict next token: t4 = model([t1, t2, t3])
3. Append: [t1, t2, t3, t4]
4. Predict next: t5 = model([t1, t2, t3, t4])
5. Repeat...
```

### Sampling Strategies

**1. Temperature Sampling**

Controls randomness of predictions:
```python
logits = logits / temperature
probs = softmax(logits)
next_token = sample(probs)
```

- `temperature < 1`: More deterministic (sharper distribution)
- `temperature = 1`: Original distribution
- `temperature > 1`: More random (flatter distribution)

**2. Top-k Sampling**

Only sample from k most likely tokens:
```python
top_k_logits, top_k_indices = torch.topk(logits, k)
# Set other logits to -inf
next_token = sample(softmax(top_k_logits))
```

**3. Top-p (Nucleus) Sampling**

Sample from smallest set of tokens with cumulative probability ≥ p:
```python
sorted_probs = sort(softmax(logits), descending=True)
cumsum = cumulative_sum(sorted_probs)
# Keep tokens until cumsum > p
next_token = sample(from_nucleus)
```

## Model Configurations

### GPT-2 Small (Default)
- Parameters: ~117M
- Layers: 12
- d_model: 768
- num_heads: 12
- d_ff: 3072
- Vocabulary: ~50k

### GPT-2 Medium
- Parameters: ~345M
- Layers: 24
- d_model: 1024
- num_heads: 16
- d_ff: 4096

### GPT-2 Large
- Parameters: ~774M
- Layers: 36
- d_model: 1280
- num_heads: 20
- d_ff: 5120

### GPT-2 XL
- Parameters: ~1.5B
- Layers: 48
- d_model: 1600
- num_heads: 25
- d_ff: 6400

## Memory Considerations

### Parameter Memory

```
Total Parameters ≈ 
  + vocab_size × d_model (token embeddings)
  + max_seq_length × d_model (positional encodings)
  + num_layers × (
      4 × d_model² (attention projections)
      + 2 × d_model × d_ff (FFN)
      + 4 × d_model (layer norms)
    )
  + d_model × vocab_size (output projection)
```

### Activation Memory (Forward Pass)

Per layer per sequence:
```
Memory ≈ batch_size × seq_length × (
  d_model (residual)
  + num_heads × seq_length (attention scores)
  + d_model (attention output)
  + d_ff (FFN intermediate)
)
```

## Key Implementation Details

### Initialization

**Linear Layers**: `N(0, 0.02)`
**Embeddings**: `N(0, 0.02)`
**Layer Norms**: `γ=1, β=0`

### Attention Scaling

Dot product is scaled by `1/sqrt(d_k)` to prevent:
- Softmax saturation
- Gradient vanishing
- Numerical instability

### Causal Masking

Implemented as upper triangular mask filled with `-inf`:
```python
mask = torch.tril(torch.ones(seq_len, seq_len))
scores = scores.masked_fill(mask == 0, -1e9)
```

## Training Best Practices

1. **Start small**: Test with small model first
2. **Gradient clipping**: Essential for stability
3. **Learning rate warmup**: Prevents early instability
4. **Batch size**: As large as memory allows
5. **Sequence length**: Start shorter, increase gradually
6. **Validation**: Monitor for overfitting
7. **Checkpointing**: Save regularly during training

## Common Issues and Solutions

### Issue: Loss not decreasing
- Check learning rate (try smaller)
- Verify gradient flow (check for NaN)
- Ensure data is shuffled
- Check for label leakage

### Issue: Out of memory
- Reduce batch size
- Reduce sequence length
- Use gradient checkpointing
- Enable mixed precision training

### Issue: Generated text is repetitive
- Use nucleus (top-p) sampling
- Increase temperature
- Add repetition penalty
- Ensure diverse training data

### Issue: Training instability
- Enable gradient clipping
- Reduce learning rate
- Increase warmup steps
- Check for NaN in data

## References

- Vaswani et al. (2017): "Attention is All You Need"
- Radford et al. (2018): "Improving Language Understanding by Generative Pre-Training"
- Radford et al. (2019): "Language Models are Unsupervised Multitask Learners"
