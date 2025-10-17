# Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
```

## Run Tests

Verify the implementation works correctly:

```bash
python test_model.py
```

Expected output:
```
============================================================
Testing GPT Model Implementation
============================================================
...
All tests passed! ✓
```

## Basic Usage

### 1. Create and Train a Simple Model

```python
import torch
from gpt_model import GPTModel
from tokenizer import SimpleTokenizer
from trainer import Trainer, TextDataset

# Prepare your text data
text = """
Your training text goes here.
Can be multiple sentences and paragraphs.
"""

# Build vocabulary
tokenizer = SimpleTokenizer()
vocab_size = tokenizer.build_vocab(text)

# Encode text
encoded = tokenizer.encode(text, add_special_tokens=False)

# Create model
model = GPTModel(
    vocab_size=vocab_size,
    d_model=256,        # Embedding dimension
    num_heads=8,        # Number of attention heads
    num_layers=6,       # Number of transformer layers
    d_ff=1024,          # Feed-forward dimension
    max_seq_length=128, # Maximum sequence length
    dropout=0.1
)

print(f"Model has {model.count_parameters():,} parameters")

# Create dataset
dataset = TextDataset(encoded, block_size=64)

# Train
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    batch_size=16,
    learning_rate=3e-4,
    max_epochs=10
)

model = trainer.train()

# Save checkpoint
trainer.save_checkpoint('my_model.pt')
```

### 2. Generate Text

```python
# Set model to evaluation mode
model.eval()

# Prepare prompt
prompt = "The quick brown"
prompt_encoded = torch.tensor(
    [tokenizer.encode(prompt, add_special_tokens=False)],
    dtype=torch.long
)

# Generate
with torch.no_grad():
    generated = model.generate(
        prompt_encoded,
        max_new_tokens=50,
        temperature=0.8,    # Lower = more focused, Higher = more random
        top_k=40,           # Sample from top 40 tokens
        top_p=0.9           # Nucleus sampling threshold
    )

# Decode and print
text = tokenizer.decode(generated[0].tolist())
print(text)
```

### 3. Load a Saved Model

```python
# Create model with same configuration
model = GPTModel(
    vocab_size=vocab_size,
    d_model=256,
    num_heads=8,
    num_layers=6,
    d_ff=1024,
    max_seq_length=128,
    dropout=0.1
)

# Load checkpoint
checkpoint = torch.load('my_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## Run Example Training

The repository includes a complete example with sample text:

```bash
python example.py
```

This will:
1. Build a vocabulary from sample text
2. Train a small GPT model
3. Generate text from various prompts
4. Save the model checkpoint

## Model Configurations

### Tiny (for testing)
```python
model = GPTModel(
    vocab_size=vocab_size,
    d_model=128,
    num_heads=4,
    num_layers=2,
    d_ff=512,
    max_seq_length=64
)
# ~400K parameters
```

### Small (recommended for CPU)
```python
model = GPTModel(
    vocab_size=vocab_size,
    d_model=256,
    num_heads=8,
    num_layers=6,
    d_ff=1024,
    max_seq_length=256
)
# ~15M parameters
```

### Medium (requires GPU)
```python
model = GPTModel(
    vocab_size=vocab_size,
    d_model=512,
    num_heads=8,
    num_layers=8,
    d_ff=2048,
    max_seq_length=512
)
# ~60M parameters
```

### Large (GPT-2 Small equivalent)
```python
model = GPTModel(
    vocab_size=50000,
    d_model=768,
    num_heads=12,
    num_layers=12,
    d_ff=3072,
    max_seq_length=1024
)
# ~117M parameters
```

## Generation Parameters

### Temperature
- **0.1-0.5**: Very focused, deterministic
- **0.7-0.8**: Balanced (recommended)
- **1.0**: Original distribution
- **1.5-2.0**: Creative, random

### Top-k
- **1**: Always pick most likely (greedy)
- **10-50**: Moderate diversity (recommended)
- **100+**: High diversity
- **None**: No filtering

### Top-p (Nucleus Sampling)
- **0.5**: Very focused
- **0.9**: Balanced (recommended)
- **0.95-1.0**: More diverse
- **None**: No filtering

## Tips for Better Results

### Training
1. **Use GPU if available**: Training on CPU is slow
2. **Start with small model**: Test with tiny model first
3. **Monitor loss**: Should decrease steadily
4. **Use validation set**: Split data into train/val
5. **Save checkpoints**: Save regularly during training

### Data Preparation
1. **Clean your text**: Remove unwanted characters
2. **Sufficient data**: More data = better results
3. **Diverse content**: Varied text improves generalization
4. **Preprocessing**: Lowercase, normalize punctuation

### Generation
1. **Try different temperatures**: Experiment to find sweet spot
2. **Combine top-k and top-p**: Use both for best results
3. **Adjust max_new_tokens**: Longer sequences need more tokens
4. **Use good prompts**: Clear, specific prompts work best

## Troubleshooting

### Model not learning
- Increase model size
- Lower learning rate
- Train for more epochs
- Check data quality

### Out of memory
- Reduce batch_size
- Reduce max_seq_length
- Use smaller model
- Enable CPU offloading

### Generated text is gibberish
- Train for more epochs
- Use more/better training data
- Increase model size
- Check tokenizer vocabulary

### Training is slow
- Reduce model size
- Use GPU if available
- Increase batch size
- Reduce sequence length

## Next Steps

1. Read [README.md](README.md) for detailed documentation
2. Read [ARCHITECTURE.md](ARCHITECTURE.md) for architecture details
3. Experiment with different model sizes
4. Try training on your own text data
5. Implement custom tokenizers (BPE, WordPiece)
6. Add features like beam search, repetition penalty

## File Structure

```
chatgpt_model/
├── attention.py         # Multi-head attention implementation
├── transformer_block.py # Transformer block with FFN
├── gpt_model.py        # Complete GPT model
├── trainer.py          # Training utilities
├── tokenizer.py        # Simple tokenizer
├── example.py          # Training example
├── test_model.py       # Unit tests
├── requirements.txt    # Dependencies
├── README.md          # Full documentation
├── ARCHITECTURE.md    # Architecture details
└── QUICKSTART.md      # This file
```

## Resources

- Original Transformer paper: "Attention is All You Need"
- GPT paper: "Improving Language Understanding by Generative Pre-Training"
- GPT-2 paper: "Language Models are Unsupervised Multitask Learners"
- PyTorch documentation: https://pytorch.org/docs/
