# ChatGPT Model Implementation

A PyTorch implementation of a GPT-style language model with transformer architecture. This implementation includes multi-head attention, transformer blocks, positional encodings, and text generation capabilities.

## Architecture Overview

This implementation follows the GPT (Generative Pre-trained Transformer) architecture with the following key components:

### Multi-Head Attention
The attention mechanism allows the model to focus on different parts of the input sequence when making predictions. The scaled dot-product attention is computed as:

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

Multi-head attention runs multiple attention operations in parallel and concatenates their outputs, allowing the model to attend to information from different representation subspaces.

### Transformer Block
Each transformer block consists of:
- Multi-head self-attention layer with residual connection
- Layer normalization
- Feed-forward network (with GELU activation) with residual connection
- Layer normalization
- Dropout for regularization

### Positional Encoding
Since transformers don't have inherent notion of sequence order, positional encodings are added to the input embeddings using sine and cosine functions:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### GPT Model Architecture
The complete model consists of:
- Token embeddings
- Positional encodings
- Stack of N transformer blocks
- Layer normalization
- Output projection to vocabulary size

## Files

- `attention.py`: Multi-head attention mechanism implementation
- `transformer_block.py`: Transformer block with feed-forward network
- `gpt_model.py`: Complete GPT model with embeddings, positional encoding, and generation
- `trainer.py`: Training utilities with optimizer, learning rate scheduling, and validation
- `tokenizer.py`: Simple tokenizer for text processing
- `example.py`: Example usage for training and inference

## Model Configuration

Default model parameters:
- `d_model`: 768 (embedding dimension)
- `num_heads`: 12 (number of attention heads)
- `num_layers`: 12 (number of transformer blocks)
- `d_ff`: 3072 (feed-forward network dimension)
- `max_seq_length`: 1024 (maximum sequence length)
- `dropout`: 0.1

## Usage

### Training a Model

```python
import torch
from gpt_model import GPTModel
from tokenizer import SimpleTokenizer
from trainer import Trainer, TextDataset

# Prepare data
text = "Your training text here..."
tokenizer = SimpleTokenizer()
vocab_size = tokenizer.build_vocab(text)
encoded_text = tokenizer.encode(text, add_special_tokens=False)

# Create model
model = GPTModel(
    vocab_size=vocab_size,
    d_model=768,
    num_heads=12,
    num_layers=12,
    d_ff=3072,
    max_seq_length=1024,
    dropout=0.1
)

# Create dataset
train_dataset = TextDataset(encoded_text, block_size=128)

# Train
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    batch_size=32,
    learning_rate=3e-4,
    max_epochs=10
)
model = trainer.train()
```

### Generating Text

```python
# Prepare prompt
prompt = "The quick brown"
prompt_encoded = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)])

# Generate
model.eval()
generated = model.generate(
    prompt_encoded,
    max_new_tokens=50,
    temperature=0.8,
    top_k=40,
    top_p=0.9
)

# Decode
generated_text = tokenizer.decode(generated[0].tolist())
print(generated_text)
```

## Text Generation Strategies

The model supports several sampling strategies:

1. **Temperature Sampling**: Controls randomness of predictions (lower = more deterministic, higher = more random)

2. **Top-k Sampling**: Only samples from the k most likely tokens

3. **Top-p (Nucleus) Sampling**: Samples from the smallest set of tokens whose cumulative probability exceeds p

## Training Features

- **AdamW Optimizer**: With configurable learning rate and weight decay
- **Learning Rate Warmup**: Gradual learning rate increase at the start
- **Cosine Annealing Schedule**: Smooth learning rate decay
- **Gradient Clipping**: Prevents exploding gradients
- **Validation**: Optional validation set for monitoring overfitting
- **Checkpointing**: Save and load model states

## Requirements

```
torch>=2.0.0
tqdm
```

## Running the Example

```bash
python example.py
```

This will:
1. Build a vocabulary from sample text
2. Create and train a small GPT model
3. Generate text from prompts
4. Save a model checkpoint

## Model Size

The parameter count for the default configuration (GPT-2 small equivalent):
- Embedding parameters: vocab_size × d_model
- Transformer parameters: ~85M (for 12 layers, 768 dim, 12 heads)
- Total: ~117M parameters (with 50k vocab)

## Notes

- This is an educational implementation focusing on clarity and understanding
- For production use, consider using more sophisticated tokenizers (BPE, SentencePiece)
- Training large models requires significant computational resources (GPU recommended)
- The simple tokenizer is basic; real applications should use subword tokenization
- Causal masking ensures autoregressive generation (each position can only attend to previous positions)

## Key Concepts

### Autoregressive Generation
The model generates text one token at a time, using previously generated tokens as context for predicting the next token.

### Causal Masking
A triangular mask prevents the model from "seeing" future tokens during training, ensuring it learns to predict based only on past context.

### Layer Normalization
Normalizes activations across features, stabilizing training and improving convergence.

### Residual Connections
Skip connections that add the input of a layer to its output, enabling training of very deep networks.

## Future Enhancements

Potential improvements for this implementation:
- Implement more advanced tokenizers (BPE, WordPiece)
- Add support for distributed training
- Implement gradient checkpointing for memory efficiency
- Add beam search for generation
- Implement different attention variants (flash attention, sparse attention)
- Add support for fine-tuning on specific tasks
- Implement model parallelism for larger models
