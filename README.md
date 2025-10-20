# GPT Model Implementation & Mobile GUI Agent Framework

This repository contains two projects:

1. **GPT Model Implementation**: A PyTorch implementation of a GPT-style language model
2. **Mobile GUI Agent Framework**: A framework for mobile GUI automation using reasoning-enabled VLMs

---

## 1. Mobile GUI Agent Framework

A mobile GUI agent framework that uses reasoning-enabled Vision-Language Models (VLMs) for automated smartphone control, based on the paper "Does Chain-of-Thought Reasoning Help Mobile GUI Agent? An Empirical Study".

### Features

- Support for reasoning-enabled VLMs (Claude 3.7 Sonnet, Gemini 2.0 Flash, GPT-4)
- Multiple prompting strategies (Set-of-Mark, Accessibility Tree)
- GUI element grounding with coordinate prediction
- Action execution for mobile automation
- Chain-of-thought reasoning for complex tasks

### Installation

```bash
pip install -r requirements.txt
```

### Usage

```python
from mobile_gui_agent import MobileGUIAgent
from mobile_gui_agent.vlm import ClaudeVLM

# Initialize agent with reasoning-enabled VLM
agent = MobileGUIAgent(
    vlm=ClaudeVLM(model="claude-3-7-sonnet", use_reasoning=True),
    prompting_strategy="set_of_mark"
)

# Execute a task
result = agent.execute_task(
    screenshot_path="screenshot.png",
    task_instruction="Set my DM Spam filter to 'Do not filter direct messages' on Discord app"
)

print(f"Action: {result['action']}")
print(f"Reasoning: {result['reasoning']}")
```

### Architecture

- `mobile_gui_agent/`: Core framework
  - `agent.py`: Main agent implementation
  - `vlm/`: VLM provider implementations
  - `grounding.py`: GUI element grounding
  - `actions.py`: Action execution
  - `prompting.py`: Prompting strategies

### Supported VLMs

- Claude 3.7 Sonnet (with/without reasoning)
- Gemini 2.0 Flash (with/without reasoning)
- GPT-4o

---

## 2. GPT Model Implementation

A PyTorch implementation of a GPT-style language model with transformer architecture. This implementation includes multi-head attention, transformer blocks, positional encodings, and text generation capabilities.

### Architecture Overview

This implementation follows the GPT (Generative Pre-trained Transformer) architecture with the following key components:

#### Multi-Head Attention
The attention mechanism allows the model to focus on different parts of the input sequence when making predictions. The scaled dot-product attention is computed as:

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

Multi-head attention runs multiple attention operations in parallel and concatenates their outputs, allowing the model to attend to information from different representation subspaces.

#### Transformer Block
Each transformer block consists of:
- Multi-head self-attention layer with residual connection
- Layer normalization
- Feed-forward network (with GELU activation) with residual connection
- Layer normalization
- Dropout for regularization

#### Positional Encoding
Since transformers don't have inherent notion of sequence order, positional encodings are added to the input embeddings using sine and cosine functions:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### Files

- `attention.py`: Multi-head attention mechanism implementation
- `transformer_block.py`: Transformer block with feed-forward network
- `gpt_model.py`: Complete GPT model with embeddings, positional encoding, and generation
- `trainer.py`: Training utilities with optimizer, learning rate scheduling, and validation
- `tokenizer.py`: Simple tokenizer for text processing
- `example.py`: Example usage for training and inference

### Model Configuration

Default model parameters:
- `d_model`: 768 (embedding dimension)
- `num_heads`: 12 (number of attention heads)
- `num_layers`: 12 (number of transformer blocks)
- `d_ff`: 3072 (feed-forward network dimension)
- `max_seq_length`: 1024 (maximum sequence length)
- `dropout`: 0.1

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

## License

MIT
