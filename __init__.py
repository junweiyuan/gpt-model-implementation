"""
ChatGPT Model Implementation

A complete PyTorch implementation of a GPT-style language model with transformer architecture.

Main Components:
- MultiHeadAttention: Scaled dot-product attention with multiple heads
- TransformerBlock: Attention + Feed-forward with residual connections
- GPTModel: Complete model with embeddings, positional encoding, and generation
- Trainer: Training utilities with optimization and scheduling
- SimpleTokenizer: Basic tokenizer for text processing

Example Usage:
    from gpt_model import GPTModel
    from tokenizer import SimpleTokenizer
    from trainer import Trainer, TextDataset
    
    model = GPTModel(vocab_size=10000, d_model=768, num_heads=12, num_layers=12)
    
    trainer = Trainer(model, train_dataset)
    model = trainer.train()
    
    output = model.generate(prompt, max_new_tokens=50)
"""

__version__ = "1.0.0"
__author__ = "ChatGPT Model Implementation"

from attention import MultiHeadAttention
from transformer_block import TransformerBlock, FeedForward
from gpt_model import GPTModel, PositionalEncoding
from trainer import Trainer, TextDataset
from tokenizer import SimpleTokenizer

__all__ = [
    'MultiHeadAttention',
    'TransformerBlock',
    'FeedForward',
    'GPTModel',
    'PositionalEncoding',
    'Trainer',
    'TextDataset',
    'SimpleTokenizer',
]
