#!/usr/bin/env python3
"""
Interactive demo of the GPT model implementation.
Shows the complete workflow from training to generation.
"""

import torch
from gpt_model import GPTModel
from tokenizer import SimpleTokenizer
from trainer import Trainer, TextDataset


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    print_section("GPT Model Implementation Demo")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    sample_text = """
    Once upon a time, in a world of artificial intelligence, there lived a powerful language model.
    This model could understand and generate human-like text with remarkable accuracy.
    The model was built using transformer architecture, which revolutionized natural language processing.
    Deep learning algorithms enabled the model to learn patterns from vast amounts of text data.
    Machine learning researchers worked tirelessly to improve the model's capabilities.
    Natural language understanding became more sophisticated with each iteration.
    The attention mechanism allowed the model to focus on relevant parts of the input.
    Training large neural networks required significant computational resources and expertise.
    The model could perform various tasks such as translation, summarization, and question answering.
    Artificial intelligence continued to advance, bringing new possibilities for human-computer interaction.
    Language models became essential tools for many applications in modern technology.
    The future of AI looked promising as models grew more capable and efficient.
    """
    
    print_section("Step 1: Building Tokenizer")
    
    tokenizer = SimpleTokenizer()
    vocab_size = tokenizer.build_vocab(sample_text, min_freq=1)
    
    print(f"\n✓ Vocabulary built successfully")
    print(f"  - Vocabulary size: {vocab_size}")
    print(f"  - Special tokens: <PAD>, <UNK>, <BOS>, <EOS>")
    
    sample_encode = "artificial intelligence"
    encoded = tokenizer.encode(sample_encode, add_special_tokens=False)
    decoded = tokenizer.decode(encoded)
    
    print(f"\n  Example encoding:")
    print(f"  - Original: '{sample_encode}'")
    print(f"  - Encoded: {encoded}")
    print(f"  - Decoded: '{decoded}'")
    
    print_section("Step 2: Creating Model Architecture")
    
    model_config = {
        'vocab_size': vocab_size,
        'd_model': 256,
        'num_heads': 8,
        'num_layers': 4,
        'd_ff': 1024,
        'max_seq_length': 128,
        'dropout': 0.1
    }
    
    model = GPTModel(**model_config)
    
    print(f"\n✓ Model created successfully")
    print(f"  - Total parameters: {model.count_parameters():,}")
    print(f"\n  Model configuration:")
    for key, value in model_config.items():
        print(f"    {key}: {value}")
    
    print_section("Step 3: Preparing Training Data")
    
    encoded_text = tokenizer.encode(sample_text, add_special_tokens=False)
    
    block_size = 32
    train_dataset = TextDataset(encoded_text, block_size)
    
    print(f"\n✓ Datasets created successfully")
    print(f"  - Encoded text length: {len(encoded_text)} tokens")
    print(f"  - Block size: {block_size}")
    print(f"  - Train dataset: {len(train_dataset)} samples")
    print(f"  - Validation: Using training data (demo purposes)")
    
    print_section("Step 4: Training Model")
    
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=None,
        batch_size=4,
        learning_rate=3e-4,
        max_epochs=20,
        device=device,
        warmup_steps=50,
        grad_clip=1.0
    )
    
    print("\n✓ Trainer initialized")
    print(f"  - Batch size: 4")
    print(f"  - Learning rate: 3e-4")
    print(f"  - Max epochs: 20")
    print(f"  - Warmup steps: 50")
    print(f"  - Gradient clipping: 1.0")
    
    print("\n[Training in progress...]")
    model = trainer.train()
    
    print_section("Step 5: Generating Text")
    
    prompts = [
        "artificial intelligence",
        "the model",
        "language models",
        "deep learning"
    ]
    
    model.eval()
    
    print("\n✓ Generating text from multiple prompts\n")
    
    for i, prompt in enumerate(prompts, 1):
        print(f"  Example {i}:")
        print(f"  Prompt: '{prompt}'")
        
        prompt_encoded = torch.tensor(
            [tokenizer.encode(prompt, add_special_tokens=False)],
            dtype=torch.long
        ).to(device)
        
        with torch.no_grad():
            generated = model.generate(
                prompt_encoded,
                max_new_tokens=25,
                temperature=0.8,
                top_k=40,
                top_p=0.9
            )
        
        generated_text = tokenizer.decode(generated[0].tolist())
        print(f"  Generated: {generated_text}")
        print()
    
    print_section("Step 6: Different Sampling Strategies")
    
    test_prompt = "artificial intelligence"
    prompt_encoded = torch.tensor(
        [tokenizer.encode(test_prompt, add_special_tokens=False)],
        dtype=torch.long
    ).to(device)
    
    strategies = [
        ("Greedy (temp=0.1)", {"temperature": 0.1, "top_k": 1}),
        ("Balanced (temp=0.8)", {"temperature": 0.8, "top_k": 40}),
        ("Creative (temp=1.5)", {"temperature": 1.5, "top_k": None}),
        ("Top-p Sampling", {"temperature": 1.0, "top_p": 0.9}),
    ]
    
    print(f"\n✓ Comparing sampling strategies for prompt: '{test_prompt}'\n")
    
    for strategy_name, params in strategies:
        with torch.no_grad():
            generated = model.generate(
                prompt_encoded,
                max_new_tokens=20,
                **params
            )
        
        generated_text = tokenizer.decode(generated[0].tolist())
        print(f"  {strategy_name}:")
        print(f"  → {generated_text}")
        print()
    
    print_section("Step 7: Model Analysis")
    
    print(f"\n✓ Model Statistics:")
    print(f"  - Total parameters: {model.count_parameters():,}")
    print(f"  - Embedding parameters: {vocab_size * model_config['d_model']:,}")
    print(f"  - Transformer parameters: ~{model.count_parameters() - vocab_size * model_config['d_model']:,}")
    print(f"  - Model size (approx): {model.count_parameters() * 4 / 1024 / 1024:.2f} MB (fp32)")
    
    print(f"\n✓ Architecture Details:")
    print(f"  - Number of layers: {model_config['num_layers']}")
    print(f"  - Attention heads per layer: {model_config['num_heads']}")
    print(f"  - Hidden dimension: {model_config['d_model']}")
    print(f"  - FFN dimension: {model_config['d_ff']}")
    print(f"  - Context window: {model_config['max_seq_length']} tokens")
    
    print_section("Demo Complete!")
    
    print("\n✓ All steps completed successfully!")
    print("\nNext steps:")
    print("  1. Try training with your own text data")
    print("  2. Experiment with different model sizes")
    print("  3. Adjust generation parameters for better results")
    print("  4. Read ARCHITECTURE.md for detailed explanations")
    print("  5. Read QUICKSTART.md for more examples")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
