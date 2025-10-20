import torch
from gpt_model import GPTModel
from tokenizer import SimpleTokenizer
from trainer import Trainer, TextDataset


def train_example():
    sample_text = """
    The quick brown fox jumps over the lazy dog. This is a sample text for training a language model.
    Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.
    Natural language processing is a field of study that focuses on the interaction between computers and human language.
    Deep learning has revolutionized many fields including computer vision, speech recognition, and natural language processing.
    Transformers are a type of neural network architecture that has become very popular for language modeling tasks.
    The attention mechanism allows the model to focus on different parts of the input when making predictions.
    GPT stands for Generative Pre-trained Transformer and is a powerful language model architecture.
    Training large language models requires significant computational resources and large amounts of data.
    Fine-tuning pre-trained models on specific tasks often yields better results than training from scratch.
    Language models can be used for various tasks such as text generation, translation, and question answering.
    """
    
    print("Building tokenizer vocabulary...")
    tokenizer = SimpleTokenizer()
    vocab_size = tokenizer.build_vocab(sample_text, min_freq=1)
    print(f"Vocabulary size: {vocab_size}")
    
    print("\nEncoding text...")
    encoded_text = tokenizer.encode(sample_text, add_special_tokens=False)
    print(f"Encoded text length: {len(encoded_text)}")
    
    print("\nCreating model...")
    model = GPTModel(
        vocab_size=vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=6,
        d_ff=1024,
        max_seq_length=128,
        dropout=0.1
    )
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    print("\nCreating datasets...")
    block_size = 32
    train_data = encoded_text[:int(0.9 * len(encoded_text))]
    val_data = encoded_text[int(0.9 * len(encoded_text)):]
    
    train_dataset = TextDataset(train_data, block_size)
    val_dataset = TextDataset(val_data, block_size)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    print("\nTraining model...")
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=8,
        learning_rate=3e-4,
        max_epochs=50,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    model = trainer.train()
    
    print("\nGenerating text...")
    model.eval()
    
    prompt = "the quick brown"
    prompt_encoded = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], dtype=torch.long)
    
    if torch.cuda.is_available():
        prompt_encoded = prompt_encoded.cuda()
    
    generated = model.generate(
        prompt_encoded,
        max_new_tokens=50,
        temperature=0.8,
        top_k=40
    )
    
    generated_text = tokenizer.decode(generated[0].tolist())
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated_text}")
    
    trainer.save_checkpoint('model_checkpoint.pt')


def inference_example():
    print("Loading model for inference...")
    
    sample_text = "The quick brown fox jumps over the lazy dog."
    
    tokenizer = SimpleTokenizer()
    vocab_size = tokenizer.build_vocab(sample_text)
    
    model = GPTModel(
        vocab_size=vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=6,
        d_ff=1024,
        max_seq_length=128,
        dropout=0.1
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    prompts = [
        "the quick",
        "machine learning",
        "natural language"
    ]
    
    print("\nGenerating text from different prompts:\n")
    
    for prompt in prompts:
        prompt_encoded = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], dtype=torch.long).to(device)
        
        generated = model.generate(
            prompt_encoded,
            max_new_tokens=30,
            temperature=0.7,
            top_k=50
        )
        
        generated_text = tokenizer.decode(generated[0].tolist())
        print(f"Prompt: '{prompt}'")
        print(f"Generated: {generated_text}\n")


if __name__ == "__main__":
    print("=" * 60)
    print("GPT Model Training Example")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    train_example()
