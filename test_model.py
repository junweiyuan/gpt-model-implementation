import torch
from gpt_model import GPTModel
from tokenizer import SimpleTokenizer

print("=" * 60)
print("Testing GPT Model Implementation")
print("=" * 60)

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

print("\n1. Testing Tokenizer...")
tokenizer = SimpleTokenizer()
sample_text = "The quick brown fox jumps over the lazy dog. Machine learning is amazing!"
vocab_size = tokenizer.build_vocab(sample_text)
print(f"   Vocabulary size: {vocab_size}")

encoded = tokenizer.encode("the quick brown", add_special_tokens=False)
print(f"   Encoded 'the quick brown': {encoded}")

decoded = tokenizer.decode(encoded)
print(f"   Decoded back: {decoded}")

print("\n2. Creating GPT Model...")
model = GPTModel(
    vocab_size=vocab_size,
    d_model=128,
    num_heads=4,
    num_layers=2,
    d_ff=512,
    max_seq_length=64,
    dropout=0.1
)
print(f"   Model created successfully!")
print(f"   Total parameters: {model.count_parameters():,}")

print("\n3. Testing Forward Pass...")
batch_size = 2
seq_length = 10
dummy_input = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
model = model.to(device)

model.eval()
with torch.no_grad():
    logits, loss = model(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output logits shape: {logits.shape}")
    print(f"   Expected shape: ({batch_size}, {seq_length}, {vocab_size})")
    assert logits.shape == (batch_size, seq_length, vocab_size), "Output shape mismatch!"
    print(f"   ✓ Forward pass successful!")

print("\n4. Testing Text Generation...")
prompt_text = "the quick"
prompt_encoded = torch.tensor([tokenizer.encode(prompt_text, add_special_tokens=False)], dtype=torch.long).to(device)

generated = model.generate(
    prompt_encoded,
    max_new_tokens=10,
    temperature=1.0,
    top_k=10
)

generated_text = tokenizer.decode(generated[0].tolist())
print(f"   Prompt: '{prompt_text}'")
print(f"   Generated: {generated_text}")
print(f"   ✓ Generation successful!")

print("\n5. Testing Training Step...")
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

dummy_targets = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
logits, loss = model(dummy_input, dummy_targets)

print(f"   Loss value: {loss.item():.4f}")
print(f"   ✓ Loss computation successful!")

loss.backward()
optimizer.step()
print(f"   ✓ Backward pass and optimization successful!")

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
