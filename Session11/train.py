import os
import time
from tokenizers.basic import BasicTokenizer
from dataset import get_clean_data, prepare_data
from utils import get_unique_chars

def train_tokenizer():
    # Create models directory
    os.makedirs("models", exist_ok=True)
    prepare_data()
    # Get clean Telugu text
    text = get_clean_data()
    
    # Print initial statistics
    vocab_size = get_unique_chars(text)
    print("Initial unique characters:", vocab_size)
    print("Text length:", len(text))
    print("Words length:", len(text.split()))
    print('Sample words:', text.split()[:10])

    # Initialize and train tokenizer
    print("\nTraining BPE tokenizer...")
    t0 = time.time()
    
    tokenizer = BasicTokenizer()
    target_vocab_size = 4800  # Keep under 5000
    tokenizer.train(text, target_vocab_size, verbose=True)
    
    # Save the model
    prefix = os.path.join("models", "telugu_bpe")
    tokenizer.save(prefix)
    
    t1 = time.time()
    print(f"\nTraining completed in {t1 - t0:.2f} seconds")

    # Test the tokenizer
    test_text = "చెట్టు పెరగాలంటే విత్తనం నాటాలి"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    # Calculate compression ratio
    original_size = len(test_text.encode('utf-8'))
    encoded_size = len(encoded) * 2
    compression_ratio = original_size / encoded_size
    
    print("\nTest Results:")
    print(f"Test text: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Compression ratio: {compression_ratio:.2f}X")
    print(f"Final vocabulary size: {len(tokenizer.vocab)}")

if __name__ == "__main__":
    train_tokenizer()
