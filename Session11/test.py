import json
from tokenizers.basic import BasicTokenizer

def test_telugu_tokenization():
    # Test text
    test_text = "ఒక వ్యక్తిత్వం అంటూ ఏర్పడ్డాక ఆ రహస్యం తెలిస్తే లోతుగా గాయపడతారు."
    
    # Initialize tokenizer and load saved model
    tokenizer = BasicTokenizer()
    try:
        # Load the trained model
        tokenizer.load("models/telugu_bpe.model")
        
        # Load vocabulary from JSON
        with open('vocabulary.json', 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
            tokenizer.token_to_id = {k: int(v) for k, v in vocab_data['token_to_id'].items()}
            tokenizer.id_to_token = {int(k): v for k, v in vocab_data['id_to_token'].items()}
            # Convert merge strings back to tuples
            tokenizer.merges = {tuple(map(int, k.split(','))): int(v) 
                              for k, v in vocab_data['merges'].items()}
        
        print("Successfully loaded model and vocabulary")
    except FileNotFoundError as e:
        print(f"Error loading model or vocabulary: {e}")
        return
    
    # Test encoding
    encoded = tokenizer.encode(test_text)
    
    # Test decoding
    decoded = tokenizer.decode(encoded)
    
    # Calculate compression ratio
    original_size = len(test_text.encode('utf-8'))
    encoded_size = len(encoded) * 2  # 2 bytes per token
    compression_ratio = original_size / encoded_size
    
    # Print results
    print("\nTest Results:")
    print(f"Original text: {test_text}")
    print(f"Encoded: {encoded[:10]}... (showing first 10 tokens)")
    print(f"Decoded: {decoded}")
    print(f"Compression ratio: {compression_ratio:.2f}X")
    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    
    # Verify requirements
    try:
        assert len(tokenizer.vocab) <= 5000, "Vocabulary size exceeds 5000"
        assert compression_ratio >= 3.2, "Compression ratio below 3.2"
        assert decoded == test_text, "Decoded text doesn't match original"
        print("\nAll requirements met:")
        print("✓ Vocabulary size <= 5000")
        print("✓ Compression ratio >= 3.2")
        print("✓ Perfect reconstruction")
    except AssertionError as e:
        print(f"\nRequirement failed: {e}")

def test_multiple_sentences():
    test_sentences = [
        "అది ఒక అందమైన రోజు",
        "నేను తెలుగు నేర్చుకుంటున్నాను",
        "ప్రతి ఒక్కరూ సంతోషంగా ఉండాలి",
        "అరణ్యంలో రాముడు అనేక రాక్షసులను సంహరిస్తాడు. కానీ రావణుడు తన చెల్లెలు శూర్పణఖ కోసం ప్రతీకారంగా, సీతను అపహరించుకుపోతాడు. రావణుడు సీతను లంకకు తీసుకువెళ్తాడు. రాముడు సీతను వెతుకుతూ హనుమంతుడిని కలుసుకుంటాడు. హనుమంతుడు వానర సేనను ఏర్పాటు చేసి సీతకు సందేశం అందిస్తాడు"
        "పోయి పనీ చూసుకో, ఎవ్వర్లను అప్పెసి"
    ]
    
    tokenizer = BasicTokenizer()
    try:
        tokenizer.load("models/telugu_bpe.model")
        with open('vocabulary.json', 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
            tokenizer.token_to_id = {k: int(v) for k, v in vocab_data['token_to_id'].items()}
            tokenizer.id_to_token = {int(k): v for k, v in vocab_data['id_to_token'].items()}
            tokenizer.merges = {tuple(map(int, k.split(','))): int(v) 
                              for k, v in vocab_data['merges'].items()}
    except FileNotFoundError as e:
        print(f"Error loading model or vocabulary: {e}")
        return
    
    print("\nTesting multiple sentences:")
    for sentence in test_sentences:
        encoded = tokenizer.encode(sentence)
        decoded = tokenizer.decode(encoded)
        compression_ratio = len(sentence.encode('utf-8')) / (len(encoded) * 2)
        
        print(f"\nOriginal: {sentence}")
        print(f"Encoded: {encoded[:10]}... (first 10 tokens)")
        print(f"Decoded: {decoded}")
        print(f"Compression ratio: {compression_ratio:.2f}X")
        assert decoded == sentence, f"Decoding failed for: {sentence}"

if __name__ == "__main__":
    print("Testing single long sentence...")
    test_telugu_tokenization()
    
    print("\nTesting multiple sentences...")
    test_multiple_sentences()
