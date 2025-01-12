from bpe import Tokenizer, get_stats, merge
import json


class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
        self.token_to_id = {}
        self.id_to_token = {}

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256 and vocab_size <= 5000
        num_merges = vocab_size - 256

        text_bytes = text.encode('utf-8')
        ids = list(text_bytes)

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        
        # Initialize token mappings
        self.token_to_id = {bytes([idx]).decode('utf-8', errors='replace'): idx for idx in range(256)}
        self.id_to_token = {idx: bytes([idx]).decode('utf-8', errors='replace') for idx in range(256)}

        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            
            # Update token mappings
            token = vocab[idx].decode('utf-8', errors='replace')
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            
            if verbose and i % 20 == 0:
                print(f"merge {i + 1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        self.merges = merges
        self.vocab = vocab
        self._save_vocabulary()

    def _save_vocabulary(self):
        vocabulary = {
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token,
            'merges': {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        }
        with open('vocabulary.json', 'w', encoding='utf-8') as f:
            json.dump(vocabulary, f, ensure_ascii=False, indent=2)

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break

            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids
