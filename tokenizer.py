import re
from collections import Counter


class SimpleTokenizer:
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.vocab_size = 0
        
    def build_vocab(self, text, min_freq=1):
        tokens = self._tokenize(text)
        token_counts = Counter(tokens)
        
        self.vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        idx = 4
        
        for token, count in token_counts.items():
            if count >= min_freq:
                self.vocab[token] = idx
                idx += 1
        
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        
        return self.vocab_size
    
    def _tokenize(self, text):
        text = text.lower()
        tokens = re.findall(r'\b\w+\b|[.,!?;]', text)
        return tokens
    
    def encode(self, text, add_special_tokens=True):
        tokens = self._tokenize(text)
        encoded = []
        
        if add_special_tokens:
            encoded.append(self.vocab['<BOS>'])
        
        for token in tokens:
            encoded.append(self.vocab.get(token, self.vocab['<UNK>']))
        
        if add_special_tokens:
            encoded.append(self.vocab['<EOS>'])
        
        return encoded
    
    def decode(self, indices, skip_special_tokens=True):
        tokens = []
        special_tokens = {'<PAD>', '<UNK>', '<BOS>', '<EOS>'}
        
        for idx in indices:
            token = self.inverse_vocab.get(idx, '<UNK>')
            if skip_special_tokens and token in special_tokens:
                continue
            tokens.append(token)
        
        text = ' '.join(tokens)
        text = re.sub(r'\s+([.,!?;])', r'\1', text)
        
        return text
    
    def save_vocab(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            for token, idx in sorted(self.vocab.items(), key=lambda x: x[1]):
                f.write(f'{token}\t{idx}\n')
    
    def load_vocab(self, path):
        self.vocab = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                token, idx = line.strip().split('\t')
                self.vocab[token] = int(idx)
        
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)
