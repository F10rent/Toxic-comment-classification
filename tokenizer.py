import regex as re
import json

class Tokenizer:
    def __init__(self, vocab=None):
        self.vocab = vocab if vocab is not None else {}
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.vocab = {self.unk_token: 0, self.pad_token: 1}
        # `vocab` isn't empty
        if vocab:
            for i, word in enumerate(vocab, start=2):  # index start with 2
                self.vocab[word] = i
        
            

    def change_vocab(self, vocab):
        for i, word in enumerate(vocab, start=2):  # index start with 2
            self.vocab[word] = i
        

    def get_vacab_size(self):
        return len(self.vocab)

    def tokenize(self, text: str):
        """
        Tokenize a text using a regex pattern.

        :param text: A text data.
        :return: A list of tokens
        """
        text = re.sub(r"[^a-zA-Z0-9!?.,\s']", "", text)
        tokens = re.findall(self.pat, text)
        tokens = [token.lower() for token in tokens]
        return tokens
        

    def encode(self, text: str):
        tokens = self.tokenize(text)
        return [self.vocab.get(token, 0) for token in tokens]
    
    def token_to_id(self, token: str):
        return self.vocab.get(token, self.vocab[self.unk_token])
    
    def save_vocab(self, file_path="vocab.json"):
        with open(file_path, "w") as f:
            json.dump(self.vocab, f)

    def load_vocab(self, file_path="vocab.json"):
        with open(file_path, "r") as f:
            self.vocab = json.load(f)
        
    