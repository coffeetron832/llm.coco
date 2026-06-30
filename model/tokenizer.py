class CharacterTokenizer:

    def __init__(self, text):

        self.chars = sorted(list(set(text)))

        self.vocab_size = len(self.chars)

        self.stoi = {
            ch: i
            for i, ch in enumerate(self.chars)
        }

        self.itos = {
            i: ch
            for i, ch in enumerate(self.chars)
        }

    def encode(self, text):

        return [
            self.stoi[c]
            for c in text
            if c in self.stoi
        ]

    def decode(self, tokens):

        return "".join(
            self.itos[t]
            for t in tokens
            if t in self.itos
        )
