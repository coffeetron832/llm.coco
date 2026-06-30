import json


class CharacterTokenizer:

    def __init__(self, text=None):

        self.chars = []

        self.stoi = {}

        self.itos = {}

        self.vocab_size = 0

        if text is not None:
            self.build(text)

    def build(self, text):

        self.chars = sorted(
            list(set(text))
        )

        self.stoi = {
            ch: i
            for i, ch in enumerate(self.chars)
        }

        self.itos = {
            i: ch
            for i, ch in enumerate(self.chars)
        }

        self.vocab_size = len(
            self.chars
        )

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

    def save(self, path):

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                {
                    "chars": self.chars
                },
                f,
                ensure_ascii=False,
                indent=4
            )

    @classmethod
    def load(cls, path):

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        tokenizer = cls()

        tokenizer.chars = data["chars"]

        tokenizer.stoi = {
            ch: i
            for i, ch in enumerate(
                tokenizer.chars
            )
        }

        tokenizer.itos = {
            i: ch
            for i, ch in enumerate(
                tokenizer.chars
            )
        }

        tokenizer.vocab_size = len(
            tokenizer.chars
        )

        return tokenizer
