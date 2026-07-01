import torch

from config import *
from tokenizer import CharacterTokenizer
from transformer import PalmeraTransformer

CHECKPOINT = CHECKPOINT_PATH

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu"
)

tokenizer = CharacterTokenizer.load(
    TOKENIZER_PATH
)

model = PalmeraTransformer(
    vocab_size=tokenizer.vocab_size
)

model.load_state_dict(
    checkpoint["model_state"]
)

model.eval()


while True:

    prompt = input("\nTú: ")

    if prompt.lower() in ["salir", "exit", "quit"]:
        break

    context = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long
    )

    output = model.generate(
        context,
        max_new_tokens=100
    )

    text = tokenizer.decode(
        output[0].tolist()
    )

    print(f"\npalmera: {text}")
