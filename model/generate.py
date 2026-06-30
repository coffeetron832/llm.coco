import torch

from tokenizer import CharacterTokenizer
from transformer import CocoTransformer


CHECKPOINT = "../checkpoints/coco_v0.pt"

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu"
)

tokenizer = CharacterTokenizer.load(
    "../checkpoints/tokenizer.json"
)

model = CocoTransformer(
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

    print(f"\nCoco: {text}")
