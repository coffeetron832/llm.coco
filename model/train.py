import torch
from torch.utils.data import DataLoader

from tokenizer import CharacterTokenizer
from dataset import TextDataset
from transformer import CocoTransformer


BATCH_SIZE = 32
EPOCHS = 10
BLOCK_SIZE = 64
LR = 3e-4


with open("../data/train.txt", "r", encoding="utf-8") as f:
    text = f.read()


tokenizer = CharacterTokenizer(text)

dataset = TextDataset(
    text,
    tokenizer,
    BLOCK_SIZE
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

model = CocoTransformer(
    vocab_size=tokenizer.vocab_size,
    block_size=BLOCK_SIZE
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR
)

for epoch in range(EPOCHS):

    for x, y in loader:

        logits, loss = model(x, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | Loss: {loss.item():.4f}"
    )


torch.save(
    {
        "model_state": model.state_dict()
    },
    "../checkpoints/palmera_v0.pt"
)

tokenizer.save(
    "../checkpoints/tokenizer.json"
)

print("Modelo guardado.")
print("Tokenizer guardado.")
