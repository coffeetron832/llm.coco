import torch
from config import *
from torch.utils.data import DataLoader

from tokenizer import CharacterTokenizer
from dataset import TextDataset
from transformer import PalmeraTransformer



with open(TRAIN_FILE, "r", encoding="utf-8") as f:
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

model = PalmeraTransformer(
    vocab_size=tokenizer.vocab_size,
    block_size=BLOCK_SIZE
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
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
    CHECKPOINT_PATH
)

tokenizer.save(
    TOKENIZER_PATH
)

print("Modelo guardado.")
print("Tokenizer guardado.")
