# Palmera Configuration

# Dataset

BLOCK_SIZE = 64

# Training

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 3e-4

# Transformer

N_EMBD = 128
N_HEAD = 4
N_LAYER = 2

# Generation

MAX_NEW_TOKENS = 100

# Paths

TRAIN_FILE = "../data/train.txt"

VALIDATION_FILE = "../data/validation.txt"

CHECKPOINT_PATH = "../checkpoints/palmera_v0.pt"

TOKENIZER_PATH = "../checkpoints/tokenizer.json"
