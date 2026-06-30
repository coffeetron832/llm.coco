from flask import Flask, request, jsonify
from flask import send_from_directory

import torch
import os
import sys

sys.path.append("../model")

from tokenizer import CharacterTokenizer
from transformer import CocoTransformer


app = Flask(
    __name__,
    static_folder="../web",
    static_url_path=""
)

CHECKPOINT_PATH = "../checkpoints/palmer_v0.pt"

print("Cargando Palmer...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
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

print("Palmer listo.")


@app.route("/")
def home():
    return send_from_directory(
        app.static_folder,
        "index.html"
    )


@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    prompt = data.get(
        "message",
        ""
    ).strip()

    if not prompt:
        return jsonify({
            "response": "Escribe algo."
        })

    context = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long
    )

    with torch.no_grad():

        output = model.generate(
            context,
            max_new_tokens=80
        )

    text = tokenizer.decode(
        output[0].tolist()
    )

    generated = text[len(prompt):].strip()

    if not generated:
        generated = "No sé qué responder todavía."

    return jsonify({
        "response": generated
    })


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
