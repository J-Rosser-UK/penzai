import numpy as np
import os
from flax.training import checkpoints

from model import GPT, GPTConfig
from trainer import Trainer

import numpy as np
from bpe import get_encoder
from inference import load_model_params
from flax.training import checkpoints


def prepare_shakespeare_data_bpe(path, block_size):
    """
    Read the entire corpus from `path`, tokenize it with the GPT-2 BPE tokenizer,
    then produce (x, y) training examples of length `block_size`.

    Args:
        path (str): Path to your Shakespeare text file (e.g. "input.txt").
        block_size (int): The context length for each training sample (e.g. 128).

    Returns:
        dataset (dict): A dictionary containing:
            - "x": np.array of shape (num_samples, block_size), token IDs.
            - "y": np.array of shape (num_samples, block_size), token IDs shifted by 1.
            - "vocab_size": int, size of the GPT-2 BPE vocabulary (should be 50257).
            - "encoder": the BPE encoder object (for decode/encode if needed).
    """

    # 1) Read the entire text file
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # 2) Get the GPT-2 BPE encoder from bpe.py
    enc = get_encoder()
    # This should download any GPT-2 files if not cached under ~/.cache/mingpt/
    # and construct a tokenizer with vocab of size 50257.

    # 3) Encode the entire text into a list of token IDs
    token_ids = enc.encode(text)
    data = np.array(token_ids, dtype=np.int32)

    # 4) For training, we want input sequences of length block_size
    #    and targets that are the same sequence shifted by one.
    #    So x[i] = data[i : i+block_size]
    #       y[i] = data[i+1 : i+block_size+1]
    #    We skip the very last token to align shapes properly.
    num_samples = len(data) - block_size
    x = []
    y = []
    for i in range(num_samples):
        x.append(data[i : i + block_size])
        y.append(data[i + 1 : i + block_size + 1])
    x = np.array(x, dtype=np.int32)
    y = np.array(y, dtype=np.int32)

    # 5) Construct the dataset dict
    dataset = {
        "x": x,
        "y": y,
        "vocab_size": len(enc.encoder),  # should be 50257
        "encoder": enc,
    }
    return dataset


def main():
    # 1) Prepare data
    block_size = 128
    train_data = prepare_shakespeare_data_bpe("data/input.txt", block_size)

    # 2) Construct GPT model
    vocab_size = train_data["vocab_size"]
    config = GPTConfig(
        vocab_size=vocab_size, block_size=block_size, n_layer=4, n_head=4, n_embd=256
    )
    model = GPT(config)

    # 3) Setup trainer config
    trainer_cfg = Trainer.get_default_config()
    trainer_cfg.block_size = block_size
    trainer_cfg.max_iters = 50000
    trainer_cfg.batch_size = 64
    trainer_cfg.learning_rate = 3e-4

    ckpt_dir = os.path.abspath("checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 4) Create trainer
    trainer = Trainer(trainer_cfg, model, train_data, params=None, ckpt_dir=ckpt_dir)

    # 5) Run training loop
    trainer.run()
    print("Finished training, final loss:", float(trainer.loss))


if __name__ == "__main__":
    main()
