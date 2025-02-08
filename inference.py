# inference.py

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints
from model import GPT, GPTConfig
from utils import CfgNode
import os

# Import the JAX-friendly GPT-2 BPE tokenizer
from bpe import BPETokenizerJax


def load_model_params(ckpt_dir: str, config: GPTConfig):
    """
    Creates a GPT model and loads the parameters from the checkpoint.
    Returns: (model, params)
    """
    model = GPT(config)
    params = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=None)
    if params is None:
        raise ValueError(f"No checkpoint found in {ckpt_dir}")
    return model, params


def generate_text(
    prompt: str,
    tokenizer: BPETokenizerJax,
    model,
    params,
    max_new_tokens=5,
    temperature=1.0,
    top_k=None,
):
    """
    Generate continuation given a text prompt, using the BPE tokenizer.
    """
    # Tokenize the prompt via BPE
    # The tokenizer(...) call returns a list of jnp arrays (one per input string).
    encoded_list = tokenizer(prompt, return_tensors="jax")
    # For a single prompt string, encoded_list is a list of length 1.
    idx = encoded_list[0]  # shape (sequence_length,)

    # Add a batch dimension (shape: (1, seq_len))
    idx_jax = idx[jnp.newaxis, :]

    # Initialize RNG key
    rng = jax.random.PRNGKey(0)

    # Generate text
    out_idx = model.apply(
        {"params": params},
        idx_jax,
        max_new_tokens=max_new_tokens,
        rng=rng,
        temperature=temperature,
        do_sample=True,
        top_k=top_k,
        method=model.generate,
    )  # shape: (1, seq_len + max_new_tokens)

    # Remove batch dimension, decode to text
    out_idx_1d = out_idx[0]  # shape: (seq_len + max_new_tokens,)
    generated_text = tokenizer.decode(out_idx_1d)

    return generated_text


def main():
    # Configuration matching your training setup.
    # IMPORTANT: GPT-2 BPE has vocab_size=50257 by default.
    config = GPTConfig(
        vocab_size=50257,  # Updated for GPT-2 BPE
        block_size=128,  # or whichever block size you used
        n_layer=4,
        n_head=4,
        n_embd=256,
        embd_pdrop=0.0,  # 0 for inference
        resid_pdrop=0.0,
        attn_pdrop=0.0,
    )

    # Load your model from a checkpoint directory
    ckpt_dir = os.path.abspath("checkpoints")
    model, params = load_model_params(ckpt_dir, config)

    # Instantiate the tokenizer
    tokenizer = BPETokenizerJax()

    # Example prompt
    prompt = "Shakespeare wrote:"

    # Generate
    completion = generate_text(
        prompt=prompt,
        tokenizer=tokenizer,
        model=model,
        params=params,
        max_new_tokens=100,
        temperature=0.7,
        top_k=10,
    )[len(prompt) :]

    print(f"Prompt: {prompt}")
    print("Generated text:")
    print(completion)


if __name__ == "__main__":
    main()
