"""
Full definition of a GPT Language Model rewritten in Flax Linen (JAX).

References:
1) The official GPT-2 TensorFlow implementation released by OpenAI:
    https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
3) Original PyTorch version in minGPT
    https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
"""

import math
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple, Any
from flax.linen import initializers

import math
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple, Any
from flax.linen import initializers
from tqdm import tqdm


import jax
import jax.numpy as jnp
from gpt2.layer_norm import LayerNorm
from functools import partial


def apply_p_rope(
    inputs: jnp.ndarray,  # [..., seq_len, head_dim]
    positions: jnp.ndarray,  # [..., seq_len]
    head_dim: int,
    max_wavelength: int = 10000,
    rope_percentage: float = 1.0,
) -> jnp.ndarray:
    """
    Applies partial Rotary Positional Embeddings (p-RoPE) to `inputs`.

    inputs:    shape = [..., seq_len, head_dim]
    positions: shape = [..., seq_len]

    For the non-rotated dimensions, an infinite wavelength is used
    effectively leaving those dimensions unrotated, preserving their semantic information.
    """
    # How many dimensions get the rotary transform:
    rope_angles = int(rope_percentage * head_dim // 2)
    # Remaining "no-rotation" dims:
    nope_angles = head_dim // 2 - rope_angles

    fraction = 2.0 * jnp.arange(0, rope_angles) / head_dim
    timescale = max_wavelength**fraction

    # Pad so leftover dims see "infinite" wavelength => effectively no rotation
    timescale = jnp.pad(
        timescale,
        (0, nope_angles),
        mode="constant",
        constant_values=(0, jnp.inf),
    )

    # positions[..., None] => shape [..., seq_len, 1]
    # timescale[None, :]   => shape [1, rotary_dims]
    # => results in shape [..., seq_len, rotary_dims]
    sinusoid_inp = positions[..., None] / timescale[None, :]

    sin = jnp.sin(sinusoid_inp)  # [..., seq_len, d/2]
    cos = jnp.cos(sinusoid_inp)  # [..., seq_len, d/2]

    # Split the input along the last dimension
    first_half, second_half = jnp.split(inputs, 2, axis=-1)
    # Each half has shape [..., seq_len, head_dim/2]

    # Standard RoPE transformation
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin

    out = jnp.concatenate([first_part, second_part], axis=-1)
    return out.astype(inputs.dtype)


class GPTConfig:
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        embd_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        model_type: Optional[str] = None,
        rope_percentage: float = 1.0,
        max_wavelength: int = 10000,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.attn_pdrop = attn_pdrop
        self.model_type = model_type

        self.rope_percentage = rope_percentage
        self.max_wavelength = max_wavelength


def new_gelu(x: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * x * (1.0 + jnp.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


class CausalSelfAttention(nn.Module):
    config: GPTConfig

    def setup(self):
        assert self.config.n_embd % self.config.n_head == 0
        self.c_attn = nn.Dense(
            3 * self.config.n_embd,
            use_bias=True,
            kernel_init=initializers.normal(stddev=0.02),
        )
        self.c_proj = nn.Dense(
            self.config.n_embd,
            use_bias=True,
            kernel_init=initializers.normal(stddev=0.02),
        )
        self.attn_dropout = nn.Dropout(self.config.attn_pdrop)
        self.resid_dropout = nn.Dropout(self.config.resid_pdrop)
        self.n_head = self.config.n_head
        self.n_embd = self.config.n_embd

        self.rope = partial(
            apply_p_rope,
            max_wavelength=self.config.max_wavelength,
            rope_percentage=self.config.rope_percentage,
        )

    def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        B, T, C = x.shape

        # Calculate head dimension
        head_dim = C // self.n_head

        # QKV projection
        qkv = self.c_attn(x)  # [B, T, 3*C]
        q, k, v = jnp.split(qkv, 3, axis=-1)  # each [B, T, C]

        # Reshape and transpose q, k, v for attention
        def split_heads(tensor):
            return tensor.reshape(B, T, self.n_head, head_dim).transpose(0, 2, 1, 3)

        q = split_heads(q)  # [B, nh, T, hd]
        k = split_heads(k)  # [B, nh, T, hd]
        v = split_heads(v)  # [B, nh, T, hd]

        # Apply RoPE
        positions = jnp.arange(T, dtype=q.dtype)

        # Reshape for RoPE
        q_rope = q.reshape(-1, T, head_dim)  # [B*nh, T, hd]
        k_rope = k.reshape(-1, T, head_dim)  # [B*nh, T, hd]

        # Apply RoPE and reshape back
        q = self.rope(q_rope, positions, head_dim).reshape(B, self.n_head, T, head_dim)
        k = self.rope(k_rope, positions, head_dim).reshape(B, self.n_head, T, head_dim)

        # Scaled dot-product attention
        scale = jnp.sqrt(head_dim).astype(k.dtype)
        attention = (q @ k.transpose(0, 1, 3, 2)) / scale  # [B, nh, T, T]

        # Causal mask
        causal_mask = jnp.tril(jnp.ones((T, T))).astype(bool)
        mask_value = jnp.finfo(attention.dtype).min
        attention = jnp.where(causal_mask, attention, mask_value)

        # Softmax and dropout
        attention = nn.softmax(attention, axis=-1)
        attention = self.attn_dropout(attention, deterministic=deterministic)

        # Attention output
        y = attention @ v  # [B, nh, T, hd]

        # Reshape back to [B, T, C]
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)

        # Output projection
        y = self.c_proj(y)
        y = self.resid_dropout(y, deterministic=deterministic)

        return y


class MLP(nn.Module):
    config: GPTConfig

    def setup(self):
        self.c_fc = nn.Dense(
            4 * self.config.n_embd, kernel_init=initializers.normal(stddev=0.02)
        )
        self.c_proj = nn.Dense(
            self.config.n_embd, kernel_init=initializers.normal(stddev=0.02)
        )
        self.dropout = nn.Dropout(self.config.resid_pdrop)

    def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x, deterministic=deterministic)
        return x


class Block(nn.Module):
    config: GPTConfig

    def setup(self):
        self.ln_1 = LayerNorm(features=self.config.n_embd, epsilon=1e-5)
        self.attn = CausalSelfAttention(self.config)
        self.ln_2 = LayerNorm(features=self.config.n_embd, epsilon=1e-5)
        self.mlp = MLP(self.config)

    def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:

        ln_1_out, _ = self.ln_1(x)
        ln_2_out, _ = self.ln_2(x)
        x = x + self.attn(ln_1_out, deterministic=deterministic)
        x = x + self.mlp(ln_2_out, deterministic=deterministic)
        return x


class GPT(nn.Module):
    """A 28M Parameter GPT-2 model."""

    config: GPTConfig

    def setup(self):
        self.wte = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.n_embd,
            embedding_init=initializers.normal(stddev=0.02),
        )
        self.drop = nn.Dropout(self.config.embd_pdrop)
        self.h = [Block(self.config) for _ in range(self.config.n_layer)]
        self.ln_f = LayerNorm(features=self.config.n_embd, epsilon=1e-5)
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            kernel_init=initializers.normal(stddev=0.02),
        )

    def __call__(
        self,
        idx: jnp.ndarray,
        targets: Optional[jnp.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        B, T = idx.shape
        # validate
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}, "
            f"block size is only {self.config.block_size}"
        )

        # tokens -> embeddings
        tok_emb = self.wte(idx)  # [B, T, n_embd]

        # NO additive pos_emb if using RoPE. So just:
        x = self.drop(tok_emb, deterministic=deterministic)

        # forward the Transformer blocks
        for block in self.h:
            x = block(x, deterministic=deterministic)

        x, _ = self.ln_f(x)
        logits = self.lm_head(x)

        # optional loss
        loss = None
        if targets is not None:
            vocab_size = logits.shape[-1]
            logits_2d = logits.reshape(-1, vocab_size)
            targets_1d = targets.reshape(-1)
            one_hot = jax.nn.one_hot(targets_1d, vocab_size)
            mask = targets_1d != -1
            losses = -jnp.sum(one_hot * nn.log_softmax(logits_2d, axis=-1), axis=-1)
            loss = jnp.sum(losses * mask) / jnp.sum(mask)

        return logits, loss

    def generate(
        self,
        idx: jnp.ndarray,
        max_new_tokens: int,
        rng: Any,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: Optional[int] = None,
    ) -> jnp.ndarray:
        """
        Take a conditioning sequence of indices (idx) and complete
        the sequence max_new_tokens times.
        """
        for _ in tqdm(range(max_new_tokens), desc="Generating tokens"):
            # Crop context if it exceeds block_size
            if idx.shape[1] > self.config.block_size:
                idx_cond = idx[:, -self.config.block_size :]
            else:
                idx_cond = idx

            # Forward the model (deterministic = True for generation)
            logits, _ = self(idx_cond, deterministic=True)

            # Pluck logits at the final step
            logits = logits[:, -1, :] / temperature

            # Optionally top-k
            if top_k is not None:
                # Fixed top-k implementation
                top_logits, _ = jax.lax.top_k(logits, top_k)
                k_threshold = jnp.min(top_logits, axis=-1, keepdims=True)
                logits = jnp.where(logits < k_threshold, -jnp.inf, logits)

            # Convert to probabilities
            probs = nn.softmax(logits, axis=-1)

            # Sample or take argmax
            rng, subkey = jax.random.split(rng)
            if do_sample:
                next_token = jax.random.categorical(subkey, jnp.log(probs), axis=-1)
            else:
                next_token = jnp.argmax(probs, axis=-1)

            # Append to sequence
            next_token = next_token[:, None]  # shape (B, 1)
            idx = jnp.concatenate([idx, next_token], axis=1)

        return idx

    def generate_yield(
        self,
        idx: jnp.ndarray,
        max_new_tokens: int,
        rng: Any,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: Optional[int] = None,
    ):
        """
        Take a conditioning sequence of indices (idx) and complete
        the sequence max_new_tokens times.
        """
        for _ in range(max_new_tokens):
            # Crop context if it exceeds block_size
            if idx.shape[1] > self.config.block_size:
                idx_cond = idx[:, -self.config.block_size :]
            else:
                idx_cond = idx

            # Forward the model (deterministic = True for generation)
            logits, _ = self(idx_cond, deterministic=True)

            # Pluck logits at the final step
            logits = logits[:, -1, :] / temperature

            # Optionally top-k
            if top_k is not None:
                # Fixed top-k implementation
                top_logits, _ = jax.lax.top_k(logits, top_k)
                k_threshold = jnp.min(top_logits, axis=-1, keepdims=True)
                logits = jnp.where(logits < k_threshold, -jnp.inf, logits)

            # Convert to probabilities
            probs = nn.softmax(logits, axis=-1)

            # Sample or take argmax
            rng, subkey = jax.random.split(rng)
            if do_sample:
                next_token = jax.random.categorical(subkey, jnp.log(probs), axis=-1)
            else:
                next_token = jnp.argmax(probs, axis=-1)

            # Append to sequence
            next_token = next_token[:, None]  # shape (B, 1)
            idx = jnp.concatenate([idx, next_token], axis=1)

            yield next_token[0]
