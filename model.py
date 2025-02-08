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

    def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        B, T, C = x.shape

        qkv = self.c_attn(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        head_dim = C // self.n_head
        q = q.reshape(B, T, self.n_head, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, head_dim).transpose(0, 2, 1, 3)

        att = jnp.einsum("bhqd,bhkd->bhqk", q, k)
        att = att * (1.0 / math.sqrt(k.shape[-1]))

        # Modified causal mask implementation
        causal_mask = jnp.tril(jnp.ones((T, T)))
        mask_value = jnp.finfo(att.dtype).min
        att = jnp.where(causal_mask, att, mask_value)

        att = nn.softmax(att, axis=-1)
        att = self.attn_dropout(att, deterministic=deterministic)

        y = jnp.einsum("bhqk,bhkd->bhqd", att, v)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)

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
        self.ln_1 = nn.LayerNorm(epsilon=1e-5)
        self.attn = CausalSelfAttention(self.config)
        self.ln_2 = nn.LayerNorm(epsilon=1e-5)
        self.mlp = MLP(self.config)

    def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        x = x + self.attn(self.ln_1(x), deterministic=deterministic)
        x = x + self.mlp(self.ln_2(x), deterministic=deterministic)
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
        self.wpe = nn.Embed(
            num_embeddings=self.config.block_size,
            features=self.config.n_embd,
            embedding_init=initializers.normal(stddev=0.02),
        )
        self.drop = nn.Dropout(self.config.embd_pdrop)
        self.h = [Block(self.config) for _ in range(self.config.n_layer)]
        self.ln_f = nn.LayerNorm(epsilon=1e-5)
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
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        pos = jnp.arange(T)[None, :]

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb, deterministic=deterministic)

        for block in self.h:
            x = block(x, deterministic=deterministic)
        x = self.ln_f(x)

        logits = self.lm_head(x)

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
