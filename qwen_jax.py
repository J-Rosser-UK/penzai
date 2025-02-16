from typing import Any, Optional, Tuple
from functools import partial
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import lax
import optax
from rich import print


class DynamicCache:
    """
    A cache that grows dynamically as more tokens are generated.
    """

    def __init__(self, num_hidden_layers: Optional[int] = None):
        self._seen_tokens = 0
        self.key_cache = []
        self.value_cache = []

    def update(
        self,
        key_states: jnp.ndarray,
        value_states: jnp.ndarray,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if key_states is not None:
            while len(self.key_cache) <= layer_idx:
                self.key_cache.append([])
                self.value_cache.append([])

            if len(self.key_cache[layer_idx]) == 0:
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else:
                self.key_cache[layer_idx] = jnp.concatenate(
                    [self.key_cache[layer_idx], key_states], axis=-2
                )
                self.value_cache[layer_idx] = jnp.concatenate(
                    [self.value_cache[layer_idx], value_states], axis=-2
                )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        is_empty_layer = (
            len(self.key_cache) == 0
            or len(self.key_cache) <= layer_idx
            or len(self.key_cache[layer_idx]) == 0
        )
        return 0 if is_empty_layer else self.key_cache[layer_idx].shape[-2]


class Qwen2Config:
    def __init__(
        self,
        vocab_size=151936,
        hidden_size=1536,
        intermediate_size=8960,
        num_hidden_layers=28,
        num_attention_heads=12,
        num_key_value_heads=2,
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout


class Qwen2RMSNorm(nn.Module):
    hidden_size: int
    eps: float = 1e-6
    param_dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.weight = self.param(
            "weight",
            nn.initializers.ones,
            (self.hidden_size,),
            self.param_dtype,
        )

    def __call__(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(jnp.float32)
        variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * lax.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).astype(input_dtype)


class Qwen2MLP(nn.Module):
    config: Qwen2Config
    param_dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.gate_proj = nn.Dense(
            features=self.config.intermediate_size,
            use_bias=False,
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
        )
        self.up_proj = nn.Dense(
            features=self.config.intermediate_size,
            use_bias=False,
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
        )
        self.down_proj = nn.Dense(
            features=self.config.hidden_size,
            use_bias=False,
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
        )

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2Attention(nn.Module):
    config: Qwen2Config
    layer_idx: int
    param_dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.num_key_value_groups = (
            self.config.num_attention_heads // self.config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Dense(
            features=self.config.num_attention_heads * self.head_dim,
            use_bias=True,
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
        )
        self.k_proj = nn.Dense(
            features=self.config.num_key_value_heads * self.head_dim,
            use_bias=True,
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
        )
        self.v_proj = nn.Dense(
            features=self.config.num_key_value_heads * self.head_dim,
            use_bias=True,
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
        )
        self.o_proj = nn.Dense(
            features=self.config.hidden_size,
            use_bias=False,
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
        )

    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([-x2, x1], axis=-1)

    def apply_rotary_pos_emb(self, q, k, cos, sin, unsqueeze_dim=1):
        cos = jnp.expand_dims(cos, axis=unsqueeze_dim)
        sin = jnp.expand_dims(sin, axis=unsqueeze_dim)
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    def repeat_kv(self, hidden_states: jnp.ndarray, n_rep: int) -> jnp.ndarray:
        if n_rep == 1:
            return hidden_states
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        hidden_states = jnp.expand_dims(hidden_states, axis=2)
        hidden_states = jnp.broadcast_to(
            hidden_states, (batch, num_key_value_heads, n_rep, slen, head_dim)
        )
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        position_embeddings: Tuple[jnp.ndarray, jnp.ndarray],
        attention_mask: Optional[jnp.ndarray] = None,
        past_key_value: Optional[DynamicCache] = None,
        deterministic: bool = True,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        batch_size, seq_length = hidden_states.shape[:2]

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape(
            batch_size, seq_length, self.config.num_attention_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        key_states = key_states.reshape(
            batch_size, seq_length, self.config.num_key_value_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        value_states = value_states.reshape(
            batch_size, seq_length, self.config.num_key_value_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        cos, sin = position_embeddings
        query_states, key_states = self.apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = self.repeat_kv(key_states, self.num_key_value_groups)
        value_states = self.repeat_kv(value_states, self.num_key_value_groups)

        # Scaled dot-product attention
        attn_weights = (
            jnp.matmul(query_states, key_states.transpose(0, 1, 3, 2)) * self.scaling
        )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.softmax(attn_weights, axis=-1)

        if not deterministic and self.config.attention_dropout > 0:
            attn_weights = nn.dropout(
                attn_weights,
                rate=self.config.attention_dropout,
                deterministic=deterministic,
            )

        attn_output = jnp.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_length, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights if output_attentions else None


class Qwen2DecoderLayer(nn.Module):
    config: Qwen2Config
    layer_idx: int
    param_dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.self_attn = Qwen2Attention(
            config=self.config,
            layer_idx=self.layer_idx,
            param_dtype=self.param_dtype,
        )
        self.mlp = Qwen2MLP(
            config=self.config,
            param_dtype=self.param_dtype,
        )
        self.input_layernorm = Qwen2RMSNorm(
            hidden_size=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            param_dtype=self.param_dtype,
        )
        self.post_attention_layernorm = Qwen2RMSNorm(
            hidden_size=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            param_dtype=self.param_dtype,
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_value: Optional[DynamicCache] = None,
        deterministic: bool = True,
        output_attentions: bool = False,
        position_embeddings: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        **kwargs,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            deterministic=deterministic,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class Qwen2Model(nn.Module):
    config: Qwen2Config
    param_dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embed_tokens = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
        )

        self.layers = [
            Qwen2DecoderLayer(
                config=self.config,
                layer_idx=i,
                param_dtype=self.param_dtype,
            )
            for i in range(self.config.num_hidden_layers)
        ]

        self.norm = Qwen2RMSNorm(
            hidden_size=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
            param_dtype=self.param_dtype,
        )

    def _compute_rope_embeddings(self, hidden_states, position_ids, inv_freq=None):
        # Compute RoPE embeddings
        if inv_freq is None:
            dim = self.config.hidden_size // self.config.num_attention_heads
            inv_freq = 1.0 / (
                self.config.rope_theta
                ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
            )

        inv_freq = jnp.expand_dims(jnp.expand_dims(inv_freq, 0), 0)
        position_ids = jnp.expand_dims(position_ids.astype(jnp.float32), -1)
        freqs = jnp.matmul(position_ids, inv_freq)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        cos = jnp.cos(emb)
        sin = jnp.sin(emb)

        return cos, sin

    def _prepare_4d_causal_attention_mask(
        self,
        attention_mask: jnp.ndarray,
        sequence_length: int,
        target_length: int,
        dtype: jnp.dtype,
        batch_size: int,
        cache_position: jnp.ndarray,
    ):
        min_dtype = jnp.finfo(dtype).min
        causal_mask = jnp.full(
            (sequence_length, target_length),
            min_dtype,
            dtype=dtype,
        )

        diagonal_attend_mask = jnp.arange(target_length) > cache_position.reshape(-1, 1)
        causal_mask = jnp.where(diagonal_attend_mask, causal_mask, 0)
        causal_mask = jnp.broadcast_to(
            causal_mask[None, None, :, :],
            (batch_size, 1, sequence_length, target_length),
        )

        return causal_mask

    def __call__(
        self,
        input_ids: jnp.ndarray = None,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_values: Optional[DynamicCache] = None,
        inputs_embeds: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if past_key_values is None:
            past_key_values = DynamicCache()

        past_seen_tokens = past_key_values.get_seq_length()
        cache_position = jnp.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
        )

        if position_ids is None:
            position_ids = jnp.expand_dims(cache_position, 0)

        # Create causal mask
        causal_mask = self._prepare_4d_causal_attention_mask(
            attention_mask,
            inputs_embeds.shape[1],
            inputs_embeds.shape[1] + past_seen_tokens,
            inputs_embeds.dtype,
            inputs_embeds.shape[0],
            cache_position,
        )

        hidden_states = inputs_embeds

        # Create position embeddings to be shared across layers
        position_embeddings = self._compute_rope_embeddings(hidden_states, position_ids)

        # Process through layers
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                deterministic=deterministic,
                output_attentions=output_attentions,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        return hidden_states, past_key_values


class Qwen2ForCausalLM(nn.Module):
    config: Qwen2Config
    param_dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.model = Qwen2Model(
            config=self.config,
            param_dtype=self.param_dtype,
        )
        self.lm_head = nn.Dense(
            features=self.config.vocab_size,
            use_bias=False,
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
            # kernel_init=initializers.normal(stddev=0.02),  # Added initialization
        )

    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        past_key_values: Optional[DynamicCache] = None,
        inputs_embeds: Optional[jnp.ndarray] = None,
        labels: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        # Loss computation - changed to match working implementation
        loss = None
        if labels is not None:
            vocab_size = logits.shape[-1]
            # Shift logits and labels
            shift_logits = logits[..., :-1, :].reshape(-1, vocab_size)
            shift_labels = labels[..., 1:].reshape(-1)

            one_hot = jax.nn.one_hot(shift_labels, vocab_size)
            mask = shift_labels != -1  # Assuming -1 is the padding token
            losses = -jnp.sum(one_hot * nn.log_softmax(shift_logits, axis=-1), axis=-1)
            loss = jnp.sum(losses * mask) / jnp.sum(mask)

        return loss, logits, past_key_values, hidden_states

    def generate(
        self,
        input_ids: jnp.ndarray,
        max_new_tokens: int,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: Optional[int] = None,
        prng_key: Optional[jax.random.PRNGKey] = None,
        **kwargs,
    ) -> jnp.ndarray:
        """Generate tokens autoregressively."""
        print("input_ids", input_ids)
        print("GENERATE")

        generated = input_ids
        past = None

        for _ in range(max_new_tokens):
            # Get input for current step
            input_ids_cond = generated if past is None else generated[:, -1:]

            # Model forward pass
            _, logits, past, _ = self(
                input_ids=input_ids_cond,
                past_key_values=past,
                deterministic=True,
                **kwargs,
            )

            # Get logits for last token and scale by temperature
            next_token_logits = logits[:, -1, :] / temperature

            # Optionally apply top-k
            if top_k is not None:
                top_logits, _ = jax.lax.top_k(next_token_logits, top_k)
                k_threshold = jnp.min(top_logits, axis=-1, keepdims=True)
                next_token_logits = jnp.where(
                    next_token_logits < k_threshold, -jnp.inf, next_token_logits
                )

            # Convert to probabilities
            probs = jax.nn.softmax(next_token_logits, axis=-1)

            # Sample or take argmax
            if do_sample and prng_key is not None:
                prng_key, subkey = jax.random.split(prng_key)
                next_token = jax.random.categorical(subkey, next_token_logits, axis=-1)
            else:
                next_token = jnp.argmax(probs, axis=-1)

            # Append generated token
            next_token = next_token[:, None]  # Add sequence dimension
            generated = jnp.concatenate([generated, next_token], axis=1)

        return generated


import torch
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import freeze, unfreeze
import numpy as np


def convert_pytorch_to_jax_cpu(pytorch_state_dict, flax_params):
    """
    Convert PyTorch state dict to JAX/Flax parameters using CPU memory.

    Args:
        pytorch_state_dict: PyTorch state dict containing model weights
        flax_params: Initial Flax parameter structure

    Returns:
        Converted Flax parameters
    """
    jax_params = unfreeze(flax_params)

    # Helper function to convert torch tensor to numpy array
    def torch_to_numpy(tensor):
        return tensor.cpu().numpy()

    # Same parameter mappings as before, but without "params" prefix since it's already in flax_params
    name_mapping = {
        "model.embed_tokens.weight": ("model", "embed_tokens", "embedding"),
        "model.norm.weight": ("model", "norm", "weight"),
        "lm_head.weight": ("lm_head", "kernel"),
    }

    layer_mapping = {
        "model.layers.{}.input_layernorm.weight": (
            "model",
            "layers_{}",
            "input_layernorm",
            "weight",
        ),
        "model.layers.{}.post_attention_layernorm.weight": (
            "model",
            "layers_{}",
            "post_attention_layernorm",
            "weight",
        ),
        "model.layers.{}.self_attn.q_proj.weight": (
            "model",
            "layers_{}",
            "self_attn",
            "q_proj",
            "kernel",
        ),
        "model.layers.{}.self_attn.q_proj.bias": (
            "model",
            "layers_{}",
            "self_attn",
            "q_proj",
            "bias",
        ),
        "model.layers.{}.self_attn.k_proj.weight": (
            "model",
            "layers_{}",
            "self_attn",
            "k_proj",
            "kernel",
        ),
        "model.layers.{}.self_attn.k_proj.bias": (
            "model",
            "layers_{}",
            "self_attn",
            "k_proj",
            "bias",
        ),
        "model.layers.{}.self_attn.v_proj.weight": (
            "model",
            "layers_{}",
            "self_attn",
            "v_proj",
            "kernel",
        ),
        "model.layers.{}.self_attn.v_proj.bias": (
            "model",
            "layers_{}",
            "self_attn",
            "v_proj",
            "bias",
        ),
        "model.layers.{}.self_attn.o_proj.weight": (
            "model",
            "layers_{}",
            "self_attn",
            "o_proj",
            "kernel",
        ),
        "model.layers.{}.mlp.gate_proj.weight": (
            "model",
            "layers_{}",
            "mlp",
            "gate_proj",
            "kernel",
        ),
        "model.layers.{}.mlp.up_proj.weight": (
            "model",
            "layers_{}",
            "mlp",
            "up_proj",
            "kernel",
        ),
        "model.layers.{}.mlp.down_proj.weight": (
            "model",
            "layers_{}",
            "mlp",
            "down_proj",
            "kernel",
        ),
    }

    # Process tensors in batches to save memory
    def process_tensor(tensor, needs_transpose=False):
        # Convert to NumPy first
        np_array = torch_to_numpy(tensor)
        if needs_transpose:
            np_array = np_array.T
        # Only convert to JAX array at the end
        return jnp.array(np_array)

    # Convert non-layer parameters
    for torch_name, jax_path in name_mapping.items():
        if torch_name in pytorch_state_dict:
            tensor = pytorch_state_dict[torch_name]
            needs_transpose = torch_name == "lm_head.weight"

            # Process tensor on CPU
            jax_tensor = process_tensor(tensor, needs_transpose)

            # Navigate and assign
            current_dict = jax_params
            for key in jax_path[:-1]:
                current_dict = current_dict[key]
            current_dict[jax_path[-1]] = jax_tensor

            # Clear PyTorch cache
            del tensor
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Convert layer parameters
    num_layers = 28
    for layer_idx in range(num_layers):
        for torch_template, jax_path in layer_mapping.items():
            torch_name = torch_template.format(layer_idx)
            if torch_name in pytorch_state_dict:
                tensor = pytorch_state_dict[torch_name]
                needs_transpose = "proj.weight" in torch_name

                # Process tensor on CPU
                jax_tensor = process_tensor(tensor, needs_transpose)

                # Format and navigate path
                formatted_path = [
                    p.format(layer_idx) if "{}" in p else p for p in jax_path
                ]
                current_dict = jax_params
                for key in formatted_path[:-1]:
                    current_dict = current_dict[key]
                current_dict[formatted_path[-1]] = jax_tensor

                # Clear memory
                del tensor
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return freeze(jax_params)


# Usage example:
def convert_model_cpu(flax_model, torch_model):
    # Initialize with minimal input on CPU
    key = jax.random.PRNGKey(0)
    init_inputs = jnp.ones((1, 1), dtype=jnp.int32)

    with jax.default_device(jax.devices("cpu")[0]):
        variables = flax_model.init(key, input_ids=init_inputs)

    # Load PyTorch weights
    pytorch_state_dict = torch_model.state_dict()

    # Convert weights on CPU
    converted_params = convert_pytorch_to_jax_cpu(
        pytorch_state_dict, variables["params"]
    )

    return converted_params


# Tokenize and prepare input
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
torch_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)


# First, modify the end of your script to properly bind the parameters:
with jax.default_device(jax.devices("cpu")[0]):
    flax_model = Qwen2ForCausalLM(Qwen2Config())
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 1), dtype=jnp.int32)

    # Get initial parameters
    variables = flax_model.init(rng, dummy_input)
    initial_params = variables["params"]

    # Convert the PyTorch parameters
    converted_params = convert_model_cpu(flax_model, torch_model)

prompt = "What is 3 + 4? <think>\n"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = jnp.array(inputs["input_ids"].numpy())

print("prompt:", prompt)
print(input_ids)


# Create generate function that uses the params
@jax.jit
def generate_fn(params, input_ids):
    return flax_model.apply(
        {"params": params},
        input_ids,
        max_new_tokens=5,
        method=flax_model.generate,
    )


# Generate
output_ids = generate_fn(converted_params, input_ids)

print("output_ids", output_ids)

# Decode
generated_text = tokenizer.batch_decode(
    output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print("end")

print(generated_text)
