"""
Simple training loop in JAX/Flax; boilerplate that can apply to any model.
This is a direct rewrite of the PyTorch-based trainer to a Flax-based version.
"""

import time
from collections import defaultdict
from typing import Any, Dict, Callable

from tqdm import tqdm

import jax
import jax.numpy as jnp
import optax

from flax.training import train_state
from flax.core import FrozenDict

from utils import CfgNode, set_seed


class TrainState(train_state.TrainState):
    """
    Custom TrainState to carry extra states like the RNG for dropout, etc.
    """

    dropout_rng: jax.random.PRNGKey


def create_train_state(
    rng: jax.random.PRNGKey, model, config: CfgNode, params=None
) -> TrainState:
    """
    Initialize parameters, define the optimizer (with gradient clipping, weight decay, etc.),
    and return a TrainState instance.
    """

    # Split RNGs for parameter init vs. dropout
    init_rng, dropout_init_rng = jax.random.split(rng)

    # Example shape: (batch_size, sequence_length) for tokens
    # but for initialization, we often pass a dummy shape (B, T).
    # Adjust as needed. Suppose block_size=128 for a dummy example:
    dummy_input = jnp.zeros((1, config.block_size), dtype=jnp.int32)

    # Initialize params; model.__call__ expects (idx, targets=None, deterministic=False)
    variables = model.init(
        {"params": init_rng, "dropout": dropout_init_rng},
        dummy_input,
        targets=None,
        deterministic=False,
    )
    params = variables["params"] if not params else params

    # Create an optimizer. For example:
    # - Clip by global norm
    # - Use AdamW
    scheduler = optax.constant_schedule(config.learning_rate)
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_norm_clip),
        optax.adamw(
            learning_rate=scheduler,
            b1=config.betas[0],
            b2=config.betas[1],
            weight_decay=config.weight_decay,
        ),
    )

    # Construct TrainState
    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer, dropout_rng=dropout_init_rng
    )
    return state


def train_step(state: TrainState, batch: Dict[str, jnp.ndarray]):
    """
    Single training step: forward pass, compute loss, backprop via grad, apply updates.
    Args:
      state: the current TrainState (includes params, optimizer, rng, etc.)
      batch: dict with "x" and "y" (token indices, targets), or however you store it.

    Returns:
      new_state: the updated TrainState
      loss: the scalar loss
    """

    def loss_fn(params):
        # We pass the dropout rng explicitly as {'dropout': state.dropout_rng}
        logits, loss = state.apply_fn(
            {"params": params},
            batch["x"],
            targets=batch["y"],
            deterministic=False,  # training mode
            rngs={"dropout": state.dropout_rng},
        )
        return loss

    # Compute gradients
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)

    # Update parameters
    new_state = state.apply_gradients(grads=grads)

    # Update dropout rng so next step is different
    # We can split from the current dropout RNG to get the next one
    new_dropout_rng, _ = jax.random.split(new_state.dropout_rng)
    new_state = new_state.replace(dropout_rng=new_dropout_rng)

    return new_state, loss


def infinite_random_sampler(
    dataset: Any, batch_size: int = 64, key: jax.random.PRNGKey = jax.random.PRNGKey(0)
):
    """
    Yields random samples from `dataset` with replacement, infinitely.

    dataset is assumed to be something indexable:
      e.g. a list of (x, y) or dicts
    """
    num_examples = len(dataset["x"])  # or however your dataset is stored
    while True:
        subkey, key = jax.random.split(key)
        # pick random indices
        idx = jax.random.randint(subkey, (batch_size,), 0, num_examples)
        # gather from dataset
        # Example: dataset["x"] and dataset["y"] are each np.array / jnp.array
        x_b = dataset["x"][idx, ...]
        y_b = dataset["y"][idx, ...]
        yield {"x": x_b, "y": y_b}


class Trainer:
    @staticmethod
    def get_default_config():
        C = CfgNode()
        # device: we won't specifically set "cuda" or "cpu" in JAX;
        # JAX will auto-detect. But let's keep the key for consistency.
        C.device = "auto"
        # data config
        C.num_workers = 4  # not strictly used here
        # optimizer parameters
        C.max_iters = 1000  # make sure to set a finite number here
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1
        C.grad_norm_clip = 1.0
        # block_size could be used for dummy init shape, etc.
        C.block_size = 128
        return C

    def __init__(self, config: CfgNode, model, train_dataset, params=None):
        self.config = config
        self.model = model

        # For reproducibility
        self.rng = jax.random.PRNGKey(42)  # or any seed
        set_seed(42)  # sets Python & NumPy seeds

        # Create the initial TrainState
        self.state = create_train_state(self.rng, model, config, params)

        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        # iteration variables for logging
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0
        self.loss = 0.0

        print("Trainer initialized with JAX. Using device:", jax.devices()[0])

    def add_callback(self, onevent: str, callback: Callable):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback: Callable):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        """
        Main training loop with a tqdm progress bar.
        """
        if self.config.max_iters is None:
            raise ValueError("config.max_iters must be set to a finite integer")

        # Prepare infinite sampler
        sampler_key, self.rng = jax.random.split(self.rng)
        data_iter = infinite_random_sampler(
            self.train_dataset, batch_size=self.config.batch_size, key=sampler_key
        )

        self.iter_time = time.time()

        # Initialize tqdm progress bar
        with tqdm(total=self.config.max_iters, desc="Training", unit="batch") as pbar:
            for _ in range(self.config.max_iters):
                # get next batch
                batch = next(data_iter)
                # move to device if needed
                batch = jax.device_put(batch)

                # single training step
                self.state, loss = train_step(self.state, batch)
                self.loss = loss

                # Trigger any callbacks
                self.trigger_callbacks("on_batch_end")

                # Logging and updating timing info
                self.iter_num += 1
                tnow = time.time()
                self.iter_dt = tnow - self.iter_time
                self.iter_time = tnow

                # Update tqdm progress bar
                pbar.set_postfix({"loss": float(loss)})
                pbar.update(1)
