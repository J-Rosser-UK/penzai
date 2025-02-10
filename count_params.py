import jax
import jax.numpy as jnp
from jax import random
from model import GPT, GPTConfig

# Define a sample configuration (modify these as needed)
config = GPTConfig(
    vocab_size=50257,  # for example, GPT-2's vocab size
    block_size=256,  # maximum sequence length
    n_layer=4,  # number of Transformer blocks
    n_head=4,  # number of attention heads
    n_embd=256,  # embedding dimension
)

# Instantiate the model
model = GPT(config)

# Create a dummy input. The shape should match the model's expected input shape.
# Here we use a batch size of 1 and sequence length equal to block_size.
dummy_input = jnp.zeros((1, config.block_size), dtype=jnp.int32)

# Initialize the model parameters with a PRNG key
key = random.PRNGKey(0)
variables = model.init(key, dummy_input)
params = variables["params"]


# Function to count total parameters
def count_params(params):
    # Use jax.tree_util.tree_leaves to get all arrays in the nested dict
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


total_params = count_params(params)
print("Total number of parameters:", total_params)
