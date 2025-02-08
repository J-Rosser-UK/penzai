# utils.py

import os
import sys
import json
import random
from ast import literal_eval
from typing import Union

import numpy as np
import jax


class CfgNode:
    """
    A lightweight configuration class, loosely inspired by PyTorch's yacs-based config.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append(f"{k}:\n")
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append(f"{k}: {v}\n")
        parts = [" " * (indent * 4) + p for p in parts]
        return "".join(parts)

    def to_dict(self):
        """
        Return a dict representation of the config
        """
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                d[k] = v.to_dict()
            else:
                d[k] = v
        return d

    def merge_from_dict(self, d):
        self.__dict__.update(d)

    def merge_from_args(self, args):
        """
        Update the configuration from a list of strings that is expected
        to come from the command line, e.g. sys.argv[1:].

        The arguments are expected to be in the form of `--arg=value`, and
        the arg can use . to denote nested sub-attributes. Example:
          --model.n_layer=10 --trainer.batch_size=32
        """
        for arg in args:
            keyval = arg.split("=")
            assert len(keyval) == 2, (
                "Expecting each override arg to be of form --arg=value, got %s" % arg
            )
            key, val = keyval

            # first translate val into a python object
            try:
                val = literal_eval(val)
            except ValueError:
                pass

            assert key.startswith("--")
            key = key[2:]  # strip the '--'
            keys = key.split(".")
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            leaf_key = keys[-1]

            # ensure that this attribute exists
            assert hasattr(obj, leaf_key), f"{key} is not an attribute in the config"

            print(f"Overriding config attribute {key} to {val}")
            setattr(obj, leaf_key, val)


def set_seed(seed: int):
    """
    Set Python and NumPy seeds.
    For JAX, you generally manage PRNGKeys directly, so there's no global 'jax seed' in the same sense,
    but we can do the minimal for replicating some determinism in Python code.
    """
    random.seed(seed)
    np.random.seed(seed)
    # There's no exact equivalent of `torch.cuda.manual_seed_all(seed)` in JAX.
    # Instead you handle jax.random.PRNGKey(seed) in your code, as done in the Trainer init.


def setup_logging(config: CfgNode):
    """
    Minimal logging setup: saves sys.argv and config to a JSON in your work_dir.
    """
    work_dir = config.system.work_dir
    os.makedirs(work_dir, exist_ok=True)
    # log the args (if any)
    with open(os.path.join(work_dir, "args.txt"), "w") as f:
        f.write(" ".join(sys.argv))
    # log the config itself
    with open(os.path.join(work_dir, "config.json"), "w") as f:
        f.write(json.dumps(config.to_dict(), indent=4))
