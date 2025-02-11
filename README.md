# Penzai Exploration

A quick repo exploring Penzai! I reimplemented GPT-2 in flax linen with sharding and added in p-RoPE for fun! I trained it on a small Shakespeare dataset both locally on my 3080 and on colab with 8 TPUs (that ran 30x faster!).
I then unflaxified it and had a look at the attention patterns (through dropout as I didn't get time to turn that into a removable layer!).

Normally my code is much cleaner and with unittests!


# Summary

On the whole, I felt like Penzai was much "closer" to the actual model than TransformerLens, operating directly on the Jax PyTree and supporting jitting and sharding. One drawback is that when you `unflaxify` a model, you lose all the methods that run on it e.g. `model.generate` so the `unflaxified` model is really cemented as a duplicate for inspecting. Equally, treescope doesn't work as well on `unflaxified` models compared to ones natively built in Penzai, wrapping every layer in `InterceptedFlaxModuleMethod`.





| Feature | Penzai | TransformerLens|
|--------------------------|--------------------------|--------------------------|
| Supported frameworks? | `flax.linen` and `penzai` | `torch` inside `transformer-lens` |
|Visualizations | Treescope - really nice interactivity especially with named axes. Missed the input-sequence visualizations but I can see the functionality is built into treescope, would need custom integration. | CircuitsVis - really love the token-level visualizations where hovering over a word in the input sequence highlights the attention patterns on the other ones.|
|Loading Models | Either build the model in Penzai or "unflaxify" it | Only comes with small pre-loaded `HookedTransformer` models, difficult to define a new one|
|Parallelism | Shard & jit model like normal | No support |
|Caching activations | No caching, only saved if hooked. More practical for larger models. | `model.run_with_cache` to save activations, less practical for larger models |
|Hooks | Use `Selector` to assign hooks directly on the JAX PyTree then run forward pass as normal | `model.run_with_hooks` has to be explictly called|
|Generate functionality | Lost when `unflaxify` is called. | `model.generate` for basic text generation|
|Bonus Features | `NamedArray` and the `copy and paste` functionality of treescope` | Easy cache all activations|

# Colab for running training sharded across TPUs
https://colab.research.google.com/drive/1oZ5_KPPOqHP4wLQLYuvI_nO36TCV6Hfv#scrollTo=llETZKq21sd2