# PyTorch Foundations for Weight Manipulation

We do not need to be PyTorch experts to build Chiquito. We only need to understand a handful of concepts that are central to how we load, move, and free model weights.

## Tensors and devices

A **tensor** is a typed multi-dimensional array. Every tensor lives on a specific **device**: CPU, a CUDA GPU, or a special device called `meta`.

```python
import torch

# A tensor on CPU (default)
a = torch.randn(3, 4)              # shape (3, 4), device=cpu

# A tensor on GPU
b = torch.randn(3, 4, device="cuda:0")

# Move between devices
c = a.to("cuda:0")   # copies data from CPU to GPU
d = b.to("cpu")      # copies data from GPU to CPU
```

The key operations we care about:
- **CPU to GPU**: this is the weight loading step that happens for every layer
- **GPU to meta**: this is how we free GPU memory after running a layer

## `nn.Module`: the building block

PyTorch models are built from `nn.Module` objects. A module can contain:

- **Parameters**: tensors that hold learned weights (e.g., the weight matrix of a linear layer)
- **Buffers**: tensors that are part of the module but are not learned (e.g., precomputed values like rotary embedding frequencies)
- **Sub-modules**: other `nn.Module` instances (this is how layers nest)

```python
import torch.nn as nn

linear = nn.Linear(768, 768)        # a single linear layer
print(linear.weight.shape)          # torch.Size([768, 768])
print(linear.weight.device)         # cpu
```

## `state_dict`: the flat view of all weights

Every module can export its weights as a flat dictionary mapping string names to tensors:

```python
model = SomeModel()
sd = model.state_dict()
# {
#   "layer1.weight": tensor(...),
#   "layer1.bias": tensor(...),
#   "layer2.weight": tensor(...),
#   ...
# }
```

The naming convention uses dots to separate the module hierarchy: `model.layers.5.self_attn.q_proj.weight` means "the *weight* parameter of the *q_proj* submodule of the *self_attn* submodule of layer *5*."

This flat naming is critical for Chiquito: we use name prefixes (like `model.layers.5.`) to identify which parameters belong to which layer.

## The meta device: shapes without memory

PyTorch has a special device called `meta`. A tensor on the meta device has a shape and a dtype but **occupies zero bytes of memory**:

```python
t = torch.empty(10000, 10000, device="meta")
print(t.shape)    # torch.Size([10000, 10000])
print(t.device)   # meta
# This uses 0 bytes — there is no data backing this tensor
```

The `accelerate` library provides a context manager that creates entire models on the meta device:

```python
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM, AutoConfig

config = AutoConfig.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)
```

This gives us a fully structured model object — with all layers, sub-modules, and correct shapes — but **zero memory usage**. The model exists as a skeleton.

This is exactly what Chiquito does at initialization: it creates the full model architecture on the meta device, then loads real weights into individual layers one at a time during the forward pass.

You can see this in [`model.py:196-208`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/model.py#L196-L208), where `_init_model()` creates the model on meta:

```python
def _init_model(self):
    self.model = None
    try:
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(
                self.config, attn_implementation="sdpa", trust_remote_code=True
            )
    except (TypeError, ValueError):
        self.model = None

    if self.model is None:
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(
                self.config, trust_remote_code=True
            )
```

## `set_module_tensor_to_device`: replacing meta with real data

The bridge between meta and real tensors is `set_module_tensor_to_device()` from `accelerate`. It takes a module, a parameter name, and a target device, and replaces the parameter in-place:

```python
from accelerate.utils.modeling import set_module_tensor_to_device

# Replace a meta parameter with a real tensor on GPU
set_module_tensor_to_device(
    model,                                     # the module
    "model.layers.0.self_attn.q_proj.weight",  # parameter name
    "cuda:0",                                  # target device
    value=real_tensor,                         # the actual data
    dtype=torch.float16,                       # desired dtype
)
```

This is the fundamental operation for layer-by-layer inference:
1. Load weights from a file into a CPU tensor
2. Call `set_module_tensor_to_device` to place them on GPU — the meta parameter is replaced with a real one
3. Run the layer
4. Call `set_module_tensor_to_device` with device `"meta"` to free the GPU memory

You can see steps 2 and 4 in [`model.py:318-328`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/model.py#L318-L328) and [`model.py:517-518`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/model.py#L517-L518):

```python
# Step 2: move weights to GPU
def _move_layer_to_device(self, state_dict):
    moved = list(state_dict.keys())
    for param_name in moved:
        set_module_tensor_to_device(
            self.model, param_name, self._device,
            value=state_dict[param_name], dtype=self._dtype,
        )
    return moved

# Step 4: free GPU memory (in forward())
for param_name in moved:
    set_module_tensor_to_device(self.model, param_name, "meta")
```

## `torch.inference_mode`

When running inference (not training), we do not need PyTorch to track gradients. `torch.inference_mode()` disables all gradient tracking, which saves memory and speeds up computation:

```python
with torch.inference_mode():
    output = model(input)
```

The entire forward pass in Chiquito runs inside this context manager ([`model.py:470`](https://github.com/elcapo/chiquito/blob/0.1.0/src/chiquito/model.py#L470)).

## Summary

The PyTorch tools we need for Chiquito are:

| Concept | What it does | Where we use it |
|---------|-------------|----------------|
| Tensors & devices | Store data on CPU/GPU/meta | Everywhere |
| `nn.Module` | Container for weights and sub-modules | The model itself |
| `state_dict()` | Flat dictionary of all weights | Loading/saving weights |
| Meta device | Shape without memory | Model initialization |
| `init_empty_weights()` | Create full model on meta | `_init_model()` |
| `set_module_tensor_to_device()` | Replace meta with real tensors | Layer loading/freeing |
| `torch.inference_mode()` | Disable gradient tracking | Forward pass |
