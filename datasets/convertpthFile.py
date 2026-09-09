import torch

raw = torch.load("gpt2-small-124M.pth", map_location="cpu")
state_dict = raw.get("model", raw.get("state_dict", raw))

# GPT-2 Conv1D weights are [in, out]; transpose to Linear's [out, in]
# if your C++ module uses standard Linear layers. Skip this if your
# module already expects the Conv1D layout as-is.
conv1d_suffixes = ("attn.c_attn.weight", "attn.c_proj.weight",
                   "mlp.c_fc.weight", "mlp.c_proj.weight")
for k in list(state_dict.keys()):
    if k.endswith(conv1d_suffixes):
        state_dict[k] = state_dict[k].t().contiguous()

class TensorContainer(torch.nn.Module):
    def __init__(self, sd):
        super().__init__()
        for k, v in sd.items():
            self.register_buffer(k.replace(".", "__"), v)

container = torch.jit.script(TensorContainer(state_dict))
container.save("gpt2-small-124M.torchscript")
print(f"Exported {len(state_dict)} tensors:")
for k in state_dict: print(" ", k)