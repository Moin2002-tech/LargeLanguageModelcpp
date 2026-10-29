import json, os, torch
import tensorflow as tf

def export_for_cpp(model_dir, out_path="gpt2_weights.pt"):
    ckpt = tf.train.latest_checkpoint(model_dir)
    params = {}
    for name, _ in tf.train.list_variables(ckpt):
        arr = tf.train.load_variable(ckpt, name).squeeze()
        params[name.replace("model/", "")] = torch.from_numpy(arr)

    class Holder(torch.nn.Module):
        def __init__(self, tensors):
            super().__init__()
            for k, v in tensors.items():
                # buffer names can't contain '.' or '/'
                self.register_buffer(k.replace("/", "__"), v)

    torch.jit.script(Holder(params)).save(out_path)

export_for_cpp("models/124M")


