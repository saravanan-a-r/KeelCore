"""
Convert a trained LoRA adapter (safetensors) into the encoder/decoder
.onnx_adapter files ONNX Runtime loads, in the tensor layout of the shipped
professional reference adapter.

    /root/venv/bin/python SFT_adapter_rephrase/tools/export_onnx_adapter.py \\
        <adapter.safetensors> <adapter_config.json> <out_dir>

lora_A is written transposed, lora_B transposed and pre-multiplied by the
adapter's scaling; the learned frame-token embedding rows go along as
token_delta.ids / token_delta.values.
"""

import json
import sys

import numpy as np
import onnxruntime as ort
from safetensors import safe_open

REFERENCE_DIR = "/home/admin1/projects/EchoMeBetter/service_package_onnx/adapters/professional"


def reference_layout(ref_dir: str) -> dict:
    """{graph: (adapter_version, model_version, LoRA tensor names)} of the reference adapter."""
    layout = {}
    for graph in ("encoder", "decoder"):
        adapter = ort.AdapterFormat.read_adapter(f"{ref_dir}/{graph}.onnx_adapter")
        names = sorted(k for k in adapter.get_parameters() if not k.startswith("token_delta"))
        layout[graph] = (adapter.get_adapter_version(), adapter.get_model_version(), names)
    return layout


def export(st_path: str, cfg_path: str, out_dir: str, ref: str = REFERENCE_DIR) -> list[str]:
    """Write <out_dir>/{encoder,decoder}.onnx_adapter; returns the trainable tokens in id order."""
    with open(cfg_path, encoding="utf-8") as handle:
        cfg = json.load(handle)
    scaling = cfg["scaling"]
    tensors = safe_open(st_path, "np")
    ids = np.array(sorted(cfg["trainable_tokens"].values()), dtype=np.int64)
    order = [token for token, _ in sorted(cfg["trainable_tokens"].items(), key=lambda kv: kv[1])]
    delta = tensors.get_tensor("token_embedding_delta").astype(np.float32)
    assert delta.shape[0] == len(ids), (delta.shape, len(ids))

    for graph, (adapter_version, model_version, names) in reference_layout(ref).items():
        params = {}
        for name in names:
            weight = tensors.get_tensor(name).astype(np.float32)
            params[name] = np.ascontiguousarray(weight.T if name.endswith(".lora_A") else (weight * scaling).T)
        params["token_delta.ids"] = ids
        params["token_delta.values"] = np.ascontiguousarray(delta)
        fmt = ort.AdapterFormat()
        fmt.set_adapter_version(adapter_version)
        fmt.set_model_version(model_version)
        fmt.set_parameters({k: ort.OrtValue.ortvalue_from_numpy(v) for k, v in params.items()})
        fmt.export_adapter(f"{out_dir}/{graph}.onnx_adapter")
        print(graph, len(params), "tensors")
    return order


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__.strip())
    export(*sys.argv[1:4])
