"""Qwen adapter conversion and compatibility policy."""

import re


def split_fused_lora(state_dict):
    """Convert ComfyUI's gate/up LoRA exactly: shared A and split B output rows."""
    result = dict(state_dict)
    suffix = ".img_mlp.gate_up.lora_A.weight"
    for key in state_dict:
        if not key.endswith(suffix):
            continue
        up_key = key.removesuffix("lora_A.weight") + "lora_B.weight"
        down, up = result.pop(key), result.pop(up_key)
        if len(down.shape) != 2 or len(up.shape) != 2 or up.shape[0] % 2 or up.shape[1] != down.shape[0]:
            raise ValueError(f"Invalid fused gate/up LoRA dimensions: {key}")
        gate, projection = up.chunk(2, dim=0)
        for name, weight in (("gate_layer", gate), ("proj", projection)):
            for old, value in ((key, down), (up_key, weight)):
                target = old.replace(".img_mlp.gate_up.", f".img_mlp.{name}.")
                if target in result:
                    raise ValueError(f"LoRA contains both fused and separate weights: {target}")
                result[target] = value
    return result


def load_qwen21_lora(pipeline, path, adapter_name):
    """Load a complete named adapter, normalizing fused Qwen 2.1 projections."""
    # Let Diffusers normalize native/AI Toolkit keys and alpha scaling, then
    # apply the same gate-first split as its Qwen 2.1 checkpoint converter.
    state, metadata = pipeline.lora_state_dict(
        str(path.parent),
        weight_name=path.name,
        local_files_only=True,
        use_safetensors=True,
        return_lora_metadata=True,
    )
    state = split_fused_lora(state)
    if not state:
        raise ValueError("The LoRA contains no adapter weights")
    for key in state:
        match = re.fullmatch(r"transformer\.(.+)\.lora_[AB]\.weight", key)
        if match is None:
            raise ValueError(f"Unsupported Qwen 2.1 LoRA key: {key}")
        # PEFT otherwise logs and ignores unknown targets, silently applying
        # only part of an incompatible adapter. Fail before installing it.
        pipeline.transformer.get_submodule(match[1])
    pipeline.load_lora_into_transformer(
        state,
        transformer=pipeline.transformer,
        adapter_name=adapter_name,
        metadata=metadata,
        _pipeline=pipeline,
        low_cpu_mem_usage=True,
    )
