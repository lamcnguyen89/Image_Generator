"""Construct a registered native ComfyUI sampler using the loaded node API."""


def build_native(*, spec, nodes, **runtime):
    return nodes.KSamplerSelect.execute(sampler_name=spec.native_name)[0]
