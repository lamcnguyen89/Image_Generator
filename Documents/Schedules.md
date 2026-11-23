# Shared noise-schedule registry

Samplers define the numerical update; noise schedules define the noise levels
at which updates run. Diffusers scheduler classes combine both responsibilities,
so this project keeps a sampler registry and a separate schedule registry while
constructing one compatible runtime scheduler for each request.

`workflows/shared/schedules/registry.py` owns stable names, display labels,
backend bindings, required configuration flags, trusted adapters, and native ComfyUI names.
Discovery and validation import no Torch, Diffusers, or ComfyUI runtime modules.
The [sampler registry](Samplers.md) selects a schedule backend profile for each
sampler. UI choices, CLI choices, settings saves, request validation, and runtime
construction resolve the same registered definitions.

## Supported schedules

| Sampler/workflow | Available schedules before checkpoint restrictions |
| --- | --- |
| SDXL Euler | Default, Normal, Karras, Exponential, Beta |
| SDXL DPM++ 2M SDE | Default, Normal, Karras, Exponential, Beta |
| SDXL DPM++ 2M / 3M / 3M SDE, UniPC | Default, Normal, Karras, Exponential, Beta |
| SDXL Euler A | Default, Normal, Karras, Exponential, Beta |
| SDXL Heun, LMS, DPM2, DPM2 ancestral, DEIS | Default, Karras, Exponential, Beta |
| SDXL DDIM, DDPM, PNDM | Default |
| SDXL model default | Default |
| FLUX, Qwen 2.1, Krea2: Flow Match Euler / Heun, ER SDE, UniPC, DPM++ 2M / 2M SDE, model default | Default, Normal, Karras, Exponential, Beta |
| WAN model default, UniPC, DPM++ 2M | Default, Normal, Simple |
| MiniMax H3 RES multistep, Euler | Simple, Normal, Karras, Exponential, Beta |

Default preserves the original component scheduler configuration and checkpoint
overrides. Selecting a named curve explicitly enables its flag and disables
competing grids supported by that backend. For example, selecting Exponential
on DPM++ enables `use_exponential_sigmas` and disables Karras, Beta, Lu, and flow
sigma grids. Conflicting `requirements.scheduler_config` flags fail validation.
Defaults and named schedules use fresh request state, allowing model reuse
without carrying scheduler history or previous schedule flags between jobs.

Euler ancestral uses a schedule adapter for Karras, Exponential, and Beta.
Diffusers' Euler scheduler prepares the sigma grid and matching fractional model
timesteps using the checkpoint's training configuration. Euler ancestral keeps
its native input scaling, prediction conversion, and seeded stochastic update.
The adapter resets solver indices on every run and records the selected curve
and original scheduler class in output metadata.

The supplied FLUX, Qwen 2.1, and Krea2 checkpoint manifests and examples permit
Default, Normal, Karras, Exponential, and Beta for all six explicit flow samplers.
Their selected defaults are unchanged. Checkpoint-specific restrictions such as
NoobAI's Euler/Default requirement remain in force. WAN defaults to Default and
MiniMax defaults to Simple; additional video schedule choices do not change these selections.

WAN Normal uses evenly spaced unshifted training times from 1 through
`1 / num_train_timesteps`, inclusive. WAN Simple samples the discrete training
grid using the reverse-table indices from ComfyUI Simple. Both support 1 through
`num_train_timesteps` inference steps. Native UniPC applies WAN's flow shift once,
appends zero, and offsets the initial sigma below one to avoid singular log-SNR.
All three WAN samplers use the selected grid's exact model timesteps for expert
selection. Default retains its original step-dependent grid and configuration.
Named WAN schedules reject conflicting checkpoint curve, prediction, and grid flags.

MiniMax resolves Simple, Normal, Karras, Exponential, and Beta to ComfyUI's native
schedule names for both RES multistep and Euler. The runtime validates the
descending flow grid and records actual sigmas and the effective inference-step
count under `scheduler` in output metadata. Beta can remove duplicate training
indices, reducing the effective count; initial sampling progress uses that count.

Heun, LMS, DPM2, DPM2 ancestral, and DEIS use native curve flags through separate
profiles. Normal is not registered for these profiles; the existing adapter has
only been verified for the profiles offering it above. Heun and DPM2 also require
special handling of intermediate evaluation grids. DDIM, DDPM, and PNDM retain
their native Default schedules.

Normal generates an evenly spaced grid across the model's training timesteps.
For SDXL, its adapter interpolates the training log-sigma table and appends zero,
following [ComfyUI's Normal schedule](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/samplers.py)
when given the same training table. It supplies the grid explicitly because
native DPM++ timestep spacing differs. Discrete Normal supports up to the
scheduler's number of training timesteps. For flow models, Normal uses the
native full training-range grid with the model's shift; the pipeline's usual
step-count-dependent input grid is replaced. Default continues to inherit the
checkpoint configuration and pipeline grid. Identical output across backends
also depends on model configuration, solver behavior, and random state.

These bindings use the installed Diffusers implementations:
[Euler](https://huggingface.co/docs/diffusers/api/schedulers/euler),
[DPM multistep](https://huggingface.co/docs/diffusers/api/schedulers/multistep_dpm_solver),
and [Flow Match Euler](https://huggingface.co/docs/diffusers/api/schedulers/flow_match_euler_discrete).
Supporting a schedule in one backend does not automatically enable it elsewhere.

## Checkpoint configuration

The existing JSON structure is retained:

```json
"sampling": {
  "sampler": "dpmpp_2m_sde",
  "noise_schedule": "normal"
}
```

Checkpoint `requirements.allowed_sampling`, when present, permits exact pairs:

```json
"allowed_sampling": [
  {"sampler": "dpmpp_2m_sde", "noise_schedule": "default"},
  {"sampler": "dpmpp_2m_sde", "noise_schedule": "normal"},
  {"sampler": "dpmpp_2m_sde", "noise_schedule": "karras"},
  {"sampler": "dpmpp_2m_sde", "noise_schedule": "exponential"},
  {"sampler": "dpmpp_2m_sde", "noise_schedule": "beta"}
]
```

An omitted allowlist permits every registered pair for the selected workflow;
an empty list permits none. Existing checkpoint defaults and allowlists remain
authoritative. A manifest cannot register code, select arbitrary Python classes,
or bypass backend compatibility. The generation form filters schedule choices
as the sampler changes, and Python independently validates the exact pair.

The CLI equivalents are `--sampler dpmpp_2m_sde
--sampling-noise-schedule normal` (or `exponential`, `beta`, `karras`, `default`).

## Qwen sigma lists and metadata

With Default selected, Qwen retains its existing precedence: an explicit
`settings.sigmas` list overrides repository `sample_sigmas`, which overrides the
step-count-derived input grid. Resolution-aware shifting still applies through
the native scheduler. Karras, Exponential, and Beta may transform the
repository/default grid; Normal replaces it with the full training-range grid.
Combining a named schedule with an explicit custom sigma list fails validation so
the custom input cannot be silently reinterpreted.

Diffusers output metadata records the selected sampling pair, actual scheduler
class/configuration, and actual `scheduler.sigmas` and `scheduler.timesteps`
where available. Qwen also records `input_sample_sigmas` and
`effective_sample_sigmas` (the applied sigma grid without the terminal endpoint,
falling back to the input grid for runtimes that expose no sigma tensor).
MiniMax records its sampling selection and ComfyUI runtime revision as before.
Seeded scheduler noise remains owned by the sampler implementation.

## Adding a schedule

1. Add a `ScheduleSpec` with a stable name and display label.
2. Register `ScheduleBinding` entries only for verified backend profiles. Native
   Diffusers bindings declare constructor flags; native ComfyUI bindings declare
   `native_name`. Adapter-only configuration goes in `adapter_options`; it is
   checked against checkpoint overrides without being passed to the underlying
   scheduler constructor. A new curve requiring different numerical code also needs a
   compatible runtime adapter before it can be registered; declare its trusted
   module and class in `ScheduleBinding.adapter`, as Normal does.
3. For a new sampler/backend implementation, set its `schedule_backend` in
   `SamplerSpec`. Profile membership is explicit; adding a global schedule name
   never grants support to every backend.
4. Verify resulting sigma grids, constructor conflicts, repeatability, scheduler
   resets, checkpoint pair restrictions, and UI saves before enabling the pair
   in a checkpoint's allowlist.

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_schedule_registry.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_diffusers_samplers.py' -v
```
