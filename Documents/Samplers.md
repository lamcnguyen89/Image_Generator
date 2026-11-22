# Shared sampler registry

`workflows/shared/samplers/registry.py` defines trusted sampler bindings keyed by
`(workflow, sampler name)`. Each binding supplies its display label, a schedule
backend profile, and a lazy runtime factory. Bindings can also declare source
schedule constraints, incompatible options, and native scheduler classes and
defaults. Forms, settings saves, request validation, and runtime
construction use these bindings. Importing capabilities and forms does not load
Torch or Diffusers. Krea2, SDXL, FLUX, Qwen 2.1, and WAN use the registry.
Noise-schedule names, labels, backend configuration, and compatibility live in
the separate [schedule registry](Schedules.md), also used by MiniMax's native
Simple schedule.

Krea2 implements `flow_match_euler`, `er_sde`, and the existing `model_default`
Euler alias. Every request constructs a fresh scheduler from the original
component scheduler configuration, allowing model reuse without retaining solver
history between jobs. Output metadata includes the selected sampling settings,
actual scheduler class and configuration, seeds, and library versions.

FLUX and Qwen 2.1 register `flow_match_euler` and the existing `model_default`
Euler alias. Their UI choices, settings
saves, backend validation, and scheduler construction use the same registry.
Checkpoint `allowed_sampling` restrictions and scheduler configuration apply to
each request. Qwen's explicit `sigmas` and repository `sample_sigmas` retain their
existing precedence, progress reporting, and metadata. Registering samplers for
SDXL or Krea2 does not make them available to FLUX or Qwen automatically. Flow
samplers support Default, Normal, Karras, Exponential, and Beta where checkpoint
restrictions allow them. An explicit Qwen `sigmas` list requires Default.

SDXL implements `model_default`, `euler`, `euler_ancestral` (Euler A), and
`dpmpp_2m_sde` (DPM++ 2M SDE). Native scheduler classes and default options live
in their registry bindings. `model_default` uses the original loader's scheduler
class; switching from another sampler restores that original configuration.
Checkpoint scheduler configuration overrides sampler defaults, preserving SDXL's
existing handling of prediction type, zero-SNR rescaling, and other requirements.
SDXL's Euler and DPM++ 2M SDE bindings support `default`, `normal`, `karras`,
`exponential`, and `beta`. Euler A supports Default and Normal; model default
retains Default only. Normal generates an explicit training-timestep grid.
The default schedule inherits the checkpoint loader's configuration. Karras
explicitly sets `use_karras_sigmas: true` and disables alternative sigma grids;
conflicting checkpoint schedule options fail validation. WAN retains its
loader's default scheduler and schedule; MiniMax retains res_multistep/Simple.

## Checkpoint configuration

The selected sampler and noise schedule remain separate settings:

```json
"sampling": {
  "sampler": "er_sde",
  "noise_schedule": "default"
}
```

If `requirements.allowed_sampling` is present, it must include the exact pair:

```json
"allowed_sampling": [
  {"sampler": "er_sde", "noise_schedule": "default"},
  {"sampler": "flow_match_euler", "noise_schedule": "default"}
]
```

Omitting the restriction allows every registered pair for that workflow; an
empty list allows none. A manifest cannot import an implementation or enable a
sampler in another workflow. The Big Love KreaEdit3 manifest permits both pairs
above and selects ER_SDE. Other checkpoint restrictions continue to apply.

For SDXL, select either of these objects in `settings.sampling` (or the model
page's `sampling` editor), and include the pair in `allowed_sampling` when that
restriction is present:

```json
{"sampler": "euler_ancestral", "noise_schedule": "default"}
```

```json
{"sampler": "dpmpp_2m_sde", "noise_schedule": "default"}
```

For **DPM++ 2M SDE Karras**, keep the sampler ID and select the Karras schedule:

```json
{"sampler": "dpmpp_2m_sde", "noise_schedule": "karras"}
```

Lustify GGWP v7 uses this pair by default. In the generation form, select
**DPM++ 2M SDE**, then choose **Karras** under **Advanced settings → Noise
schedule**. Choices follow the selected sampler and checkpoint restrictions;
changing to a sampler that does not support Karras selects its first allowed
schedule. Backend validation also checks the exact pair, including for CLI and
inline JSON edits.

## ER_SDE implementation

`FlowMatchERSDEScheduler` adapts the VP ER-SDE Taylor method to Krea2 flow
predictions, using `alpha = 1 - sigma`, `lambda = sigma / alpha`, and the clean
prediction `x0 = sample - sigma * velocity`. It ramps from first to third order,
uses 200 quadrature points and the noise scaler
`phi(lambda) = lambda * (exp(lambda**0.3) + 10)`, and returns the clean prediction
at the final zero sigma. See [Cui et al.](https://arxiv.org/abs/2309.06169) and the
[authors' reference implementation](https://github.com/QinpengCui/ER-SDE-Solver).

The existing Diffusers flow scheduler prepares Krea2's resolution-aware schedule.
An initial sigma of exactly 1 is moved to `1 - 1e-4`, with a matching model
timestep, to avoid infinite lambda. The adapter requires strictly decreasing
sigmas in `[0, 1)` ending at zero. Inverted schedules and the separate Euler
`stochastic_sampling` option are unsupported for ER_SDE.

Krea2Pipeline passes generators to latent preparation but not scheduler steps.
The adapter therefore binds the request's generators, one per image, and uses
them for every injected noise tensor. It never relies on global random state.
Scalar integration coefficients are computed once in CPU float64; latent updates
use float32 and return the model's dtype. Numerical tests cover reference outputs,
batch independence, seeds, resolution shifting, history reset, and bfloat16.
Image equivalence with an external workflow also depends on that workflow's
schedule, prediction conversion, conditioning, and random-number consumption.

## Adding a sampler or workflow

1. Implement a factory in a runtime module. It receives the registry `spec`,
   resolved `sampling`, checkpoint `requirements`, and runtime inputs supplied by
   its workflow. The
   factory must return fresh state and implement the pipeline's scheduler
   interface, including prediction conversion and seeded noise where needed.
2. Add an explicit `SamplerSpec` binding in the registry. Register each supported
   workflow separately, even when sampler names are shared. For native Diffusers
   schedulers, reuse `diffusers.build_native` with `scheduler_class` and
   `scheduler_options`. `schedule_class` separately constrains the source schedule
   for adapters that require a specific schedule preparation method, such as Krea2.
3. Use `shared.samplers.ui.sampling_fields` for form choices and
   `validate_sampling` for both form saves and backend request validation.
   Construct the scheduler with `create_scheduler` in the workflow runtime.
4. Add numerical reference and workflow integration coverage before allowing
   the new pair in a checkpoint manifest.

Derived workflows that reuse a built-in implementation can inherit its sampler
bindings through the runtime's `sampler_workflow_id`. Their form builder passes
the same base name to `sampling_fields(model, workflow=...)`, while the model
and job retain the custom workflow identity. This is explicit implementation
inheritance; unrelated workflow names receive no sampler bindings automatically.
Form saves validate the declared sampling pair while preserving unedited
extension metadata; execution keeps the strict runtime settings contract.

Additional schedule definitions and backend bindings belong in
`shared/schedules/registry.py`. Sampler factories obtain their constructor
options from that registry; native ComfyUI scheduling resolves its registered
name. The form's dependent choices come from these registered sampling pairs,
and Python validates those dependencies even when JavaScript is disabled.
