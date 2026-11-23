# Krea 2 image generation in the web UI

Start the existing web app and worker with `bash scripts/start.sh`. Choose
**Krea 2 RAW** in the model catalog, open **Configure generation**, enter a
prompt, and click **Generate images**. Krea 2 uses the shared responsive form,
queue, live job progress, image gallery, and PNG/JSON download links.

Controls include positive/negative prompts, dimensions, image count, seed,
inference steps, guidance, sampling, prompt sequence length, prompt prefix/suffix,
trigger words, and multiple independently weighted LoRAs. The guidance hint
updates when guidance or the negative prompt changes: negative prompting is
active above guidance 0 with a nonempty negative prompt. Krea 2 does not expose
FLUX's separate true CFG control.

The sampler dropdown comes from the shared registry, filtered by the checkpoint's
`requirements.allowed_sampling`. Big Love KreaEdit3 selects **ER SDE** by default
and also offers **Flow match Euler**, **Flow match Heun**, **UniPC**, **DPM++ 2M**,
and **DPM++ 2M SDE**. All six samplers offer **Default**, **Normal**,
**Karras**, **Exponential**, and **Beta** noise schedules. **Default** remains selected,
preserving Krea 2's resolution-aware schedule. To edit the `sampling` row on the model page,
enter the complete JSON object:

```json
{"sampler": "er_sde", "noise_schedule": "default"}
```

**Preview request** checks form values without loading weights or queueing a
job. **Save model settings**, **Reset model settings**, and shared adapter
save/reset actions use the same persistence and conflict checks as other families.
Generation snapshots the current settings without saving them. A repeated
submission returns its original job; changed values with the same submission
key are rejected.

The job page shows runtime/model loading, model reuse or switching, LoRA setup,
denoising, decoding, and saving. Images and sidecars go under
`Images/<job_id>/` by default. Configure `JOB_OUTPUT_ROOT` or the launcher's
`--output-dir` to choose a different location. Failures display their error and
release the pipeline before the next job.

For a form demo without weights, run:

```bash
IMAGE_GENERATOR_CATALOG_ROOT=tests/fixtures/model_families .venv/bin/python main.py
```

Open `/generate/fixture-krea2`. Fixture weights are intentionally absent;
previewing and saving preferences work without a GPU. Generation requires real
weights and the [runtime configuration](Krea2_Workflow.md).

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_krea2_execution.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_ui_contract.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_sampler_registry.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_er_sde.py' -v
```

Tests use the real form, queue, worker, workflow, and PNG/JSON saving with a
synthetic model runtime. They cover RAW loading, guidance semantics, validation,
multi-adapter strengths, adapter reset, checkpoint reuse/switching, live progress,
failure recovery, duplicate submissions, output isolation, and downloads.

For a GPU smoke test, generate one small image with a fixed seed and no adapters,
then enable the notebook adapter, generate again, and clear the selection for a
third image. Confirm progress, downloads, and adapter reset with model reuse.
