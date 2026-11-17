# WAN generation in the web interface

Start the web server and shared worker together:

```bash
bash scripts/start.sh --device cuda:0
```

Open the single **Wan 2.2** catalog card's **Configure generation** page and
choose **Image to video** or **Text to video** in the mode dropdown. The form
refreshes automatically; **Apply generation mode** also works without JavaScript.
Changing modes loads that checkpoint's saved settings and adapters and keeps
your prompt. Each queued job records the concrete checkpoint and selected mode.

For image-to-video, select a source image. Text-to-video uses the requested width
and height directly and requires no source image. Enter a prompt, configure the
video variations, and choose **Generate video**.
The form validates frame count, dimensions, both guidance values, and each
adapter's explicit expert target before queueing. Settings and adapter selections
are frozen for that job. The web process does not import the GPU runtime.

Image-to-video previews validate image content without saving uploads. After a preview or
validation response, select the source again; the page explicitly reminds you.

## Source storage and retries

Text-to-video jobs do not stage or retain source files. Their retries use the
same submission-key checks on the checkpoint, mode, and settings.

Original image bytes are stored at `data/inputs/<submission_key>/source.<format>`.
The extension comes from decoded content; the client filename never determines
the destination. Files are read-only and are retained for job provenance and
restarts. The workflow applies EXIF orientation and prepares the resized/cropped
conditioning image separately.

Identical retries reuse the path and file fingerprint and redirect to the same
job. Different image bytes or request values with the same submission key return
409. A per-submission filesystem lock serializes upload/enqueue/cleanup across
web processes. Validation or enqueue failures remove newly staged sources.
Source previews are only served at `/jobs/<job_id>/source` for registered inputs
confined to managed input storage.

Configuration defaults (also configurable with the `IMAGE_GENERATOR_` prefix):

| Setting | Default |
| --- | --- |
| `JOB_INPUT_ROOT` | Project `data/inputs/` |
| `INPUT_MAX_BYTES` | 20 MiB per source |
| `INPUT_MAX_PIXELS` | 25 million decoded pixels |
| `MAX_CONTENT_LENGTH` | 24 MiB for the complete HTTP request |

PNG, JPEG, WebP, BMP, TIFF, and GIF single-frame images are accepted after full
decode validation. Corrupt, animated, or oversized images receive form errors.

## Progress and outputs

The job page shows runtime/checkpoint loading or reuse, image conditioning,
adapter reset/loading, the active variation, denoising steps, video decoding,
and encoding/saving. Saved video counts are separate from the active variation's
steps; these steps do not estimate whole-job time.

Each saved variation appears immediately as a playable MP4 with byte-range
seeking and download links. Both modes show the seed, actual output dimensions,
frame count, FPS, duration, and JSON metadata. Image-to-video also shows the
original source and exact conditioning PNG; text-to-video creates neither.
Previously saved videos remain available if a later variation
fails. Polling appends new results without rebuilding existing video players.

Queued outputs use `JOB_VIDEO_OUTPUT_ROOT` (project `Videos/<job_id>/` by default),
including MP4 and JSON files, plus a conditioning PNG for image-to-video. The standalone WAN CLI continues
to default to `Videos/`. Previously saved web videos under `Images/<job_id>/`
remain playable and downloadable.

## Validation

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_wan_execution.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_wan_t2v_execution.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_transformer_workflows.py' -v
```

These tests use the real WAN workflow with a mocked Diffusers runtime and create
small real MP4s from synthetic frames. They exercise multipart submission,
immutable retry handling, cleanup, restarts, progress observed during denoising,
expert strengths/reset, checkpoint switching, failure recovery, range requests,
MP4 decoding, actual metadata, and source/output path confinement.

For a real GPU smoke test, use a known source, one video, no adapters, BF16,
5 frames, small model-valid dimensions (for example 64 × 64), and 2–4 steps.
The configured base model may need to download its pretrained components; local
WAN weights must be a complete Diffusers pipeline containing both experts.
Confirm playback and conditioning dimensions in the browser. Repeat with both
expert adapter groups, clear them, and switch checkpoints on the same worker.

For text-to-video, omit the source image. Verify output dimensions and video
playback, then switch between both modes in one worker. Check that the T2V
checkpoint reuses the notebook's shared Hugging Face cache. Only adapters with
explicit `wan_2_2_t2v_a14b` compatibility and an expert target may be enabled for
text-to-video; existing I2V adapters remain available for image-to-video.
