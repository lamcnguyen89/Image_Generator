# MiniMax H3 generation in the web interface

Start the Flask web app and shared generation worker from the project root:

```bash
bash scripts/start.sh --device cuda:0 --comfyui-dir build/ComfyUI
```

Leave `--dtype` unset: H3 precision is managed by the checkpoint and ComfyUI.
The current backend requires CUDA device 0 and a ComfyUI checkout with native
H3 support. See [MiniMax setup](MiniMax_Workflow.md) for checkpoint, video VAE,
and Qwen3-VL text encoder configuration. A missing referenced encoder may be
downloaded by the worker unless `--local-files-only` is supplied.

Open a MiniMax checkpoint's **Configure generation** page and choose **Image to
video** or **Text to video** in **Generation mode**. The controls refresh when
the mode changes; **Apply generation mode** also works without JavaScript.
Shared prompt, sampling, and adapter edits are preserved. For image-to-video,
select a source image. For text-to-video, enter a prompt and set width and height;
no image is needed. Choose **Generate video**. The request freezes its settings
and selected LoRA strengths without changing saved model or adapter settings.
The web process validates and queues; the worker loads and executes native
ComfyUI Python nodes. No separate ComfyUI server is required.

## Controls and source images

Text-to-video uses the configured dimensions directly and omits image conditioning.
Image-to-video requires a source and offers source-aspect preservation. Switching
back to image-to-video requires selecting the source again. Both modes share
the loaded checkpoint, encoder, VAE, and configured LoRAs in the worker.

H3 uses dimensions divisible by 32 (or stricter checkpoint alignment), frame
counts of `17*n + 5` below the 3600-frame ceiling, and silent video at 24 FPS.
CFG 1, an empty negative prompt, and RES multistep/simple sampling are read-only;
Python rejects forged values. Multiple explicitly H3-compatible LoRAs share
one list with individual strengths. Zero-strength adapters are skipped. Changing
or clearing adapters rebuilds from the cached unpatched base. Compatibility
declarations do not establish checkpoint verification or apply recommendations.

The source thumbnail uses a local browser object URL. **Preview request**
validates the image without saving it. Select the image again after a preview
or validation response because browsers clear file selections.

Generation retains the original image bytes, including EXIF orientation, in
immutable managed storage. Identical retries reuse the source and job; changed
images or settings under the same submission key return 409. New sources are
removed if preparation or enqueueing fails. Sources survive web/worker restarts
and are served only through the associated job's managed source route.
MiniMax shares WAN's upload limits and configuration; see
[source storage and retries](WAN_UI_Execution.md#source-storage-and-retries).

## Progress and results

The job page shows runtime preparation, text encoder resolution, component
loading/reuse, adapter loading/reset/reuse, conditioning and prompt encoding,
sampling steps, VAE decoding, and MP4 encoding/saving. ComfyUI's existing sampler
callback remains intact. Its process-global progress hook is bridged only during
sampling and restored in `finally`, including failures. Only numeric progress is
persisted; tensor and preview payloads are discarded. Variation and saved-video
counts are separate from the active sampling steps, without a whole-job ETA.

Each saved MP4 appears immediately with playback, seeking, download, and metadata
links. Image-to-video results show the original source and exact conditioning PNG;
text-to-video saves no source or conditioning image. Both modes show actual dimensions,
frame count, 24 FPS, duration, actual seed, and silent audio status from metadata.
Earlier variations remain accessible if a later one fails. Output routes are
confined to the associated job's registered files.

Web jobs write MP4/PNG/JSON files under `JOB_VIDEO_OUTPUT_ROOT` (`Videos/<job_id>/`
by default). The standalone MiniMax runner also defaults to `Videos/`. Previously
saved web videos under `Images/<job_id>/` remain playable and downloadable.

## Verification

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_minimax*.py' -v
.venv/bin/python -m unittest discover -s tests -v
```

Integration tests execute the real MiniMax workflow and shared worker with mocked
native nodes and real MP4 encoding of synthetic frames. They cover uploads,
fixed values, CSRF, retry conflicts, cleanup, restart persistence, progress visible
mid-sampling, hook restoration, byte-range seeking, metadata, managed links, and
partial results. Backend tests cover component reuse, checkpoint changes,
adapter stacking/reset, zero strengths, and quantized-transfer contexts.

For real GPU validation, start with one video, a fixed seed, no adapters,
small model-valid dimensions and frame count, and a low step count. Check
browser playback and metadata. Then verify component reuse, enabling an adapter,
clearing adapters, checkpoint switching, and cleanup when changing families.
Synthetic tests do not establish successful GPU inference or adapter quality.
