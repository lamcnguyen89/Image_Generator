# Durable queue and worker lifecycle

This is the first execution-infrastructure stage. It adds a workflow-independent
SQLite queue, executable request snapshots, and a dedicated single-worker process.
SDXL and FLUX web forms submit image jobs; WAN and MiniMax forms submit video jobs.
All expose loading/denoising progress and saved results. See
[SDXL web execution](SDXL_UI_Execution.md) and [FLUX web execution](FLUX_UI_Execution.md).

## Test it without a GPU

From the project root, run the standalone test-only demo:

```bash
.venv/bin/python tests/demo_queue.py
```

It creates a fresh directory under `/tmp/image-generator-queue-demo-*`, submits
two jobs using different fake workflow names, processes them sequentially,
reopens the database, and prints the completed job IDs and four real 16×16 PNG
paths. The PNGs and matching JSON metadata remain there for inspection. No model
weights, downloads, Torch, Diffusers, or ComfyUI are used. Fake workflows are
injected by the demo/tests and are never registered for production submissions.

Run the queue and worker tests:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_jobs.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_worker.py' -v
```

Or run the full regression suite:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Tests cover persistence, concurrent submitters, settings/LoRA snapshots, local
asset changes, FIFO execution, real fake-image output and metadata, workflow
reuse/switching, generation/saving errors, graceful stopping, heartbeats,
exclusive ownership, and recovery after killing a real worker subprocess.
Typed video output references are tested using a mocked video encoder.

## Submit a prepared request through Python

The queue accepts a `GenerationRequest` prepared by the existing generation API:

```python
from image_generator.catalog import discover_catalog
from image_generator.generation import prepare_generation
from image_generator.jobs import JobStore

catalog = discover_catalog("Models/Model_Families")
request = prepare_generation(
    catalog,
    "model_a4278bb0-94f4-4960-ada0-39ca89cec194",  # SDXL base entry
    overrides={"prompt": "A mountain lake at sunrise", "seed": 42, "loras": []},
)
store = JobStore("data/jobs.sqlite3")
job = store.enqueue(request)
print(job.id, job.status)
```

Submission resolves/validates settings and assets without loading the GPU
pipeline. The worker independently validates the restored request again before
execution. `enqueue()` itself checks the serializable shared contract and local
asset references; it does not replace workflow-specific preparation/validation.

The model family and workflow come from the snapshot. There are no SDXL-specific
settings columns in SQLite. FLUX, WAN, and MiniMax can use the same queue/worker
API; video inputs must be real local files, not the filenames returned by the
current web form's stateless preview.

## Run the worker

In a separate terminal/process:

```bash
.venv/bin/python -m image_generator.worker
```

Defaults are `data/jobs.sqlite3`, `Images/<job_id>/` for images, and
`Videos/<job_id>/` for videos. Each job's media, conditioning PNG, and JSON
metadata are grouped in its own directory.

The Generation History page has a **Delete** button at the right of each job
row. Completed, failed, and interrupted jobs can be deleted; queued and running
jobs must finish first. Deletion permanently removes the history record, its
generated media, metadata, conditioning images, and any private uploaded source
owned by that job. Both image and video directories are cleaned, including
partial files. Original inputs outside managed upload storage and model assets
are preserved. If file cleanup fails, the history entry remains so deletion can
be retried. Deletion requires a same-session POST with a valid CSRF token.

Worker options include:

```bash
.venv/bin/python -m image_generator.worker --once
.venv/bin/python -m image_generator.worker --drain
.venv/bin/python -m image_generator.worker \
  --database /path/to/jobs.sqlite3 --output-dir /path/to/images \
  --video-output-dir /path/to/videos \
  --device cuda:0 --local-files-only
```

`--once` claims at most one job; `--drain` exits when the queue becomes empty.
The default mode polls for new work. Ctrl+C and SIGTERM request a graceful stop:
finish the current job, save its results, then stop claiming jobs and close the
workflow. This stage does not provide active-job cancellation.

Runtime cache/token handling follows the existing CLIs: project `.env`,
`CACHE_LOCATION`, `HUGGINGFACE_TOKEN`/`HF_TOKEN`, and optional `--cache-dir`.
Runtime options stay in worker memory rather than job rows or output metadata.
The video root can also be set with `IMAGE_GENERATOR_JOB_VIDEO_OUTPUT_ROOT`;
`--video-output-dir` takes precedence. Configure the web server to use the same
video root for playback and downloads. The launcher shares this setting between
both processes automatically.
`--dtype` and `--comfyui-dir` are available; MiniMax's existing restriction to
checkpoint-managed precision still applies, so omit `--dtype` for mixed queues.

The library also supports explicit lifecycle control and per-workflow options:

```python
from image_generator.worker import Worker

with Worker(store, "Images", video_output_dir="Videos", runtime_options={
    "sdxl": {"device": "cuda:0", "local_files_only": True},
    "minimax": {"comfyui_dir": "build/ComfyUI", "local_files_only": True},
}) as worker:
    finished = worker.run_once()  # Job or None when no queued work remains.
```

Start/process/close a worker on its owner thread. Use `request_stop()` to ask
an active worker to finish gracefully; do not close its pipeline from another
thread. Each instance retains one workflow, reusing it for consecutive jobs of
the same workflow. The workflow handles checkpoint switches. Switching families
closes the previous workflow before constructing another. Any job failure closes
the active workflow before the next job; cleanup failure stops the worker.

## Lifecycle, durability, and recovery

Jobs transition through:

```text
queued → running → completed
                 → failed
                 → interrupted
```

SQLite stores creation/start/finish timestamps, worker identity, errors, stage
progress, request snapshots, and ordered output references. Writes use short
transactions with WAL mode and full synchronization. Generation and external
asset I/O occur outside database transactions.

One process may own a queue database at a time. A POSIX advisory file lock at
`<database>.worker.lock` is held for the whole worker lifetime. Another worker
fails immediately instead of claiming another GPU job. Keep the lock file in
place: its existence does not mean a worker is active; the OS lock does.
This implementation targets a single host and local filesystem, not a
distributed queue or a network-filesystem deployment.

The OS releases ownership after process death. On the next successful start,
the new owner marks all previous `running` jobs `interrupted` and leaves queued
jobs intact. Interrupted jobs are not retried automatically, avoiding duplicate
generation when the exact crash point is unknown. Already recorded outputs are
retained. Partial or unregistered files may remain if a process dies between
filesystem saving and the SQLite output commit; job completion is never inferred
from files alone.

A background heartbeat updates `worker_state` even during long generation calls.
An old heartbeat does not authorize taking ownership from a live worker. Heartbeat
failure stops further claiming; active work checks worker health between batches.

Shared stages include `validating`, `preparing`, `generating`, and `saving`, followed
by the terminal state. SDXL and FLUX report additional model, adapter, denoising, and
decoding stages through the shared `WorkflowProgress` callback interface. Counts
are stage-specific: denoising timesteps, adapter counts, or saved outputs.
The worker detaches observers after every job, throttles fast progress updates,
and persists stage history. WAN and MiniMax also report video-specific stages.
A job can complete only after at least one output has been saved and recorded.

Database schema version 1 migrates automatically to version 2, adding stage events
and optional submission keys without discarding jobs or outputs. `enqueue(request,
submission_key=...)` deduplicates a repeated form submission transactionally; using
the same key for different request values raises `SubmissionConflict`.

## Snapshots, assets, and results

`job_requests.py` provides versioned request encoding/reconstruction. It freezes
the effective settings, prepared prompt, full model/LoRA metadata and settings,
adapter strengths/names, mode, and absolute local asset/input paths. Workers do
not rediscover the catalog or recompute prompt additions from today's JSON files.
Subsequent edits to defaults or user manifests do not change queued work.

Local model/adapter/component files and pipeline directories are checked using
file identity, size, and timestamps; input files additionally receive SHA-256
hashes. Missing or detectably changed references fail before constructing the
workflow. Large model tensors are not read or copied to enqueue a job. These
checks are not a cryptographic weight archive and do not pin remote repositories.
Input files must remain available and unchanged; managed UI upload storage comes
with the later submission integration.

Inspect results through the store:

```python
job = store.get(job.id)
print(job.status, job.stage, job.error)
for output in job.outputs:
    print(output.id, output.kind, output.file_path, output.metadata_path, output.seed)

print(store.list_jobs(status="queued", limit=20))
print(store.worker_status())
```

Output kinds are `image` and `video`; video records also reference their
conditioning image. Each output has a stable ID for future result-serving routes.
PNG/MP4 files and JSON sidecars use the existing generation saving functions.
Completed batches are recorded as they are saved, so earlier outputs remain
inspectable even if a later streamed batch fails.
