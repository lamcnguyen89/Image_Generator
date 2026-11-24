#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
venv_python="$project_root/.venv/bin/python"

if [[ ! -x "$venv_python" ]]; then
    printf 'Project Python is missing. Run bash scripts/setup.sh first.\n' >&2
    exit 1
fi

# Keep the process supervisor in this file. exec preserves signals and exit
# status, and the quoted heredoc passes Python code without shell expansion.
exec "$venv_python" -u - "$project_root" "$@" <<'PY'
"""Start the local web server and generation worker in one terminal."""

import argparse
from ipaddress import IPv4Address
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
from threading import Event


PROJECT_ROOT = Path(sys.argv[1])


def hostname_command(hostname, host, parser):
    """Publish an app-specific .local alias without renaming the computer."""
    label = hostname.lower().removesuffix(".local")
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label):
        parser.error("IMAGE_GENERATOR_HOSTNAME must be a name such as imagegen (or imagegen.local), without http:// or a port")
    publisher = shutil.which("avahi-publish-address")
    if publisher is None:
        parser.error("Local hostname publishing needs avahi-utils and a running avahi-daemon. On Ubuntu: sudo apt install avahi-utils avahi-daemon")
    try:
        address = IPv4Address(host)
        if address.is_unspecified:
            # Select the main network route, not the multicast route: at boot,
            # multicast can select a USB bridge before Wi-Fi is connected.
            # This documentation-only destination receives no packets; UDP
            # connect only selects a local address and needs no internet access.
            # If no route exists yet, fail so systemd retries once it is ready.
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.connect(("192.0.2.1", 9))
                address = IPv4Address(probe.getsockname()[0])
    except (ValueError, OSError) as error:
        parser.error(f"Cannot select a local IPv4 address for the hostname: {error}. Set --host or IMAGE_GENERATOR_HOST to your LAN IPv4 address.")
    if address.is_loopback or address.is_unspecified or address.is_multicast:
        parser.error("Local hostname publishing requires --host / IMAGE_GENERATOR_HOST to be 0.0.0.0 or your LAN IPv4 address")
    return f"{label.lower()}.local", [publisher, "--no-reverse", f"{label.lower()}.local", str(address)]


def supervise(commands, *, cwd, env):
    """Stop the companion service on exit and allow the worker to finish a job."""
    processes = {}
    stopping = Event()
    received_signal = None

    def request_stop(number, frame):
        nonlocal received_signal
        received_signal = number
        stopping.set()

    previous_handlers = {
        number: signal.signal(number, request_stop)
        for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    try:
        for name, command in commands:
            if stopping.is_set():
                break
            # Only the launcher receives terminal Ctrl+C. It forwards SIGTERM,
            # which the worker handles by finishing its current job.
            processes[name] = subprocess.Popen(command, cwd=cwd, env=env, start_new_session=True)
        while not stopping.wait(0.2):
            for name, process in processes.items():
                status = process.poll()
                if status is not None:
                    print(f"{name} exited (status {status}); stopping services.", file=sys.stderr, flush=True)
                    return status if status > 0 else 1
        return 128 + received_signal
    except OSError as error:
        print(f"Could not start services: {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        if processes:
            print("Stopping services; waiting for any active generation to finish…", flush=True)
        for process in processes.values():
            # Signal the worker itself, allowing its encoders and other child
            # processes to finish normally with the current generation.
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            process.wait()
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)


def main(argv=None):
    arguments = sys.argv[2:] if argv is None else argv
    parser = argparse.ArgumentParser(
        prog="start.sh",
        description=__doc__,
        epilog="Other options are passed to the generation worker, e.g. --device cuda:0 --dtype float16 or --local-files-only.",
        allow_abbrev=False,
    )
    parser.add_argument("--host", help="Listening address (overrides IMAGE_GENERATOR_HOST; default: 0.0.0.0 with a hostname, otherwise 127.0.0.1)")
    parser.add_argument("--port", type=int, help="Web server port (overrides IMAGE_GENERATOR_PORT; default: 5000)")
    parser.add_argument("--database", type=Path, help="Queue database for both services")
    parser.add_argument("--output-dir", type=Path, help="Image output root for both services (default: Images)")
    parser.add_argument("--video-output-dir", type=Path, help="Video output root for both services (default: Videos)")
    args, worker_arguments = parser.parse_known_args(arguments)
    os.chdir(PROJECT_ROOT)
    try:
        from flask.cli import load_dotenv
    except ImportError:
        parser.exit(1, "Web dependencies are missing. Run bash scripts/setup.sh first.\n")
    # Load .env/.flaskenv before spawning either process, so prefixed Flask
    # settings and worker dotenv settings agree on their shared paths.
    load_dotenv()
    hostname = os.environ.get("IMAGE_GENERATOR_HOSTNAME", "").strip()
    args.host = args.host or os.environ.get("IMAGE_GENERATOR_HOST") or ("0.0.0.0" if hostname else "127.0.0.1")
    if args.port is None:
        try:
            args.port = int(os.environ.get("IMAGE_GENERATOR_PORT", "5000"))
        except ValueError:
            parser.error("IMAGE_GENERATOR_PORT must be an integer between 1 and 65535")
    if not 1 <= args.port <= 65535:
        parser.error("--port / IMAGE_GENERATOR_PORT must be between 1 and 65535")
    alias = hostname_command(hostname, args.host, parser) if hostname else None
    database = (args.database or Path(os.environ.get("IMAGE_GENERATOR_JOB_DATABASE") or
                                     PROJECT_ROOT / "data/jobs.sqlite3")).expanduser().resolve()
    outputs = (args.output_dir or Path(os.environ.get("IMAGE_GENERATOR_JOB_OUTPUT_ROOT") or
                                      PROJECT_ROOT / "Images")).expanduser().resolve()
    video_outputs = (args.video_output_dir or Path(os.environ.get("IMAGE_GENERATOR_JOB_VIDEO_OUTPUT_ROOT") or
                                                  PROJECT_ROOT / "Videos")).expanduser().resolve()
    environment = os.environ.copy()
    environment["IMAGE_GENERATOR_JOB_DATABASE"] = str(database)
    environment["IMAGE_GENERATOR_JOB_OUTPUT_ROOT"] = str(outputs)
    environment["IMAGE_GENERATOR_JOB_VIDEO_OUTPUT_ROOT"] = str(video_outputs)
    environment["PYTHONUNBUFFERED"] = "1"
    source = str(PROJECT_ROOT / "src")
    environment["PYTHONPATH"] = source + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    python = str(PROJECT_ROOT / ".venv/bin/python")
    commands = [
        ("Web server", [python, "-u", "-m", "flask", "--app", "image_generator:create_app", "run",
                        "--host", args.host, "--port", str(args.port), "--no-reload"]),
        ("Generation worker", [python, "-u", "-m", "image_generator.worker",
                               "--database", str(database), "--output-dir", str(outputs),
                               "--video-output-dir", str(video_outputs), *worker_arguments]),
    ]
    if alias:
        name, command = alias
        commands.insert(0, ("Local hostname", command))
        print(f"Publishing local address: http://{name}:{args.port}", flush=True)
    print(f"Starting web server at http://{args.host}:{args.port}", flush=True)
    print("Starting generation worker. Press Ctrl+C to stop both services.", flush=True)
    return supervise(commands, cwd=PROJECT_ROOT, env=environment)


if __name__ == "__main__":
    raise SystemExit(main())

PY
