"""Exercise combined-service shutdown with real, isolated child processes."""

import argparse
import errno
import io
import os
from pathlib import Path
import json
import shutil
import signal
import subprocess
import sys
import tempfile
from threading import Event
import time
import unittest
from unittest import mock

import dotenv


ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "scripts/start.sh"

SERVICE = '''
from pathlib import Path
import json
import os
import signal
import sys
import time
root = Path(os.environ['LAUNCHER_TEST_ROOT'])
if name == 'worker' and os.environ.get('LAUNCHER_TEST_FAIL_WORKER'):
    deadline = time.monotonic() + 5
    while not (root / 'web.ready').exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    raise SystemExit(2)
def stop(number, frame):
    if name == 'worker':
        time.sleep(0.15)  # Finish an active generation before exiting.
    (root / (name + '.finished')).write_text('saved and stopped')
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
(root / (name + '.arguments')).write_text(json.dumps({
    'argv': sys.argv[1:], 'cwd': str(Path.cwd()),
    'database': os.environ.get('IMAGE_GENERATOR_JOB_DATABASE'),
    'outputs': os.environ.get('IMAGE_GENERATOR_JOB_OUTPUT_ROOT'),
    'video_outputs': os.environ.get('IMAGE_GENERATOR_JOB_VIDEO_OUTPUT_ROOT'),
}))
(root / (name + '.ready')).write_text(str(os.getpid()))
deadline = time.monotonic() + 10
while time.monotonic() < deadline:
    trigger = root / (name + '.exit')
    if trigger.exists():
        raise SystemExit(int(trigger.read_text()))
    time.sleep(0.01)
'''


class HostnameAddressTests(unittest.TestCase):
    def setUp(self):
        source = START.read_text().split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
        namespace = {"__name__": "launcher_test"}
        with mock.patch.object(sys, "argv", [str(START), str(ROOT)]):
            exec(compile(source, str(START), "exec"), namespace)
        self.hostname_command = namespace["hostname_command"]
        self.parser = argparse.ArgumentParser(prog="start.sh")
        self.publisher = mock.patch("shutil.which", return_value="/usr/bin/avahi-publish-address")
        self.publisher.start()
        self.addCleanup(self.publisher.stop)

    def test_automatic_alias_uses_lan_route_instead_of_usb_multicast_route(self):
        with mock.patch("socket.socket") as socket_factory:
            probe = socket_factory.return_value.__enter__.return_value

            def select_route(destination):
                address = "192.168.55.1" if destination[0] == "224.0.0.251" else "192.168.1.81"
                probe.getsockname.return_value = (address, 12345)

            probe.connect.side_effect = select_route
            name, command = self.hostname_command("midnightgenerator", "0.0.0.0", self.parser)
        self.assertEqual(name, "midnightgenerator.local")
        self.assertEqual(command[-1], "192.168.1.81")

    def test_automatic_alias_fails_until_main_route_is_available(self):
        with mock.patch("socket.socket") as socket_factory, mock.patch("sys.stderr", new_callable=io.StringIO):
            probe = socket_factory.return_value.__enter__.return_value

            def select_route(destination):
                if destination[0] != "224.0.0.251":
                    raise OSError(errno.ENETUNREACH, "Network is unreachable")
                probe.getsockname.return_value = ("192.168.55.1", 12345)

            probe.connect.side_effect = select_route
            with self.assertRaises(SystemExit) as stopped:
                self.hostname_command("midnightgenerator", "0.0.0.0", self.parser)
        self.assertEqual(stopped.exception.code, 2)

    def test_explicit_lan_address_works_without_a_main_route(self):
        with mock.patch("socket.socket") as socket_factory:
            _, command = self.hostname_command("midnightgenerator", "192.168.1.81", self.parser)
            socket_factory.assert_not_called()
        self.assertEqual(command[-1], "192.168.1.81")


class StartTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "application with spaces"
        scripts = self.root / "scripts"
        scripts.mkdir(parents=True)
        self.launcher = scripts / "start.sh"
        shutil.copy2(START, self.launcher)
        python = self.root / ".venv/bin/python"
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        # Exercise the complete shell entry point with isolated CPU-only services.
        flask = self.root / "fakes/flask"
        flask.mkdir(parents=True)
        shutil.copytree(Path(dotenv.__file__).parent, self.root / "fakes/dotenv")
        (flask / "__init__.py").write_text("")
        (flask / "cli.py").write_text(
            "def load_dotenv():\n"
            "    from dotenv import load_dotenv\n"
            "    load_dotenv('.env')\n"
        )
        (flask / "__main__.py").write_text("name = 'web'\n" + SERVICE)
        worker = self.root / "src/image_generator"
        worker.mkdir(parents=True)
        (worker / "__init__.py").write_text("")
        (worker / "worker.py").write_text("name = 'worker'\n" + SERVICE)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        publisher = self.bin / "avahi-publish-address"
        publisher.write_text(f"#!{sys.executable}\nname = 'hostname'\n" + SERVICE)
        publisher.chmod(0o755)

    def wait_ready(self, names, process):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if all((self.root / f"{name}.ready").exists() for name in names):
                return
            if process.poll() is not None:
                self.fail(f"Service exited before becoming ready: {process.returncode}")
            Event().wait(0.01)
        self.fail("Timed out waiting for child services")

    def start(self, *, wait_for=("web", "worker"), fail_worker=False, arguments=()):
        environment = os.environ.copy()
        for key in ("IMAGE_GENERATOR_HOSTNAME", "IMAGE_GENERATOR_HOST", "IMAGE_GENERATOR_PORT",
                    "IMAGE_GENERATOR_JOB_VIDEO_OUTPUT_ROOT"):
            environment.pop(key, None)
        environment["PATH"] = str(self.bin) + os.pathsep + environment.get("PATH", "")
        environment["PYTHONPATH"] = str(self.root / "fakes")
        environment["LAUNCHER_TEST_ROOT"] = str(self.root)
        if fail_worker:
            environment["LAUNCHER_TEST_FAIL_WORKER"] = "1"
        process = subprocess.Popen(["bash", str(self.launcher), "--database", str(self.root / "queue.sqlite3"),
                                    "--output-dir", str(self.root / "generated images"),
                                    "--device", "cuda:3", "--dtype", "float32", *arguments],
                                   cwd="/tmp", env=environment, start_new_session=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(self.close, process)
        self.wait_ready(wait_for, process)
        return process

    @staticmethod
    def close(process):
        if process.poll() is None:
            process.terminate()
        try:
            process.communicate(timeout=12)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=12)

    def assert_reaped(self, name):
        pid = int((self.root / f"{name}.ready").read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_ctrl_c_stops_web_and_waits_for_worker_to_finish(self):
        process = self.start()
        for name in ("web", "worker"):
            settings = json.loads((self.root / f"{name}.arguments").read_text())
            self.assertEqual(settings["database"], str(self.root / "queue.sqlite3"))
            self.assertEqual(settings["outputs"], str(self.root / "generated images"))
            self.assertEqual(settings["video_outputs"], str(self.root / "Videos"))
            self.assertEqual(settings["cwd"], str(self.root))
        self.assertEqual(settings["argv"][settings["argv"].index("--video-output-dir") + 1],
                         str(self.root / "Videos"))
        self.assertEqual(settings["argv"][-4:], ["--device", "cuda:3", "--dtype", "float32"])
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=5)
        self.assertEqual(process.returncode, 130, stderr)
        self.assertIn("active generation", stdout)
        for name in ("web", "worker"):
            self.assertTrue((self.root / f"{name}.finished").is_file())
            self.assert_reaped(name)

    def test_video_output_setting_is_shared_from_dotenv(self):
        (self.root / ".env").write_text("IMAGE_GENERATOR_JOB_VIDEO_OUTPUT_ROOT=custom videos\n")
        self.start()
        for name in ("web", "worker"):
            settings = json.loads((self.root / f"{name}.arguments").read_text())
            self.assertEqual(settings["video_outputs"], str(self.root / "custom videos"))
        self.assertEqual(settings["argv"][settings["argv"].index("--video-output-dir") + 1],
                         str(self.root / "custom videos"))

    def test_video_output_option_overrides_dotenv_for_both_services(self):
        (self.root / ".env").write_text("IMAGE_GENERATOR_JOB_VIDEO_OUTPUT_ROOT=custom videos\n")
        self.start(arguments=("--video-output-dir", "selected videos"))
        for name in ("web", "worker"):
            settings = json.loads((self.root / f"{name}.arguments").read_text())
            self.assertEqual(settings["video_outputs"], str(self.root / "selected videos"))
        self.assertEqual(settings["argv"][settings["argv"].index("--video-output-dir") + 1],
                         str(self.root / "selected videos"))

    def test_service_failure_stops_companion_and_preserves_failure_status(self):
        process = self.start()
        (self.root / "web.exit").write_text("7")
        stdout, stderr = process.communicate(timeout=5)
        self.assertEqual(process.returncode, 7, stderr)
        self.assertIn("Web server exited", stderr)
        self.assertTrue((self.root / "worker.finished").is_file())
        self.assert_reaped("web")
        self.assert_reaped("worker")

    def test_worker_startup_failure_stops_web_server(self):
        process = self.start(wait_for=("web",), fail_worker=True)
        stdout, stderr = process.communicate(timeout=5)
        self.assertEqual(process.returncode, 2, stderr)
        self.assertIn("Generation worker exited", stderr)
        self.assertTrue((self.root / "web.finished").is_file())
        self.assert_reaped("web")


    def test_dotenv_hostname_and_port_publish_alias_until_shutdown(self):
        (self.root / ".env").write_text(
            "IMAGE_GENERATOR_HOSTNAME=My-Images\n"
            "IMAGE_GENERATOR_HOST=192.168.1.42\n"
            "IMAGE_GENERATOR_PORT=5050\n"
        )
        process = self.start(wait_for=("hostname", "web", "worker"))
        publisher = json.loads((self.root / "hostname.arguments").read_text())
        self.assertEqual(publisher["argv"], ["--no-reverse", "my-images.local", "192.168.1.42"])
        web = json.loads((self.root / "web.arguments").read_text())["argv"]
        self.assertEqual(web[web.index("--host") + 1], "192.168.1.42")
        self.assertEqual(web[web.index("--port") + 1], "5050")
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=5)
        self.assertEqual(process.returncode, 130, stderr)
        self.assertIn("http://my-images.local:5050", stdout)
        self.assertTrue((self.root / "hostname.finished").is_file())
        for name in ("hostname", "web", "worker"):
            self.assert_reaped(name)

    def test_command_line_overrides_dotenv_and_local_suffix_is_not_duplicated(self):
        (self.root / ".env").write_text(
            "IMAGE_GENERATOR_HOSTNAME=imagegen.local\n"
            "IMAGE_GENERATOR_HOST=127.0.0.1\n"
            "IMAGE_GENERATOR_PORT=invalid\n"
        )
        process = self.start(wait_for=("hostname", "web", "worker"),
                             arguments=("--host", "192.168.1.43", "--port", "6000"))
        publisher = json.loads((self.root / "hostname.arguments").read_text())
        self.assertEqual(publisher["argv"], ["--no-reverse", "imagegen.local", "192.168.1.43"])
        web = json.loads((self.root / "web.arguments").read_text())["argv"]
        self.assertEqual(web[web.index("--host") + 1], "192.168.1.43")
        self.assertEqual(web[web.index("--port") + 1], "6000")

    def test_no_hostname_keeps_local_defaults_and_does_not_start_publisher(self):
        self.start()
        self.assertFalse((self.root / "hostname.ready").exists())
        web = json.loads((self.root / "web.arguments").read_text())["argv"]
        self.assertEqual(web[web.index("--host") + 1], "127.0.0.1")
        self.assertEqual(web[web.index("--port") + 1], "5000")

    def test_invalid_hostname_port_and_loopback_fail_before_starting_services(self):
        for setting, message in (
            ("IMAGE_GENERATOR_HOSTNAME=imagegen.local:5000", "without http:// or a port"),
            ("IMAGE_GENERATOR_HOSTNAME=http://imagegen", "without http:// or a port"),
            ("IMAGE_GENERATOR_PORT=abc", "must be an integer"),
            ("IMAGE_GENERATOR_PORT=65536", "must be between"),
            ("IMAGE_GENERATOR_PORT=0", "must be between"),
            ("IMAGE_GENERATOR_HOSTNAME=imagegen\nIMAGE_GENERATOR_HOST=127.0.0.1", "LAN IPv4 address"),
        ):
            with self.subTest(setting=setting):
                (self.root / ".env").write_text(setting + "\n")
                process = self.start(wait_for=())
                stdout, stderr = process.communicate(timeout=5)
                self.assertEqual(process.returncode, 2, stderr)
                self.assertIn(message, stderr)
                for name in ("hostname", "web", "worker"):
                    self.assertFalse((self.root / f"{name}.ready").exists())

    def test_hostname_publisher_failure_stops_both_services(self):
        (self.root / ".env").write_text(
            "IMAGE_GENERATOR_HOSTNAME=imagegen\nIMAGE_GENERATOR_HOST=192.168.1.42\n"
        )
        process = self.start(wait_for=("hostname", "web", "worker"))
        (self.root / "hostname.exit").write_text("1")
        stdout, stderr = process.communicate(timeout=5)
        self.assertEqual(process.returncode, 1, stderr)
        self.assertIn("Local hostname exited", stderr)
        for name in ("web", "worker"):
            self.assertTrue((self.root / f"{name}.finished").is_file())
        for name in ("hostname", "web", "worker"):
            self.assert_reaped(name)


if __name__ == "__main__":
    unittest.main()
