# Start Image Generator at boot

Only `image-generator.service` is needed. It runs `bash scripts/start.sh`,
which supervises the Flask web server, generation worker, and optional Avahi
hostname publisher together. Separate web and worker units are unnecessary.
The service runs as `malneyugnfl` and uses this checkout at
`/home/malneyugnfl/SoftwareDev/Image_Generator`. Change `User=`,
`WorkingDirectory=`, and the path in `ExecStart=` if installing elsewhere.

## Prerequisites

Complete [environment setup](../Documents/Setup.md) first. The launcher uses
the project's `.venv` directly and loads the project's `.env` automatically;
activating a virtual environment or adding `EnvironmentFile=` is unnecessary.
Services do not source your interactive shell configuration. Put application
settings in `.env` instead. Additional worker arguments, such as
`--device cuda:0 --dtype float16`, can be appended to `ExecStart=`.

If `IMAGE_GENERATOR_HOSTNAME` is set in `.env`, install and enable Avahi:

```bash
sudo apt install avahi-utils avahi-daemon
sudo systemctl enable --now avahi-daemon.service
```

Avahi is an existing system service; no custom unit is needed for it. Leave
`IMAGE_GENERATOR_HOSTNAME` empty to disable hostname publishing. The app unit
orders startup after Avahi when it is being started, without requiring Avahi
for configurations that do not use a hostname.

With `IMAGE_GENERATOR_HOST=0.0.0.0`, alias publishing selects the main network
route. If Wi-Fi is not ready at boot and no route is available, the launcher
exits and `Restart=on-failure` retries every five seconds. It does not fall back
to the Jetson USB bridge. Restart the service if the network address changes
while it is running.

## Install and enable

From the project root, use the convenience toggle:

```bash
bash scripts/toggle_services.sh
```

When the service is absent, stopped, or failed, this installs the current unit
into `/etc/systemd/system/`, enables it at boot, and starts it. Run the command
again while the installed service is running to stop it, disable startup at
boot, and delete the installed unit. It uses `sudo` for system changes when
needed and works from any directory when invoked by its full path. It preserves
the project unit, `.env`, models, queue, outputs, and Avahi installation.
Stopping waits for the launcher shutdown described below. A failed stop or
disable aborts removal. A masked service or a unit loaded from another location
is reported without modifying that installation.

For manual installation instead:

Run these commands from the project root. Stop any manually started copy of
`scripts/start.sh` first so it does not conflict with the managed service.

```bash
sudo install -m 0644 systemd/image-generator.service /etc/systemd/system/image-generator.service
sudo systemctl daemon-reload
sudo systemctl enable --now image-generator.service
systemctl status image-generator.service
```

This starts the app now and on subsequent boots, without an interactive login.
It restarts after a launcher or child-process failure. The existing queue,
models, outputs, and caches remain accessible as the configured user.

## Logs, shutdown, and updates

```bash
journalctl -u image-generator.service -f
sudo systemctl stop image-generator.service
sudo systemctl restart image-generator.service
```

`KillMode=mixed` sends the initial stop signal to the launcher, which asks its
children to stop and waits for the worker's active job. Systemd allows up to
30 minutes before forcibly terminating remaining processes; increase
`TimeoutStopSec=` if generations need longer. A stop, restart, or system
shutdown can wait this long. See the official
[systemd shutdown documentation](https://www.freedesktop.org/software/systemd/man/latest/systemd.kill.html).

After changing the unit, repeat the `install` and `daemon-reload` commands,
then restart the service. After changing `.env`, restart the service to load
the new settings. To stop the app and disable automatic startup:

```bash
sudo systemctl disable --now image-generator.service
```
