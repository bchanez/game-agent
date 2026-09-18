# machine-learning

The goal is to play mario
https://www.smbgames.be/super-mario-bros.php

# Run (nothing to install locally, everything lives in Docker)

Everything runs inside a Docker container: Python, OpenCV, Jupyter Lab.
The only requirement on your machine is Docker.

## From VSCode (recommended)

Run the task **`Mario: Start (build + open)`**
(Terminal → Run Task…, or Cmd+Shift+B) — it builds the image, starts the
container, and opens Jupyter Lab in your browser at http://localhost:8888/lab

Other tasks available:
- `Docker: Jupyter Up` — build + start the container
- `Open Jupyter Lab` — open the browser tab
- `Docker: Jupyter Down` — stop and remove the container
- `Docker: Logs` — follow the container logs
- `Docker: Rebuild (no cache)` — full rebuild

## From the terminal (equivalent)

```sh
docker compose -f docker/docker-compose.yml up -d --build   # start
docker compose -f docker/docker-compose.yml down            # stop
```

Then open http://localhost:8888/lab and run `src/detection.ipynb`.

# Useful docker commands

Remove all unused containers, networks and images:

```sh
docker system prune -a
```
