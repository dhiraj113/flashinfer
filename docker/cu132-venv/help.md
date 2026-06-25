# cu132-venv Docker Setup

## How it works

- **Dockerfile** installs system dependencies (Python 3.12, CUDA dev tools, mpich) at image build time. No paths or Python packages are baked into the image.
- **`.env`** is the single place to configure all paths. Everything else is derived from it.
- **`docker-compose.yml`** mounts the host NFS directory, injects all environment variables, and runs a startup script that creates and populates the venv on first run, then skips setup on subsequent runs. The container then stays alive with `sleep infinity`.
- The container runs as your host user (uid/gid hardcoded in `docker-compose.yml`) so it can write to the NFS filesystem. Docker's daemon runs as root, which NFS root-squashes — all writes must happen as you.

All paths are mounted at the same absolute path as on the host (host `/home/scratch.dnallapa_gpu/work` → container `/home/scratch.dnallapa_gpu/work`), so paths in `.env` are valid both on your machine and inside the container.

> All commands below are run from the directory containing `docker-compose.yml` (`docker/cu132-venv/`).

## Step 1 — Configure `.env`

```bash
vi .env
```

| Variable | Description |
|---|---|
| `CONTAINER_NAME` | Name of the container, used with `docker exec -it <CONTAINER_NAME> bash` |
| `HOST_UID` / `HOST_GID` | Your host user and group IDs (`id -u` / `id -g`). The container runs as this user to write to NFS. |
| `HOST_MOUNT_PATH` | Single NFS directory mounted into the container. All other paths must be subdirectories of this. |
| `PYTHON_VENV_PATH` | Directory on the host where the venv lives. Must exist before `docker compose up`. |
| `PYTHON_VENV_NAME` | Name of the venv directory, created inside `PYTHON_VENV_PATH` by the container on first run. |
| `GIT_REPO_PATH` | Absolute path to this repo on the host. |
| `FLASHINFER_JIT_CACHE_PATH` | Where FlashInfer stores compiled JIT kernels. Persists across container restarts. Must exist before `docker compose up`. |

**Constraint:** `PYTHON_VENV_PATH`, `GIT_REPO_PATH`, and `FLASHINFER_JIT_CACHE_PATH` must all be subdirectories of `HOST_MOUNT_PATH`. The startup script validates this and exits with an error if not.

## Step 2 — Ensure required directories exist on the host

`PYTHON_VENV_PATH`, `GIT_REPO_PATH`, and `FLASHINFER_JIT_CACHE_PATH` must all exist on the host before starting the container.

`GIT_REPO_PATH` (the repo) must also already exist.

## Step 3 — Build the image

```bash
docker compose build
```

Installs system packages only. Python packages are installed at first startup, not during build.

## Step 4 — Start the container

```bash
docker compose up -d
```

**First run:** creates the venv and installs torch and all Python dependencies (~5–10 min depending on network), then goes idle.

**Subsequent runs:** venv already exists, validation passes, setup is skipped, goes idle immediately.

## Step 5 — Open a shell

```bash
docker exec -it <CONTAINER_NAME> bash
```

The venv is already activated (PATH and VIRTUAL_ENV are set by compose). You can immediately run:

```bash
pytest tests/gemm/test_mm_fp4.py -v
python -c "import flashinfer; print(flashinfer.__version__)"
```

## Stopping and restarting

```bash
# Stop (venv and cache on host are unaffected)
docker compose down

# Restart later — setup is skipped, goes idle in seconds
docker compose up -d
```

## Resetting the venv

If the venv becomes broken or you want a clean reinstall:

```bash
docker compose down
rm -rf $PYTHON_VENV_PATH/$PYTHON_VENV_NAME
docker compose up -d   # triggers full reinstall on next start
```
