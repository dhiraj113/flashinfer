#!/bin/bash
set -e

# Validate that all key paths are under HOST_MOUNT_PATH (which is the single
# mounted volume). Fails fast if .env is misconfigured.
for _path in "$VIRTUAL_ENV" "$GIT_REPO_PATH" "$FLASHINFER_JIT_CACHE_PATH" "$PIP_CACHE_DIR"; do
    case $_path in
        "$HOST_MOUNT_PATH"/*) ;;
        *) echo "ERROR: $_path is not under HOST_MOUNT_PATH ($HOST_MOUNT_PATH)"; exit 1 ;;
    esac
done

mkdir -p "$FLASHINFER_JIT_CACHE_PATH"
mkdir -p "$PIP_CACHE_DIR"

# Create and populate the venv on first run; skip if it already exists.
if [ ! -f "$VIRTUAL_ENV/bin/python" ]; then
    echo "Creating virtual environment at $VIRTUAL_ENV"
    python3.12 -m venv "$VIRTUAL_ENV"
    pip install --upgrade 'setuptools>=77' 'pip>=24'
    pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/nightly/cu132
    pip install -r "$GIT_REPO_PATH/requirements.txt"
    pip install responses pytest pytest-xdist scipy build cuda-python nvshmem4py-cu12
    pip install --upgrade cuda-python==13.0 nvidia-cudnn-cu13 'nvidia-cutlass-dsl[cu13]>=4.5.0'
    pip install tilelang cuda-tile
    pip install mpi4py
    pip install --no-build-isolation -e "$GIT_REPO_PATH"
else
    echo "Virtual environment already exists, skipping setup."
fi

# Written after both first-run install and fast restart so the HEALTHCHECK
# always has a consistent ready signal regardless of which path ran.
touch "$VIRTUAL_ENV/.ready"

exec "$@"
