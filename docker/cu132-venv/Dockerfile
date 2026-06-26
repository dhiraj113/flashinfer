FROM nvidia/cuda:13.2.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    git \
    wget \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    mpich \
    libmpich-dev \
 && rm -rf /var/lib/apt/lists/*

# Create the host user inside the container so getent/passwd lookups work.
# VS Code Dev Containers uses getent to find the home directory; without this
# it falls back to /root/.vscode-server which is not writable by the user.
ARG HOST_USERNAME=user
ARG HOST_UID=1000
ARG HOST_GID=1000
RUN groupadd -f -g ${HOST_GID} ${HOST_USERNAME} \
 && useradd -m --uid ${HOST_UID} --gid ${HOST_GID} --shell /bin/bash ${HOST_USERNAME}

COPY docker/cu132-venv/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set prompt using HOST_USERNAME env var (injected at runtime by docker compose).
RUN echo 'PS1="${HOST_USERNAME:-user}@\h:\w\$ "' >> /etc/bash.bashrc
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
