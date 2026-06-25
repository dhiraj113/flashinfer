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

COPY docker/cu132-venv/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set prompt using HOST_USERNAME env var (injected at runtime by docker compose).
RUN echo 'PS1="${HOST_USERNAME:-user}@\h:\w\$ "' >> /etc/bash.bashrc
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
