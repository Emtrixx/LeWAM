# PyTorch's locked Linux x86_64 wheels provide CUDA 12.8/cuDNN/NCCL.
# Host requirements: an NVIDIA driver compatible with CUDA 12.8 and Container Toolkit.
FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258 AS runtime
COPY --from=ghcr.io/astral-sh/uv:0.12.19@sha256:04d046b13e60d6bcec73cbc5e1cad25d680dea90c8573340950a0ac2d1aef424 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git ffmpeg libglib2.0-0 libgl1 libgomp1 build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
    LEWAM_DATA_DIR=/data/pusht
COPY pyproject.toml uv.lock ./
# Docker caches this dependency layer; avoid retaining a second multi-GB wheel cache.
RUN uv sync --frozen --no-dev --no-install-project --no-cache
COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-cache
COPY docker ./docker
RUN chmod +x /app/docker/entrypoint.sh
ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["lewam-train", "experiment=pusht", "hardware=h100", "output_dir=/runs/default", "hydra.run.dir=/runs/default"]

FROM runtime AS test
COPY tests ./tests
RUN uv sync --frozen --group dev
RUN pytest -q
