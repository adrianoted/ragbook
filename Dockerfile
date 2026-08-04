# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Builder stage: build a self-contained virtualenv with CPU-only PyTorch.
#
# The pin `torch==2.10.0` (requirements.txt) is satisfied by the local wheel
# variant `2.10.0+cpu` from the PyTorch CPU index. Per PEP 440 a local version
# matches the `==2.10.0` specifier and sorts ABOVE the plain public version, so
# pip prefers `+cpu` when the extra index is present — no CUDA wheels pulled.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Compilers for any dependency without a prebuilt manylinux wheel. This stage
# is discarded, so build-essential never reaches the runtime image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# ---------------------------------------------------------------------------
# Runtime stage: slim image + Tesseract OCR (with Italian language data),
# the prebuilt venv, a non-root user, and only the application source.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# OCR runtime dependency for pytesseract (image ingest); `-ita` covers the
# Italian demo documents. A few MB; no other extra packages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-ita \
    && rm -rf /var/lib/apt/lists/*

# Copy the ready-to-run virtualenv from the builder.
COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # HuggingFace cache on a writable path; mounted as a named volume in compose
    # so the ~8GB embedding model download survives container recreation.
    HF_HOME=/home/appuser/.cache/huggingface

# Non-root user. Pre-create $HF_HOME so the fresh hf-cache named volume
# inherits appuser ownership (Docker seeds an empty volume with the
# mountpoint's permissions from the image). /app/data is instead a bind mount
# in compose — ownership comes from the host dir (tracked via data/.gitkeep);
# pre-created here only for runs without a mount.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data "$HF_HOME" \
    && chown -R appuser:appuser /app "$HF_HOME"

WORKDIR /app
COPY --chown=appuser:appuser src/ ./src/

USER appuser

EXPOSE 8000

# run.py is intentionally NOT the entrypoint: it enables reload from DEBUG and
# force-exits via os._exit(0), neither of which suits a container. Serve uvicorn
# directly.
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
