# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Canonical KIWI EP / MORI EP benchmark and profiling image.
#
# Build the canonical public ROCm/PyTorch-derived image, e.g.:
#   docker build -f docker/ep-bench/Dockerfile \
#     --build-arg KIWI_IMAGE_REVISION="$(git rev-parse HEAD)" \
#     -t kiwi-ep-bench:rocm7.2.4-pytorch2.10 .
#
# The base is intentionally a concrete public ROCm/PyTorch tag rather than a
# required build arg or an unqualified latest tag.  Benchmark reports should
# still record the resolved base and derived image digests.
FROM rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0

ARG DEBIAN_FRONTEND=noninteractive
ARG KIWI_IMAGE_REVISION=unknown
# Metadata only: this does not clone KIWI. The checkout is bind-mounted at runtime.
ARG KIWI_IMAGE_SOURCE=https://github.com/AMDResearch/kiwi
ARG INTELLIKIT_REPO=https://github.com/AMDResearch/intellikit.git
# Pinned to AMDResearch/intellikit commit inspected when this image workflow was added.
# Override with a site-approved tag or commit when refreshing benchmark images.
ARG INTELLIKIT_REF=3f45cd314d455b652a1246678511b40547fe521e
ARG INTELLIKIT_TOOLS=accordo,kerncap,linex,metrix,nexus,rocm_mcp,uprof_mcp
ARG INTELLIKIT_SMOKE_COMMAND

LABEL org.opencontainers.image.title="kiwi-ep-bench" \
      org.opencontainers.image.description="KIWI EP, MORI EP, and IntelliKit benchmark/profiling environment" \
      org.opencontainers.image.source="$KIWI_IMAGE_SOURCE" \
      org.opencontainers.image.revision="$KIWI_IMAGE_REVISION" \
      org.opencontainers.image.base.name="rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
      bash \
      ca-certificates \
      cmake \
      g++ \
      gcc \
      git \
      jq \
      less \
      libibverbs-dev \
      libopenmpi-dev \
      libdwarf-dev \
      libzstd-dev \
      make \
      ninja-build \
      numactl \
      openmpi-bin \
      pciutils \
      pkg-config \
      pybind11-dev \
      python3-dev \
      python3-pip \
      python3-pybind11 \
      python3-venv \
      python3-pytest \
      rdma-core \
      vim-tiny \
    && rm -rf /var/lib/apt/lists/*

# Install Python helpers into the interpreter that `python3` resolves to in the
# ROCm/PyTorch image.  Installing the Ubuntu package can land outside that
# benchmark interpreter's default import path, so verify MORI's `prettytable`
# dependency is importable without any user-supplied PYTHONPATH shim.
RUN set -eux; \
    if ! python3 -m pip install --no-cache-dir prettytable; then \
      python3 -m pip install --no-cache-dir --break-system-packages prettytable; \
    fi; \
    python3 -c 'import prettytable'

# Runtime defaults for the benchmark container. ROCm/HSA tuning variables are
# intentionally left to the caller or wrapper command line.
ENV KIWI_EP_BENCH_BASE_TAG=rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0 \
    KIWI_REPO=/workspace/kiwi \
    OUTPUT_DIR=/outputs \
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    MPI_ALLOW_RUN_AS_ROOT=1 \
    MPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

RUN mkdir -p /workspace/kiwi /outputs /opt/ep-bench /opt/intellikit

# Install upstream IntelliKit into an isolated venv from a pinned ref.
# Upstream publishes tools as per-subdirectory Python packages rather than a
# repo-root "intellikit" CLI/metapackage.  Keep the source checkout and a
# provenance file in the image so benchmark logs can identify exactly what was
# installed.
RUN set -eux; \
    git clone "$INTELLIKIT_REPO" /opt/intellikit/source; \
    cd /opt/intellikit/source; \
    git checkout --detach "$INTELLIKIT_REF"; \
    intellikit_commit="$(git rev-parse HEAD)"; \
    python3 -m venv /opt/intellikit/venv; \
    /opt/intellikit/venv/bin/python -m pip install --upgrade pip setuptools wheel; \
    old_ifs="$IFS"; IFS=','; \
    for tool in $INTELLIKIT_TOOLS; do \
      IFS="$old_ifs"; \
      tool="$(printf '%s' "$tool" | tr -d '[:space:]')"; \
      [ -n "$tool" ] || continue; \
      [ -d "/opt/intellikit/source/$tool" ]; \
      /opt/intellikit/venv/bin/python -m pip install "/opt/intellikit/source/$tool"; \
      IFS=','; \
    done; \
    IFS="$old_ifs"; \
    : "${INTELLIKIT_SMOKE_COMMAND:=/opt/intellikit/venv/bin/metrix --version}"; \
    { \
      echo "INTELLIKIT_REPO=$INTELLIKIT_REPO"; \
      echo "INTELLIKIT_REF=$INTELLIKIT_REF"; \
      echo "INTELLIKIT_COMMIT=$intellikit_commit"; \
      echo "INTELLIKIT_TOOLS=$INTELLIKIT_TOOLS"; \
      echo "INTELLIKIT_ROOT=/opt/intellikit"; \
      echo "INTELLIKIT_SOURCE_DIR=/opt/intellikit/source"; \
      echo "INTELLIKIT_VENV=/opt/intellikit/venv"; \
      echo "INTELLIKIT_DEFAULT_CLI=metrix"; \
      echo "INTELLIKIT_SMOKE_COMMAND=$INTELLIKIT_SMOKE_COMMAND"; \
    } > /opt/ep-bench/intellikit-provenance.env

ENV INTELLIKIT_ROOT=/opt/intellikit \
    INTELLIKIT_SOURCE_DIR=/opt/intellikit/source \
    INTELLIKIT_VENV=/opt/intellikit/venv \
    INTELLIKIT_BIN=/opt/intellikit/venv/bin/metrix \
    PATH=/opt/ep-bench/scripts/ep_bench:${PATH}

RUN set -eux; \
    : "${INTELLIKIT_SMOKE_COMMAND:=/opt/intellikit/venv/bin/metrix --version}"; \
    if [ -n "$INTELLIKIT_SMOKE_COMMAND" ]; then \
      sh -c "$INTELLIKIT_SMOKE_COMMAND"; \
    fi

COPY scripts/ep_bench /opt/ep-bench/scripts/ep_bench
COPY docs/ep-benchmark-image.md /opt/ep-bench/docs/ep-benchmark-image.md

WORKDIR /workspace/kiwi
CMD ["bash"]
