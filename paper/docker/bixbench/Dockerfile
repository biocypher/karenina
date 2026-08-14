ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.9.30-python3.11-bookworm-slim@sha256:4f5d923c9dcea037f57bda425dd209f3ec643da2f0b74227f68d09dab0b3bb36

# Resolve and link the scientific environment on the native build platform.
# This avoids running micromamba through QEMU when cross-building while
# --platform still selects packages for the final target architecture.
FROM --platform=$BUILDPLATFORM ${UV_IMAGE} AS r-environment-builder

ARG MICROMAMBA_RELEASE=2.6.2-1
ARG TARGETARCH

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && arch="$(uname -m)" \
    && case "$arch" in \
         x86_64) mm=linux-64; mm_sha=e9683b483df06dbd3fdd8a37f1b6826d7e5caf4e85bf15a0af4fbad3d4ad1a58 ;; \
         aarch64|arm64) mm=linux-aarch64; mm_sha=8e3301c5084f673399f5c52395cfba67b9dbe64dea39c7288e102646ef7c3769 ;; \
         *) echo "unsupported build architecture $arch" && exit 1 ;; \
       esac \
    && curl -Lsf \
        "https://github.com/mamba-org/micromamba-releases/releases/download/${MICROMAMBA_RELEASE}/micromamba-${mm}" \
        -o /usr/local/bin/micromamba \
    && printf '%s  %s\n' "$mm_sha" /usr/local/bin/micromamba | sha256sum -c - \
    && chmod +x /usr/local/bin/micromamba \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN case "$TARGETARCH" in \
      amd64) target_platform=linux-64 ;; \
      arm64) target_platform=linux-aarch64 ;; \
      *) echo "unsupported target architecture $TARGETARCH" && exit 1 ;; \
    esac \
    && MAMBA_ROOT_PREFIX=/opt/_mmroot XDG_CACHE_HOME=/opt/_mmxdg \
        micromamba create -y --copy --platform "$target_platform" \
        -p /opt/renv -c conda-forge -c bioconda \
        r-base=4.5.3 r-data.table=1.18.4 r-tidyverse=2.0.0 \
        r-readxl=1.5.0 r-openxlsx=4.2.8.1 r-biocmanager=1.30.27 \
        bioconductor-biostrings=2.78.0 \
        bioconductor-genomicranges=1.62.1 \
        bioconductor-summarizedexperiment=1.40.0 \
        bioconductor-s4vectors=0.48.0 bioconductor-iranges=2.44.0 \
        cairo expat fontconfig freetype libpng pixman pkg-config \
        xorg-kbproto xorg-libx11 xorg-renderproto xorg-xextproto xorg-xproto zlib \
        compilers make \
    && micromamba clean -a -y \
    && rm -rf /opt/_mmroot /opt/_mmxdg

FROM ${UV_IMAGE}

ARG MICROMAMBA_RELEASE=2.6.2-1

LABEL org.opencontainers.image.title="Karenina BixBench scientific sandbox" \
      org.opencontainers.image.description="Shared Python, R, and bioinformatics environment for BixBench agents"

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl \
        libcairo2-dev libexpat1-dev libfontconfig1-dev libfreetype6-dev \
        libpng-dev libpixman-1-dev libx11-dev libxext-dev pkg-config \
        x11proto-dev zlib1g-dev \
    && arch="$(uname -m)" \
    && case "$arch" in \
         x86_64) mm=linux-64; mm_sha=e9683b483df06dbd3fdd8a37f1b6826d7e5caf4e85bf15a0af4fbad3d4ad1a58 ;; \
         aarch64|arm64) mm=linux-aarch64; mm_sha=8e3301c5084f673399f5c52395cfba67b9dbe64dea39c7288e102646ef7c3769 ;; \
         *) echo "unsupported architecture $arch" && exit 1 ;; \
       esac \
    && curl -Lsf \
        "https://github.com/mamba-org/micromamba-releases/releases/download/${MICROMAMBA_RELEASE}/micromamba-${mm}" \
        -o /usr/local/bin/micromamba \
    && printf '%s  %s\n' "$mm_sha" /usr/local/bin/micromamba | sha256sum -c - \
    && chmod +x /usr/local/bin/micromamba \
    && printf 'channels:\n  - conda-forge\n  - bioconda\nchannel_priority: strict\n' > /etc/condarc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY paper/docker/bixbench/python-requirements.lock /opt/karenina/python-requirements.lock

RUN uv pip install --system -r /opt/karenina/python-requirements.lock

COPY --from=r-environment-builder /opt/renv /opt/renv

# Login shells otherwise reset PATH. Each mounted task workspace receives a
# private scratch area so concurrent tasks do not share package caches or R
# library locks.
RUN { echo ''; \
      echo '# Keep the baked R environment on PATH in login shells.'; \
      echo 'case ":$PATH:" in'; \
      echo '    *:/opt/renv/bin:*) ;;'; \
      echo '    *) PATH="$PATH:/opt/renv/bin" ;;'; \
      echo 'esac'; \
      echo 'export PATH'; \
      echo '_kar_task="/tmp/task-$(stat -c %i /workspace 2>/dev/null || echo 0)"'; \
      echo 'mkdir -p "$_kar_task/rlibs" "$_kar_task/uv" "$_kar_task/mamba" "$_kar_task/xdg" "$_kar_task/home" 2>/dev/null || true'; \
      echo 'export R_LIBS_USER="$_kar_task/rlibs"'; \
      echo 'export UV_CACHE_DIR="$_kar_task/uv"'; \
      echo 'export MAMBA_ROOT_PREFIX="$_kar_task/mamba"'; \
      echo 'export XDG_CACHE_HOME="$_kar_task/xdg"'; \
      echo 'export HOME="$_kar_task/home"'; \
      echo 'export PKG_CONFIG_PATH="/opt/renv/lib/pkgconfig:/opt/renv/share/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"'; \
    } >> /etc/profile

ENV UV_LINK_MODE=copy \
    UV_CACHE_DIR=/tmp/uv-cache \
    CONDARC=/etc/condarc \
    MAMBA_ROOT_PREFIX=/tmp/mamba \
    XDG_CACHE_HOME=/tmp/.cache \
    HOME=/tmp \
    PATH=/workspace/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/renv/bin \
    R_LIBS_USER=/tmp/rlibs \
    PKG_CONFIG_PATH=/opt/renv/lib/pkgconfig:/opt/renv/share/pkgconfig
