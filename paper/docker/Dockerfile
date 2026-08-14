FROM python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d

ARG UV_VERSION=0.5.29
ARG CLAUDE_CODE_VERSION=2.1.85
ARG DOCKER_CLI_VERSION=27.5.1

LABEL org.opencontainers.image.title="Karenina paper experiment controller" \
      org.opencontainers.image.description="Locked Karenina environment for paper regeneration" \
      org.opencontainers.image.version="python-3.11.14"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        nodejs \
        npm \
        tar \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN arch="$(uname -m)" \
    && case "$arch" in \
         x86_64) docker_arch=x86_64; docker_sha=4f798b3ee1e0140eab5bf30b0edc4e84f4cdb53255a429dc3bbae9524845d640 ;; \
         aarch64|arm64) docker_arch=aarch64; docker_sha=e6b53725a73763ab3f988c73f8772eaed429754c1a579db5ff11f21990fd1817 ;; \
         *) echo "unsupported architecture $arch" && exit 1 ;; \
       esac \
    && curl -Lsf \
        "https://download.docker.com/linux/static/stable/${docker_arch}/docker-${DOCKER_CLI_VERSION}.tgz" \
        -o /tmp/docker.tgz \
    && printf '%s  %s\n' "$docker_sha" /tmp/docker.tgz | sha256sum -c - \
    && tar -xzf /tmp/docker.tgz -C /tmp docker/docker \
    && install -m 0755 /tmp/docker/docker /usr/local/bin/docker \
    && rm -rf /tmp/docker /tmp/docker.tgz

RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}" \
    && npm install -g "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" \
    && npm cache clean --force

WORKDIR /opt/karenina

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

# uv.lock is the Python dependency authority. The extras cover manuscript
# statistics and the BixBench DeepAgents harness.
RUN uv sync --frozen --no-dev --extra paper-analysis --extra deep-agents

COPY paper ./paper

ENV PATH="/opt/karenina/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy

RUN python -c "import karenina, paper" \
    && claude --version \
    && docker --version

ENTRYPOINT ["/opt/karenina/.venv/bin/python", "-m"]
CMD ["paper.otp_response_characterization.run", "--help"]
