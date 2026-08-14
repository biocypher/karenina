# Paper Experiment Dependency Contract

This document defines the environment for archive-based reproduction and fresh
reruns of the Karenina paper experiments. The two workflows have different
data, service, credential, and container requirements.

The exact command for each experiment remains in
[`REGENERATION.md`](REGENERATION.md). The container commands below provide one
shared execution environment for all packages.

## Reproducibility layers

| Layer | Purpose | Required components |
|---|---|---|
| Controller | Runs Karenina, result loading, TaskEval, analyses, MCP launchers, and benchmark authoring | Python 3.11.14, Karenina source, `uv.lock`, Git, `uv`, and network access when a live service is used |
| Downloaded data | Supplies retained benchmarks, raw model outputs, optional stored judgments, and BixBench workspaces | `karenina-paper-experiments-data`, about 12 GB, beside the repository or selected with `KARENINA_PAPER_DATA` |
| Hosted models | Performs fresh answering, parsing, rubric judging, and drafting | The endpoints and credentials listed below; fresh outputs are stochastic and incur provider or compute cost |
| Open Targets MCP | Supplies Open Targets tools to MCP experiment arms | A local checkout selected by `KARENINA_PAPER_OTP_MCP_SOURCE`, or network access so `uvx` can obtain the public MCP package |
| BixBench sandboxes | Isolates code written and executed by Claude Code and DeepAgents | Docker plus `karenina-bixbench:latest` and `karenina-bixbench-claude:latest` |
| Standalone Claude Code | Generates adversarial alternatives outside Karenina | Claude Code 2.1.85, Claude authentication, and a Claude configuration containing an `open-targets` MCP server |
| Human review | Approves drafted benchmark templates and adversarial alternatives | A domain expert; freshly generated drafts are never promoted automatically |

## Python environment

The repository [`uv.lock`](../uv.lock) is the Python dependency authority.
Native setup uses:

```bash
uv sync --frozen --extra paper-analysis --extra deep-agents
```

The core project dependencies cover every packaged tabular analysis. The
`paper-analysis` extra adds Bambi, ArviZ, and PyMC for the broader manuscript
statistics environment, while `deep-agents` supplies the BixBench DeepAgents
harness. No separate paper requirements file is maintained, because a second
lock could drift from the library version that the experiments exercise.

The controller image pins Python 3.11.14 and uv 0.5.29, then installs directly
from `uv.lock` with `--frozen`. It also contains Git, the Docker client, Node,
npm, and Claude Code 2.1.85. This is the Claude Code version specified for
adversarial alternative generation. Karenina's Claude Agent SDK dependency is
controlled by `uv.lock`. The Docker 27.5.1 CLI is installed from its
official static archive with separate amd64 and arm64 checksums, avoiding the
older Bookworm client protocol.

## Workflow matrix

| Package | Archive-based reproduction | Fresh-rerun dependencies |
|---|---|---|
| `otp_response_characterization` | Data deposit and stored judgments | GPT-OSS 120B endpoint |
| `otp_model_comparison` | Data deposit and stored result exports | Four OpenAI-compatible model endpoints, Anthropic access, Open Targets MCP, and substantial model capacity |
| `otp_citation_audit` | Data deposit and stored judgments | GPT-OSS 120B plus Claude Agent SDK access with web search |
| `otp_adversarial_generation` | Data deposit and approved sample archive | Standalone Claude Code 2.1.85, Claude authentication, Open Targets MCP configuration, and later human approval |
| `otp_sycophancy_scenarios` | Data deposit and stored scenario and sidecar judgments | Qwen endpoint, GPT-OSS endpoint, Anthropic access, and Open Targets MCP |
| `bixbench_harness_comparison` | Data deposit, stored result exports, and stored burden judgments | Z.ai GLM access, Qwen and Anthropic access for the selected cells, Docker sandboxes, task-time package downloads, and large output storage |
| `otp_benchmark_curation` | No complete offline path because drafting is the experiment | Private question table, Anthropic access, and later expert review |

Archive-based reproduction makes no LLM calls when the command includes the
experiment's complete stored-reuse option. A `run.py` invocation without that
option calls the models required by the experiment.

## Credentials and endpoints

Put secrets in the repository root `.env` file or export them in the shell.
The file is ignored by Git and is loaded at runtime. A non-secret starting
point is available at [`paper.env.example`](docker/paper.env.example).

| Setting | Used by |
|---|---|
| `ANTHROPIC_API_KEY` | LangChain-based Claude answerers and parsers, benchmark drafting, and the BixBench Opus DeepAgents cell |
| `CLAUDE_CODE_OAUTH_TOKEN` or `ANTHROPIC_AUTH_TOKEN` | Subscription authentication for the Claude Agent SDK citation audit, standalone adversarial generation, and the BixBench Opus Claude Code cell |
| `ZAI_API_KEY` or `KARENINA_BIX_API_KEY` | GLM 5.1 BixBench answerers, parsers, and burden judges |
| `KARENINA_PAPER_GPT_OSS_URL` | Response characterization and citation screening |
| `KARENINA_PAPER_GPT_OSS_120B_URL` | Sycophancy sidecar judgments and model comparison GPT-OSS |
| `KARENINA_PAPER_QWEN3_5_122B_A10B_URL` | Sycophancy Qwen answerer and parser, plus model comparison Qwen 122B |
| `KARENINA_PAPER_QWEN3_5_A3B_URL` and `KARENINA_PAPER_QWEN3_6_A3B_URL` | Model comparison smaller Qwen models |
| `KARENINA_PAPER_VLLM_KEY` | Shared key for paper OpenAI-compatible endpoints; local vLLM normally uses `EMPTY` |
| `KARENINA_BIX_QWEN_ENDPOINT` and `KARENINA_BIX_QWEN_API_KEY` | BixBench Qwen route |
| `KARENINA_PAPER_OTP_MCP_SOURCE` | Optional local Open Targets MCP checkout |
| `KARENINA_PAPER_MCP_HOST` and `KARENINA_PAPER_MCP_BASE_PORT` | Managed Open Targets MCP binding |

Credential values are runtime-only inputs. Model configurations keep them in
secret-typed fields, run manifests do not serialize them, and the controller
build context excludes both `.env` and every `**/out` directory. Do not place
credentials in the source tree, data archive, dedicated Claude configuration
mount, or command-line arguments.

The committed endpoint defaults identify the internal network used for the
reported runs; they are not public services. External researchers must supply
compatible endpoints. When an endpoint runs on the Docker host, use
`host.docker.internal` rather than `localhost` in the container environment.

## Container architecture

The container files under [`paper/docker`](docker) define three images:

1. `karenina-paper:latest` is the controller and analysis image.
2. `karenina-bixbench:latest` is the shared scientific task sandbox for
   DeepAgents.
3. `karenina-bixbench-claude:latest` adds Claude Code 2.1.146 to the identical
   task sandbox used by the Claude Code harness.

The BixBench images use a fixed base package profile. The base uses uv 0.9.30,
the checksum-verified micromamba 2.6.2-1 release, the
exact scientific Python set in
[`python-requirements.lock`](docker/bixbench/python-requirements.lock), R
4.5.3, pinned direct tidyverse and Bioconductor packages, and build tools.
Individual agents may install additional task-specific packages into their
copied workspace. Those task-time environments are inherently task outputs
rather than controller dependencies, so fresh agent runs also require outbound
package access unless a local cache or mirror is provided.

The Python controller is locked by `uv.lock`. Both multi-platform base image
manifests are pinned by digest. Debian repository packages, npm transitive
packages, conda transitive packages, and task-installed packages are not
claimed to be bitwise archival. Direct package versions, the micromamba
release and architecture checksums, and both configured Claude Code versions
are fixed in the Dockerfiles. Publish built image digests alongside a final
artifact release if bitwise container replay is required.

## Container commands

Run the wrapper from the repository root. It finds the standard sibling or
child data deposit, mounts the repository and deposit at the same absolute
paths inside the controller, and assigns new output files to the invoking
user's numeric ID.

```bash
paper/docker/compose.sh build
```

To produce amd64 images on either an amd64 host or a Buildx-enabled arm64
host, select the platform for both build commands:

```bash
paper/docker/compose.sh build-amd64
paper/docker/compose.sh build-agent-images-amd64
```

These commands tag the selected architecture as `latest` in the local image
store. The scientific Dockerfile constructs its conda environment on the
native build platform while selecting packages for the target architecture.
This keeps the environment amd64 without running the large micromamba
transaction through CPU emulation.

A complete archive-based reproduction can then be run, for example:

```bash
paper/docker/compose.sh run \
  paper.otp_model_comparison.run \
  --reuse-stored-results
```

The same form runs fresh experiments when the reuse option is omitted. It does
not grant the container access to the Docker daemon.

Fresh BixBench needs its two task images and the restricted agentic launch
path:

```bash
paper/docker/compose.sh build-agent-images
paper/docker/compose.sh run-agentic \
  paper.bixbench_harness_comparison.smoke \
  --harness both
```

`run-agentic` mounts the Docker socket into the controller. That grants the
controller control over the host Docker daemon and should be used only for the
BixBench package. The same absolute repository paths on the host and in the
controller are required because the nested task containers bind copied
workspaces directly from the host daemon.

Standalone adversarial generation needs an explicit Claude Code configuration
mount. The directory must contain authentication and a connected MCP server
named `open-targets`:

```bash
export KARENINA_CLAUDE_CONFIG_DIR=/absolute/path/to/dedicated-claude-config
paper/docker/compose.sh run-claude \
  paper.otp_adversarial_generation.smoke
```

Use a dedicated configuration directory because the container can read and
update the mounted Claude state. Do not put that directory or any credential
inside the data deposit.

## Storage and platform notes

- The data deposit needs about 12 GB before decompression overhead and new
  outputs.
- A complete BixBench matrix can copy multi-gigabyte task workspaces for every
  condition and replicate. Reserve at least 150 GB or measure a smaller smoke
  first.
- The Docker sandbox build supports x86_64 and arm64. The Dockerfiles pin
  multi-platform base manifests, while each resulting image digest remains
  architecture-specific.
- Cross-building requires a Docker daemon with Buildx and binfmt support. An
  x86_64 cluster that provides Singularity but not Docker or BuildKit can run
  native BixBench containers, but it cannot directly build these Dockerfiles.
- Docker is not needed for archive-based reproduction or the other fresh paper
  workflows.
- Singularity and Apptainer remain supported by the BixBench Python CLI when
  invoked on an appropriate host, but they are not nested inside the Docker
  controller.
- Benchmark curation requires an authorized copy of the expert-authored source
  table, which is not included in the repository, data archive, or container
  build context. The finalized benchmark checkpoint used by downstream
  experiments is included in the data archive.

## Data archive integrity

The paper's data-availability statement identifies the versioned archive
record for `karenina-paper-experiments-data`. The archive contains
`MANIFEST.tsv` and `MANIFEST.sha256`; from its root, verify all retained files
with `shasum -a 256 -c MANIFEST.sha256`.
