# KIWI/MORI EP benchmark runbook and image workflow

This page defines the machine-independent runbook plus reproducible
Docker/image and script contract for KIWI EP correctness, KIWI EP graph
benchmarks, official MORI EP compatibility/reference
runs, and IntelliKit profiling.  Keep benchmark reports tied to a resolved image
digest and the provenance emitted by `scripts/ep_bench/collect_env.sh`; do not
publish performance numbers against an unqualified `latest` image tag.

## Image layout

`docker/ep-bench/Dockerfile` builds a canonical runtime/build image containing:

- ROCm/PyTorch from the public base tag
  `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0`;
  this keeps the canonical image on the current ROCm 7.2 / PyTorch 2.10 line
  instead of the older ROCm 6.4.1 baseline;
- CMake, Ninja/Make, Open MPI, compiler, `libibverbs`/`rdma-core`, and Python
  helpers needed by KIWI EP and MORI EP smoke/benchmark runs;
- Python helper packages installed into the benchmark `python3` interpreter,
  including `prettytable` for the official MORI examples;
- upstream IntelliKit installed from a pinned `AMDResearch/intellikit` ref into
  `/opt/intellikit/venv`, with the source checkout retained at
  `/opt/intellikit/source`;
- the stable wrapper scripts copied under `/opt/ep-bench/scripts/ep_bench`;
- benchmark runtime environment kept explicit in wrapper commands; ROCm/HSA
  tuning variables are left to caller-provided environment or `EXTRA_DOCKER_ARGS`.

The Dockerfile intentionally uses a concrete public ROCm/PyTorch base tag rather
than requiring callers to pass a base image build argument.  For
benchmark-quality runs, record both the resolved base image identity and the
derived image identity in the report:

```bash
BASE='rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0'
docker pull "$BASE"
docker image inspect "$BASE" \
  --format 'base.id={{.Id}} base.repo_digests={{json .RepoDigests}}'

docker build -f docker/ep-bench/Dockerfile \
  --pull \
  --build-arg KIWI_IMAGE_REVISION="$(git rev-parse HEAD)" \
  -t kiwi-ep-bench:rocm7.2.4-pytorch2.10 .

docker image inspect kiwi-ep-bench:rocm7.2.4-pytorch2.10 \
  --format 'image.id={{.Id}} image.repo_digests={{json .RepoDigests}}'
```

If `image.repo_digests` is empty for a local-only tag, record the printed image
ID or push to an approved registry and record the resulting `RepoDigests` entry
before publishing performance comparisons.

The image also exposes pinned IntelliKit build args:

```bash
docker build -f docker/ep-bench/Dockerfile \
  --pull \
  --build-arg INTELLIKIT_REPO='https://github.com/AMDResearch/intellikit.git' \
  --build-arg INTELLIKIT_REF='3f45cd314d455b652a1246678511b40547fe521e' \
  --build-arg INTELLIKIT_TOOLS='accordo,kerncap,linex,metrix,nexus,rocm_mcp,uprof_mcp' \
  --build-arg INTELLIKIT_SMOKE_COMMAND='/opt/intellikit/venv/bin/metrix --version' \
  -t kiwi-ep-bench:rocm7.2.4-pytorch2.10 .
```

The default `INTELLIKIT_REF` is an exact upstream commit inspected when this
workflow was added.  Refresh it only to a reviewed tag or commit and record that
new value with benchmark results.  `INTELLIKIT_SMOKE_COMMAND` defaults to
`/opt/intellikit/venv/bin/metrix --version` because upstream IntelliKit currently publishes per-tool
commands rather than a repo-root `intellikit` CLI.

For local image development, use the same concrete public base unless the
Dockerfile itself is intentionally being refreshed.  If a cluster provides its
own ROCm/PyTorch/IB image, keep the same wrapper contract and set
`IMAGE=<site-image@sha256:...>` when launching.  It is fine for local dry-runs
to use tags, but logs that compare performance should include the resolved
image digest from `collect_env.sh` or `docker image inspect`.

## Canonical container runtime contract

Use `scripts/ep_bench/run_container.sh` when possible:

```bash
IMAGE=kiwi-ep-bench:rocm7.2.4-pytorch2.10 \
MORI_SOURCE_DIR=/path/to/mori-official \
scripts/ep_bench/run_container.sh -- \
  scripts/ep_bench/collect_env.sh --label container-smoke
```

The wrapper expands to the required Docker runtime shape:

```bash
docker run --rm \
  --network host --ipc host --shm-size 64g \
  --device /dev/kfd --device /dev/dri \
  --device /dev/infiniband/rdma_cm --device /dev/infiniband/uverbs0 ... \
  --group-add video --group-add render \
  --cap-add SYS_PTRACE --cap-add IPC_LOCK \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
  -v /path/to/kiwi:/workspace/kiwi:rw \
  -e KIWI_REPO=/workspace/kiwi -w /workspace/kiwi \
  -v /path/to/outputs:/outputs:rw \
  -v /path/to/mori-official:/mori:ro \
  kiwi-ep-bench:rocm7.2.4-pytorch2.10 bash
```

`run_container.sh` passes every existing `/dev/infiniband/*` node.  On systems
where MORI JIT needs verbs headers from the host, it also mounts
`/usr/include/infiniband` read-only when present.  Use `PASS_INFINIBAND=0` only
for local syntax/provenance checks that intentionally do not exercise RDMA.

## GPU Direct RDMA memory registration policy

EP benchmarks move HIP GPU memory through LCI/verbs.  The usable GPU Direct RDMA
registration path depends on the host kernel and driver stack:

- On Linux 6.3 and newer, the legacy `ibv_reg_mr` path through `peermem` is no
  longer a working GDRDMA path; DMA-BUF registration is the required mechanism.
- On tested Linux 6.8 systems, DMA-BUF is required and has not reproduced
  the earlier GPU memory-copy slowdown.
- On older kernels, DMA-BUF may or may not be the best-performing path.  If a
  node still shows DMA-BUF-related GPU memory-copy slowdown, use the site's LCI
  switch to disable DMA-BUF for benchmark comparisons, for example
  `LCI_ATTR_USE_DMABUF=0`.
- Do not carry a DMA-BUF setting between machines blindly.  Record `uname -r`,
  relevant `LCI_ATTR_*` variables, and the selected HCA/NIC mapping in every
  benchmark report.

Machine-specific runbooks should state the node kernel inventory and the local
recommended setting.

## Mounted paths and dependency provenance

The scripts avoid private hard-coded paths.  Use these environment variables to
make dependencies explicit:

| Variable | Container default | Purpose |
| --- | --- | --- |
| `KIWI_REPO` | `/workspace/kiwi` | KIWI checkout under test |
| `OUTPUT_DIR` | `/outputs/...` | Logs, matrices, profiler output, provenance |
| `KIWI_EP_BENCH_BASE_TAG` | `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0` | Public ROCm/PyTorch base tag used by the canonical image |
| `LCI_SOURCE_DIR` | unset | Optional local LCI source to report |
| `MORI_SOURCE_DIR` | `/mori` when mounted | Official MORI source/package tree |
| `INTELLIKIT_ROOT` | `/opt/intellikit` in the canonical image | IntelliKit install root |
| `INTELLIKIT_SOURCE_DIR` | `/opt/intellikit/source` in the canonical image | Pinned upstream IntelliKit source checkout |
| `INTELLIKIT_VENV` | `/opt/intellikit/venv` in the canonical image | IntelliKit Python virtual environment |
| `INTELLIKIT_BIN` | `/opt/intellikit/venv/bin/metrix` in the canonical image | Default IntelliKit profiling command |

Before any benchmark/profiling run, collect environment provenance:

```bash
scripts/ep_bench/collect_env.sh --output-dir /outputs/provenance --label before-run
```

The provenance file records source commits/status, dependency paths/SHAs, image
identity hints, ROCm/PyTorch/MPI versions, GPU/node details, HSA/MPI/runtime
environment variables, and IntelliKit provenance.  In the canonical image it
also includes `/opt/ep-bench/intellikit-provenance.env` with the upstream repo,
requested ref, resolved commit, installed tool list, source path, venv path, and
default CLI.  It probes `metrix --version` plus the available IntelliKit
per-tool command paths (`kerncap`, `accordo`, `*-mcp`, etc.).

## KIWI EP correctness

There are two correctness gates. Use the name that matches the validation scope.

### Focused smoke correctness

Build KIWI EP with tests, benchmarks, and the Python binding, then run the
focused smoke set:

```bash
scripts/ep_bench/run_kiwi_ep_smoke_correctness.sh \
  --build-dir /workspace/kiwi/build/ep-bench
```

This focused gate is intentionally small. Its CTest regex includes only:

- `test_dispatch`
- `test_combine`
- `test_combine_graph_replay`
- `test_ep_collective_primitives`
- `test_kiwi_ep_cudagraph_ws2` when Python support is enabled

Use it for quick smoke checks and while iterating locally. Do not describe this
as full EP correctness.

Useful overrides:

```bash
HIP_ARCHITECTURES=gfx942 BUILD_JOBS=32 \
CMAKE_EXTRA_ARGS='-DFETCHCONTENT_QUIET=OFF' \
scripts/ep_bench/run_kiwi_ep_smoke_correctness.sh
```

### Full EP correctness

The full EP correctness gate runs the focused smoke set and then the official
MORI compatibility CTest matrix:

```bash
MORI_SOURCE_DIR=/mori \
scripts/ep_bench/run_kiwi_ep_full_correctness.sh \
  --build-dir /workspace/kiwi/build/ep-bench
```

The official compatibility matrix is selected with:

```bash
ctest --test-dir /workspace/kiwi/build/ep-bench \
  --output-on-failure -R '^test_official_mori_compat_'
```

Run the full gate for changes to EP dispatch, combine, barriers, counters,
epochs, graph replay, routing, or public MORI-compatible behavior. If the full
gate is not run, say that clearly in the report and give the reason.

Validation reports should include the exact command or CTest regex, the test
inventory count, the number run, and pass/fail/skip counts.

Use `--dry-run` for command review without configuring/building.

## Official MORI compatibility downstream FP8 profiles

The official MORI compatibility wrapper has named sync profiles for downstream
vLLM/SGLang-style FP8 dispatch scales.  They keep MORI's default external input
buffer mode (`use_external_inp_buf=True`) and exercise the synchronous
`dispatch()`/`combine()` path with `combine(fused_expert_output, None, topk_ids)`:

```bash
scripts/ep_bench/run_mori_ep_compat.sh --dry-run \
  --profile vllm-fp8-token-sync \
  --profile vllm-fp8-block-sync \
  --profile sglang-fp8-block-sync
```

Dry-run output prints each profile expansion before the reproducible `mpirun`
command.  The profiles target `world_size=8`, `hidden_dim=7168`, `topk=8`,
`num_experts_per_rank=32`, FP8 dispatch, and float32 scales with either
`scale_dim=1` (token) or `scale_dim=56` (block).  If MORI, PyTorch, or enough
visible GPUs are unavailable during a real run, the underlying test preserves
the existing skip behavior.

## KIWI EP graph benchmark matrix

Run the MORI-style graph timing harness at the default single-node target world size:

```bash
EP_WORLD_SIZES='8' \
MAX_TOKENS_LIST='1 8 64' \
HIDDEN_DIMS='7168' \
DISPATCH_DTYPES='fp8' \
COMBINE_DTYPES='bf16' \
scripts/ep_bench/run_kiwi_ep_benchmark.sh \
  --build-dir /workspace/kiwi/build/ep-bench \
  --output-dir /outputs/kiwi-graph
```

The script writes one log per matrix point plus `matrix.tsv`.  Its default
world-size target is 8 ranks/GPUs, and its default payload mix is FP8 dispatch
and BF16 combine.  Set `DISPATCH_DTYPES` and
`COMBINE_DTYPES` to sweep payload types explicitly; there is intentionally no
single same-dtype shortcut because dispatch and combine are separate payload
formats.  The script passes explicit phase-specific scale layout flags
(`--dispatch-scale-dim/--dispatch-scale-dtype` and
`--combine-scale-dim/--combine-scale-dtype`): FP8 matrix points use
`hidden_dim/128` float32 scales, FP4 points use four E8M0 scale bytes, and
non-quantized BF16/FP32 points use no scale region.  The script derives
`--num-experts-per-rank` from `NUM_EXPERTS_TOTAL / EP_WORLD_SIZE` by default
(`NUM_EXPERTS_TOTAL=256`) to match the official MORI internode examples.  Set
`NUM_EXPERTS_PER_RANK` when intentionally comparing a different expert layout.


## MORI EP compatibility/reference entrypoint

When the official MORI source/package is available, run the compatibility gate
against the built KIWI Python binding:

```bash
MORI_SOURCE_DIR=/mori \
scripts/ep_bench/run_mori_ep_compat.sh \
  --build-dir /workspace/kiwi/build/ep-bench \
  --world-size 2 \
  --kernel-type InterNodeV1 \
  --api-mode sync \
  --dispatch-dtype float32
```

If `MORI_SOURCE_DIR` is unset or MORI is not importable, the script exits `77`
and prints `SKIP` so CI/CTest-style callers can distinguish a missing optional
reference dependency from a semantic mismatch.  Additional arguments for
`apps/ep/tests/test_official_mori_compat.py` can be supplied with
`EXTRA_MORI_COMPAT_ARGS`.

For side-by-side MORI benchmark runs, the older helper
`apps/ep/benchmarks/python/run_ep_mori_compare.sh` remains available.  Prefer the
Docker/image and provenance contract on this page when collecting new numbers.

## IntelliKit profiling entrypoint

The canonical image installs upstream IntelliKit from
`AMDResearch/intellikit` instead of expecting a site-local placeholder.  Upstream
IntelliKit is a collection of per-tool packages; it does not currently provide a
repo-root `intellikit` CLI.  The default profiling command is therefore
Metrix, checked at image build time with `metrix --version` and invoked by the
wrapper as:

```bash
metrix profile --time-only --num-replays 1 --output <out>/metrix.json '<workload>'
```

The stable entrypoint records provenance and then invokes the in-image Metrix
command around any workload:

```bash
scripts/ep_bench/run_intellikit_profile.sh -- \
  scripts/ep_bench/run_kiwi_ep_benchmark.sh \
    --build-dir /workspace/kiwi/build/ep-bench \
    --output-dir /outputs/profiled-kiwi-graph
```

Use overrides only when a site needs a different IntelliKit tool or wrapper
syntax:

```bash
INTELLIKIT_BIN=/opt/intellikit/venv/bin/metrix \
INTELLIKIT_ARGS='profile --profile memory --num-replays 3 --output /outputs/intellikit/memory.json' \
INTELLIKIT_WORKLOAD_MODE=string \
scripts/ep_bench/run_intellikit_profile.sh -- ./my_gpu_app --arg value
```

For profilers that expect `--` followed by workload argv, set
`INTELLIKIT_WORKLOAD_MODE=separator`.  If IntelliKit is not available, the
wrapper exits `77` with setup instructions.  Use `--allow-missing` for
documentation-only local smoke runs outside the canonical image.

## Minimal local validation

These commands do not require GPUs, Docker image builds, or external downloads:

```bash
bash -n scripts/ep_bench/*.sh
scripts/ep_bench/collect_env.sh --help
scripts/ep_bench/run_container.sh --dry-run -- echo container-smoke
scripts/ep_bench/run_kiwi_ep_smoke_correctness.sh --dry-run --no-python
scripts/ep_bench/run_kiwi_ep_full_correctness.sh --dry-run
scripts/ep_bench/run_kiwi_ep_benchmark.sh --dry-run
scripts/ep_bench/run_mori_ep_compat.sh --dry-run --mori /mori
scripts/ep_bench/run_intellikit_profile.sh --allow-missing -- echo profile-smoke
```

Before benchmarking on a GPU node, confirm the image digest, mounted
dependency SHAs, and HSA/MPI runtime settings in the provenance output.
