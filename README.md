# transformers-perf

Performance tracking for [transformers](https://github.com/huggingface/transformers) on a dedicated, noise-tuned GPU
host: daily runs of main, side-by-side runs of PRs, multimodal processor benchmarks and hardware baselines.

- Results: [hf-internal-testing/transformers-perf](https://huggingface.co/datasets/hf-internal-testing/transformers-perf) ([columns](docs/dataset-card.md))
- Dashboard: [Transformers CI benchmarks](https://transformers-ci.huggingface.cool/benchmarks)

## What do we measure?

The focus is **inference performance and regressions across transformers commits** on fixed hardware. There are no
training forward/backward or optimizer-step benchmarks in the tracked daily suites.

| Workload | What is timed | What the results tell you |
|---|---|---|
| Text generation | A complete `generate()` call with fixed input and output token counts, after warm-up | End-to-end latency, time to first token (TTFT), inter-token latency (ITL), and output tokens per second |
| Multimodal processors (CPU) | Image/video/audio preprocessing on fixed, already decoded inputs | Per-call latency for resize, normalization, feature extraction, and processor logic; image cases compare PIL and torchvision backends |
| Multimodal encoders (GPU, bf16) | Vision/video/audio encoder forward, including the projector where applicable | Per-call latency and synchronization diagnostics; the language model or decoder is not run |
| Processor + encoder | Preprocessing, transfer to GPU, and encoder forward together | Combined input-processing latency, separate from text generation |

**Prefill versus decode.** TTFT is measured from the start of the generation call to the first emitted token; ITL
uses the intervals between subsequent tokens. These help distinguish first-token/prefill regressions from decoding
regressions, but they are application-level timings, not isolated CUDA kernel measurements of prefill and decode.
End-to-end latency includes the full generation call and a final device synchronization. Model loading and prompt
tokenization happen outside this timing. Throughput counts generated output tokens, not input tokens.

**Eager, compile, and CUDA graphs.** Scenarios select attention implementations (for example, eager attention,
SDPA, or Flash Attention 2) and whether to use `torch.compile`. Compiled regular generation uses a static KV cache;
uncompiled scenarios provide a comparison. CUDA graph behavior follows the selected compile/runtime configuration:
the daily suite is not a separate CUDA-graphs-on/off experiment. The harness supports explicit compile modes,
including `max-autotune-no-cudagraphs`, for targeted comparisons. First-call and warm-up timings are recorded
separately, along with compile/graph-break counters; compilation during the measurement phase is flagged.

**Speed versus memory.** The dashboard's generation charts focus on latency and throughput. Every run also records
memory and disk counters at phase boundaries (`startup_*` columns, read between timed regions, never sampled): the
model's tensors by dtype, what load read from disk versus the page cache and its host RSS (anonymous versus
file-backed), each phase's GPU working set above the weights, the allocator's peak reserve and retries, and what NVML
charges the process. These say how much, not what: `tperf memory profile` reruns one scenario with every GPU and host
allocation recorded and writes an HTML report of what held memory at the peak and which code allocated it. Its
timings mean nothing, so it is never part of a measured run. The same report has a *Python time* section: a third
run under [py-spy](https://github.com/benfred/py-spy), sampled from outside the process, splits load, first
generation and later generations by area of code (generation loop, model code, cache, torch, compile) and by line,
with milliseconds per generated token: where host-side overhead comes from, which latency numbers alone cannot show.

**Serving.** `tperf serve run` starts a real `transformers serve` on one GPU and plays multi-turn conversations
against its OpenAI API with streaming, as a chat client does: each turn resends the whole history, so the prompt, and
time to first token, grow turn after turn. Per turn it reports TTFT, decode speed, inter-token latency, stalls between
streamed chunks and server memory, plus TTFT's cost per 1k prompt tokens (re-reading the history); `tperf serve
summarize <base> <run>` compares two runs turn by turn. `--profile` attaches py-spy to the server for the measured
conversations: busy time per server thread (the event loop, the inference thread) by area of code, including the
serving layer's own. Scenarios: [serve-multiturn.jsonl](scenarios/serve-multiturn.jsonl) (SDPA, long context,
continuous batching with four conversations at once, compiled). `tperf serve bench` runs the same on any machine.

### Coverage and how to read a result

The checked-in [daily suites](src/tperf/daily/main.py) currently run Llama-2-7B (128 input / 64 output tokens) and
Gemma-3-1B (448 input / 128 output tokens), both at batch size 1, plus the
[multimodal scenarios](scenarios/processors.jsonl) for Qwen2.5-VL, LLaVA-OneVision, and Whisper. These are selected
regression workloads, not an exhaustive model, batch-size, or sequence-length sweep. Generation uses fixed-length
outputs with EOS stopping disabled. Processor inputs are synthetic and generated with a fixed seed; file I/O and
media decoding are excluded. Encoder-only inputs are already on the GPU; the combined stage includes transfer.

Daily runs track main over time. PR A/B runs compare base and head on two GPUs, swap GPUs between rounds, and report
changes with 95% confidence intervals. Locked clocks, CPU/NUMA pinning, warm-ups, repeated measurements, and
interference counters help separate regressions from noise. Compare matching workloads and hardware, and check
variation before interpreting small changes.

The tooling also supports targeted tensor-parallel, continuous-batching, and hardware-baseline experiments
(GEMM, host/device and peer-to-peer transfers, NCCL). Their availability does not mean every configuration runs daily.
Continuous-batching call latency includes manager creation and warm-up, so it should not be read as persistent-server
request latency: `tperf serve` measures that.

For more detail:

- [Harness guide](src/tperf/harness/README.md): workload policy, execution settings, commands, and regression comparisons.
- [Scenario files](scenarios/): exact attention/compile and processor configurations.
- [Dataset card](docs/dataset-card.md): metric columns, hardware/software metadata, raw result locations, and queries.
- [Processor benchmark](src/tperf/processors/bench.py): stage boundaries and input definitions.
- [Findings](docs/findings.md): results and investigations from individual experiments.

## Setup

```bash
pip install -e .                       # installs the tperf command
cp tperf.example.toml tperf.toml       # set [host] ssh (tperf.toml is not committed)
tperf host install                     # CUDA, torch, transformers under /opt/bench on the host
tperf host tune apply                  # quiet the host (status / revert)
tperf deploy                           # copy tperf (harness included) and scenarios; install the daily timer
```

## Usage

```bash
MODEL=google/gemma-3-1b-it SCENARIOS=scenarios/sliding-window.jsonl tperf ab 49542   # a PR against its base
MODEL=google/gemma-3-1b-it SCENARIOS=<one scenario>.jsonl tperf bisect <good> <bad>   # which commit of main moved it
tperf processors run                   # processors and encoders
tperf baselines run                    # GEMM and host<->device bandwidth
MODEL=meta-llama/Llama-2-7b-hf NCORES=6 tperf noise soak --hours 3   # how noisy is this host? (run on two, then
                                       #   tperf noise summarize <soak> <soak>)
MODEL=google/gemma-3-1b-it tperf memory profile 49542   # memory and Python time: results/<ts>-memory/profile/report.html
MODEL=Qwen/Qwen3-0.6B tperf serve run main      # multi-turn chat against transformers serve (scenarios/serve-*.jsonl)
tperf --help                           # every command
```

`tperf ab` runs base and head at the same time on two GPUs (one per NUMA node), swaps them every round, and reports
each metric's change with a 95% confidence interval. `tperf bisect` chains such A/Bs over the commits between two
daily points: it first re-runs the two daily commits against each other and stops if the change does not reproduce
(then it was not transformers), then bisects, then confirms the culprit against its parent. The daily run (03:00 UTC) benchmarks the latest main and
rebuilds the dashboard.

## Layout

- `src/tperf/`: the package, one subpackage per feature (`ab`, `daily`, `processors`, `tp`, `baselines`, ...)
- `src/tperf/harness/`: the generation benchmark (formerly transformers' `benchmark_v2`), the same copy for every
  commit (`tperf harness run`); its tests are in `tests/harness/`
- `scenarios/`: benchmark configs; `patches/`: transformers changes A/B-tested before their PR
- `docs/`: [dataset card](docs/dataset-card.md), [findings](docs/findings.md)

## Development

```bash
python -m pytest && ruff check . && ruff format --check .
```
