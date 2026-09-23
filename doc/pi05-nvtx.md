# PI0.5 BF16 NVTX profiling

The native BF16 PI0.5 path emits nested NVTX ranges through
`apxinf_model::profiling::trace`. The ranges are enabled by the existing CUDA
NVTX feature and become no-ops in a build without CUDA/NVTX support.

The hierarchy is:

- `pi05.infer`
  - `pi05.bf16.vision` and `vision.layer_NN`
  - `pi05.bf16.prefix` and `prefix.layer_NN`
  - `pi05.bf16.denoise.prepare_modulation.step_NN`
  - `pi05.bf16.denoise.step_NN` and `denoise.action.layer_NN`
- Nested `op.*` ranges identify normalization, QKV GEMM/RoPE, attention,
  output/down projections, GeGLU, residuals, and Euler updates.
- RGB input preprocessing is `pi05.bf16.preprocess`.

Graph execution has a separate stable boundary: `pi05.graph_input_update`,
`pi05.graph_noise_update`, and `pi05.graph_replay`. The detailed BF16 ranges
are executed while a graph is warmed up/captured; a steady-state replay is a
single graph launch and must be read through `pi05.graph_replay`.

## Build

Run from the ApxInf project with its own virtual environment active:

```bash
cd /home/mt/ly_ws/ApxInf
source .venv/bin/activate
python -c "import sys; print(sys.executable)"
cargo build --release --features cuda --example pi05_bench -p apxinf-model
mkdir -p devlocal/pi05-nvtx
```

## Detailed eager trace

`APXINF_PI05_PROFILE_EAGER=1` starts and stops the CUDA profiler around one
additional eager forward. `APXINF_PI05_EAGER_ONLY=1` exits after that path, so
graph construction does not enter the trace:

```bash
APXINF_PI05_PROFILE_EAGER=1 \
APXINF_PI05_EAGER_ONLY=1 \
nsys profile \
  --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  -o devlocal/pi05-nvtx/apxinf_pi05_bf16_orin_names \
  target/release/examples/pi05_bench random \
  --model-variant bf16 \
  --image-input nhwc \
  --views 2 --image-size 224 \
  --action-horizon 10 --num-flow-steps 10 --token-count 10 \
  --iterations 1
```

## Steady-state graph replay trace

The replay profile starts after model loading, eager validation, graph capture,
and warm-up. The canonical replay range is emitted by `CapturedGraph`, so the
same marker is used by the low-level benchmark and the Python policy path:

```bash
APXINF_PI05_PROFILE_REPLAY=1 \
nsys profile \
  --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --cuda-graph-trace=node \
  -o devlocal/pi05-nvtx/apxinf_pi05_bf16_orin_names_graph \
  target/release/examples/pi05_bench random \
  --model-variant bf16 \
  --image-input nhwc \
  --views 2 --image-size 224 \
  --action-horizon 10 --num-flow-steps 10 --token-count 10 \
  --iterations 10
```

Export and inspect the NVTX events with:

```bash
nsys export --type sqlite \
  --output=devlocal/pi05-nvtx/pi05_bf16_graph.sqlite \
  devlocal/pi05-nvtx/pi05_bf16_graph.nsys-rep

sqlite3 -header -column devlocal/pi05-nvtx/pi05_bf16_graph.sqlite \
  "SELECT text, COUNT(*) AS count FROM NVTX_EVENTS WHERE text LIKE 'pi05.%' GROUP BY text ORDER BY text;"
```

Use eager ranges for stage/layer/operator attribution and graph replay ranges
for production end-to-end latency. Do not treat capture-time detailed ranges as
per-replay counts.
