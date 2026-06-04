# Optimization and Tuning

This guide aims to help users improve vLLM Ascend performance at the system level. It includes OS configuration, library optimization, deployment guide, and so on. Any feedback is welcome.

## Preparation

Run the container:

```{code-block} bash
   :substitutions:
# Update DEVICE according to your device (/dev/davinci[0-7])
export DEVICE=/dev/davinci0
# Update the cann base image
export IMAGE=m.daocloud.io/quay.io/ascend/cann:|cann_image_tag|
docker run --rm \
--name performance-test \
--shm-size=1g \
--device $DEVICE \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-it $IMAGE bash
```

Configure your environment:

```{code-block} bash
   :substitutions:
# Configure the mirror
echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy main restricted universe multiverse" > /etc/apt/sources.list && \
echo "deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy main restricted universe multiverse" >> /etc/apt/sources.list && \
echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
echo "deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
echo "deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list && \
echo "deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list

# Install os packages
apt update && apt install wget gcc g++ libnuma-dev git vim -y
```

Install vLLM and vLLM Ascend:

```{code-block} bash
   :substitutions:
# Install necessary dependencies
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install modelscope pandas datasets gevent sacrebleu rouge_score pybind11 pytest

# Configure this var to speed up model download
export VLLM_USE_MODELSCOPE=True
```

Please follow the [Installation Guide](https://docs.vllm.ai/projects/ascend/en/latest/installation.html) to make sure vLLM and vLLM Ascend are installed correctly.

:::{note}
Make sure your vLLM and vLLM Ascend are installed after your Python configuration is completed, because these packages will build binary files using python in current environment. If you install vLLM and vLLM Ascend before completing section 1.1, the binary files will not use the optimized python.
:::

## Optimizations

### 1. Compilation Optimization

#### 1.1. Install optimized `python` (OUT OF DATE)

Python supports **LTO** and **PGO** optimization starting from version `3.6` and above, which can be enabled at compile time. And we have offered optimized `python` packages directly to users for the sake of convenience. You can also reproduce the `python` build following this [tutorial](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0063.html) according to your specific scenarios.

```{code-block} bash
   :substitutions:
mkdir -p /workspace/tmp
cd /workspace/tmp

# Download prebuilt lib and packages
wget https://repo.oepkgs.net/ascend/pytorch/vllm/lib/libcrypto.so.1.1
wget https://repo.oepkgs.net/ascend/pytorch/vllm/lib/libomp.so
wget https://repo.oepkgs.net/ascend/pytorch/vllm/lib/libssl.so.1.1
wget https://repo.oepkgs.net/ascend/pytorch/vllm/python/py311_bisheng.tar.gz

# Configure python and pip

cp ./*.so* /usr/local/lib
tar -zxvf ./py311_bisheng.tar.gz -C /usr/local/
mv  /usr/local/py311_bisheng/  /usr/local/python
sed -i "1c#\!/usr/local/python/bin/python3.11" /usr/local/python/bin/pip3
sed -i "1c#\!/usr/local/python/bin/python3.11" /usr/local/python/bin/pip3.11
ln -sf  /usr/local/python/bin/python3  /usr/bin/python
ln -sf  /usr/local/python/bin/python3  /usr/bin/python3
ln -sf  /usr/local/python/bin/python3.11  /usr/bin/python3.11
ln -sf  /usr/local/python/bin/pip3  /usr/bin/pip3
ln -sf  /usr/local/python/bin/pip3  /usr/bin/pip

export PATH=/usr/bin:/usr/local/python/bin:$PATH
```

### 2. OS Optimization

#### 2.1. jemalloc

**jemalloc** is a memory allocator that improves performance for multi-threaded scenarios and can reduce memory fragmentation. jemalloc uses a local thread memory manager to allocate variables, which can avoid lock competition between threads and can hugely optimize performance.

```{code-block} bash
   :substitutions:
# Install jemalloc
sudo apt update
sudo apt install libjemalloc2

# Configure jemalloc
export LD_PRELOAD=/usr/lib/"$(uname -i)"-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
```

#### 2.2. Tcmalloc

**TCMalloc (Thread Caching Malloc)** is a universal memory allocator that improves overall performance while ensuring low latency by introducing a multi-level cache structure, reducing mutex contention and optimizing large object processing flow. Find more [details](https://www.hiascend.com/document/detail/zh/Pytorch/700/ptmoddevg/trainingmigrguide/performance_tuning_0068.html).

```{code-block} bash
   :substitutions:
# Install tcmalloc
sudo apt update
sudo apt install libgoogle-perftools4 libgoogle-perftools-dev

# Get the location of libtcmalloc.so*
find /usr -name libtcmalloc.so*

# Make the priority of tcmalloc higher
# The <path> is the location of libtcmalloc.so we get from the upper command
# Example: "$LD_PRELOAD:/usr/lib/aarch64-linux-gnu/libtcmalloc.so"
export LD_PRELOAD="$LD_PRELOAD:<path>"

# Verify your configuration
# The path of libtcmalloc.so will be contained in the result if your configuration is valid
ldd `which python`
```

### 3. `torch_npu` Optimization

Some performance tuning features in `torch_npu` are controlled by environment variables. Some features and their related environment variables are shown below.

Memory optimization:

```{code-block} bash
   :substitutions:
# Upper limit of memory block splitting allowed (MB): Setting this parameter can prevent large memory blocks from being split.
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:250"
```

or

```{code-block} bash
   :substitutions:
# When operators on the communication stream have dependencies, they all need to be ended before being released for reuse. The logic of multi-stream reuse is to release the memory on the communication stream in advance so that the computing stream can be reused.
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
```

Scheduling optimization:

```{code-block} bash
   :substitutions:
# Optimize operator delivery queue. This will affect the memory peak value, and may degrade if the memory is tight.
export TASK_QUEUE_ENABLE=2

# This will greatly improve the CPU bottleneck model and ensure the same performance for the NPU bottleneck model.
export CPU_AFFINITY_CONF=1
```

### 4. CANN Optimization

#### 4.1. HCCL Optimization

There are some performance tuning features in HCCL, which are controlled by environment variables.

You can configure HCCL to use "AIV" mode to optimize performance by setting the environment variable shown below. In "AIV" mode, the communication is scheduled by AI vector core directly with RoCE, instead of being scheduled by AI CPU.

```{code-block} bash
   :substitutions:
export HCCL_OP_EXPANSION_MODE="AIV"
```

Plus, there are more features for performance optimization in specific scenarios, which are shown below.

- `HCCL_INTRA_ROCE_ENABLE`: Use RDMA link instead of SDMA link between two 8Ps as the mesh interconnect link. Find more [details](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0044.html).
- `HCCL_RDMA_TC`: Use this var to configure traffic class of RDMA NIC. Find more [details](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0045.html).
- `HCCL_RDMA_SL`: Use this var to configure service level of RDMA NIC. Find more [details](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0046.html).
- `HCCL_BUFFSIZE`: Use this var to control the cache size for sharing data between two NPUs. Find more [details](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0047.html).

### 5. OS Optimization

This section describes operating system–level optimizations applied on the host machine (bare metal or Kubernetes node) to improve performance stability, latency, and throughput for inference workloads.

:::{note}
These settings must be applied on the host OS and with root privileges. Not inside containers.
:::

#### 5.1

Set CPU Frequency Governor to `performance`

```shell
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

Purpose

- Forces all CPU cores to run under the `performance` governor
- Disables dynamic frequency scaling (e.g., `ondemand`, `powersave`)

Benefits

- Keeps CPU cores at maximum frequency
- Reduces latency jitter
- Improves predictability for inference workloads

#### 5.2 Disable Swap Usage

```shell
sysctl -w vm.swappiness=0
```

Purpose

- Minimizes the kernel’s tendency to swap memory pages to disk

Benefits

- Prevents severe latency spikes caused by swapping
- Improves stability for large in-memory models

Notes

- For inference workloads, swap can introduce second-level latency
- Recommended values are `0` or `1`

#### 5.3 Disable Automatic NUMA Balancing

```shell
sysctl -w kernel.numa_balancing=0
```

Purpose

- Disables the kernel’s automatic NUMA page migration mechanism

Benefits

- Prevents background memory page migrations
- Reduces unpredictable memory access latency
- Improves performance stability on NUMA systems

Recommended For

- Multi-socket servers
- Ascend / NPU deployments with explicit NUMA binding
- Systems with manually managed CPU and memory affinity

#### 5.4 Increase Scheduler Migration Cost

```shell
sysctl -w kernel.sched_migration_cost_ns=50000
```

Purpose

- Increases the cost for the scheduler to migrate tasks between CPU cores

Benefits

- Reduces frequent thread migration
- Improves CPU cache locality
- Lowers latency jitter for inference workloads
  
Parameter Details

- Unit: nanoseconds (ns)
- Typical recommended range: 50000–100000
- Higher values encourage threads to stay on the same CPU core

## 6. vLLM Ascend Software Tuning

Sections 1–5 above focus on the host OS, CANN/HCCL, and `torch_npu`. This section covers **vLLM Ascend plugin behavior**: how to choose optimizations by bottleneck, what each lever actually changes, and when not to enable a feature.

For per-model recipes and measured values, see model deployment tutorials (for example, [Qwen3 Dense](../../tutorials/models/Qwen3-Dense.md)). For feature details and compatibility matrices, see the [Feature Guide](../../user_guide/feature_guide/index.md) and [Additional Configuration](../../user_guide/configuration/additional_config.md).

### 6.1 Tuning workflow

Use the same workflow for every change:

1. **Define the scenario and metrics** — throughput (tokens/s), TTFT, TPOT, max concurrency, memory headroom, and whether the workload is prefill-heavy or decode-heavy.
2. **Classify the primary bottleneck** — use {ref}`Section 6.2 <ascend-tuning-bottleneck>` (one dominant category is enough to start).
3. **Change one lever at a time** — keep topology, model weights, and unrelated flags fixed; compare A/B on the same benchmark.
4. **Re-measure and record constraints** — note mutual exclusions (for example, `enforce_eager` vs graph mode, FlashComm1 vs small batches, batch invariant vs some Ascend options).

Prefer `--additional-config` over deprecated `VLLM_ASCEND_*` environment variables; see the [migration table](../../user_guide/configuration/additional_config.md#migration-guide).

(ascend-tuning-bottleneck)=
### 6.2 Classify the primary bottleneck

Pick the category that best matches what you observe. Only the first matching row is needed to choose a starting direction.

| If you observe… | Primary category | Start here |
|-----------------|------------------|------------|
| Low QPS at modest batch size; host/scheduler hot in profiling; decode latency jitter | **Runtime overhead** | {ref}`§6.3.1 <ascend-tuning-runtime>` |
| TP scaling sublinear; large prefill batches communication-bound; long context OOM or poor TTFT | **Parallelism and communication** | {ref}`§6.3.2 <ascend-tuning-parallelism>` |
| High MTE / memory-bound linear layers; layout or quant path clearly slow | **Compute and memory bandwidth** | {ref}`§6.3.3 <ascend-tuning-bandwidth>` |
| OOM when raising concurrency; too many chunks; repeated prefixes still fully recomputed | **Capacity and batching** | {ref}`§6.3.4 <ascend-tuning-capacity>` |

### 6.3 Levers by bottleneck category

(ascend-tuning-runtime)=
#### 6.3.1 Runtime overhead (graph and scheduling)

**What this fixes:** CPU-side scheduling, per-step launch overhead, and mismatch between actual batch shapes and captured graphs (padding waste).

| Lever | What it does | When to try | Caveats |
|-------|----------------|-------------|---------|
| **Graph mode (ACLGraph, optional Npugraph_ex)** | Captures and replays the execution graph to cut dispatch cost; Npugraph_ex optimizes the FX graph **before** capture, it does not replace ACLGraph | Stable decode workloads; not `enforce_eager` | Context-parallel `FULL` mode has limited support; see [Graph Mode Guide](../../user_guide/feature_guide/graph_mode.md) |
| **`cudagraph_capture_sizes`** | Lists token sizes for which graphs are captured; other sizes are padded up to the next entry | After you fix target concurrency | With FlashComm1, sizes should be **multiples of TP**; misaligned values are filtered |
| **`--async-scheduling`** | Overlaps scheduling with execution to reduce CPU bottlenecks | Large models, high concurrency | Check compatibility with your model and features (for example, speculative decoding, batch invariant) |

**Baseline (usually not tuned manually):** compile-time fusions such as Rope reuse, AddRMSNormQuant, and related passes are enabled on typical paths. Treat graph mode and batch knobs as the first software levers, not these fusions.

**Example — graph mode (CLI):**

```bash
vllm serve Qwen/Qwen3-8B \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

(ascend-tuning-parallelism)=
#### 6.3.2 Parallelism and communication

**What this fixes:** Collective volume and count under tensor parallelism, sequence-dimension partitioning for long context, and weight sharding when full matrices do not fit.

Three mechanisms are often confused; they address **different dimensions**:

| Name | What it changes | Typical use | Notes |
|------|-----------------|-------------|-------|
| **FlashComm1 (FC1)** | Inside a **TP group**, replaces AllReduce + RMSNorm (+ quant) with **ReduceScatter → local norm → AllGather** via **custom ops**; activates only when scheduled **token count exceeds a threshold** | Non–vision-language Dense/MoE, **large** batches, often with quantization | Complementary to SP (Pass), not a substitute; may enable Matmul–ReduceScatter style fusions when FC1 is on; configure via `enable_flashcomm1` |
| **Sequence parallelism (SP, Pass)** | **Compile-time** graph rewrite for the same RS/AG pattern; requires **graph mode** | **Vision-language** models (for example, Qwen3-VL) | Quantized SP is still limited; see [Sequence Parallelism](../../user_guide/feature_guide/sequence_parallelism.md) |
| **Context parallel (PCP / DCP)** | Splits work along the **sequence length** dimension | Very long context (TTFT or KV pressure) | Orthogonal to FC1; use `--prefill-context-parallel-size` / `--decode-context-parallel-size`; see [Context Parallel](../../user_guide/feature_guide/context_parallel.md) |

**Related levers (one line each):**

- **`enable_matmul_allreduce`** — custom-op fusion of **Matmul + AllReduce** under TP (default off). This is **not** the same as the **MatmulAllReduceAddRMSNorm** inductor pass used on some Npugraph_ex paths.
- **FlashComm2 (`flashcomm2_parallel_size > 0`)** — **output projection (o_proj) sharding** and communication domains to cut memory and comm pressure; often used with [layer sharding](../../user_guide/feature_guide/layer_sharding.md), not “FC1 v2.”
- **Fine-grained TP / layer sharding** — adjust parallelism per module or layer (communication balance, PD prefill memory).

**Tuning order (parallelism):**

1. Fix TP / EP / PP / PD topology first.
2. If the issue is **long context**, evaluate PCP/DCP before FC1.
3. For **VL** models, consider SP (Pass) in graph mode; for **non-VL** large TP batches, evaluate FC1 only after confirming token counts exceed the FC1 threshold.
4. Align `cudagraph_capture_sizes` with TP when FC1 is enabled.

**Example — FlashComm1:**

```bash
vllm serve Qwen/Qwen3-32B \
  --tensor-parallel-size 4 \
  --additional-config '{"enable_flashcomm1": true}'
```

(ascend-tuning-bandwidth)=
#### 6.3.3 Compute and memory bandwidth

**What this fixes:** Weight movement (MTE), operator layout, and MoE-specific compute paths—not framework scheduling or batch capacity.

| Lever | What it does | When to try | Caveats |
|-------|----------------|-------------|---------|
| **Weight prefetch** | Uses **vector pipelines** (RMSNorm, SwiGLU, etc.) to hide **weight prefetch (CMO)** into L2 | Throughput-oriented Dense/MoE; after graph path is stable | Tuned via `weight_prefetch_config` and `prefetch_ratio`; **low-latency** scenarios often should leave it off; MLP `down` prefetch requires SP; single prefetch block cap ~18 MB — see [Weight Prefetch](../../user_guide/feature_guide/weight_prefetch.md) |
| **`weight_nz_mode`** | Controls **FRACTAL_NZ** weight layout for Cube efficiency | Quant or layout-sensitive models | Layout switch, not a communication optimization |
| **Quantization (W8A8, etc.)** | Lowers compute and collective bit width | When accuracy and model support allow | Combine with FC1/MLAPO per model guide |

**MoE-only (evaluate only for MoE deployments):** `enable_mlapo` (DeepSeek W8A8, trades memory for speed), `enable_fused_mc2` (fused dispatch/combine; PD and quant constraints), EPLB — see [Large-scale EP](../../user_guide/feature_guide/large_scale_ep.md) and model tutorials.

(ascend-tuning-capacity)=
#### 6.3.4 Capacity and batching

**What this fixes:** How many tokens and sequences fit per step, KV reuse, and splitting prefill vs decode across nodes.

| Lever | Role |
|-------|------|
| **`max-num-batched-tokens` / `max-num-seqs`** | Main trade-off between throughput, chunking, and OOM risk |
| **Chunked prefill / prefix caching** | Long prefill chunking and prefix KV reuse (vLLM core features; verify Ascend support for your model) |
| **PD / EPD disaggregation** | Separate prefill and decode for TTFT vs TPOT tuning — [disaggregated prefill](../Design_Documents/disaggregated_prefill.md) |
| **KV offload / KV pool** | Extend effective KV capacity — [KV cache offload](../../user_guide/feature_guide/kv_cache_cpu_offload.md), [KV pool](../../user_guide/feature_guide/kv_pool.md) |
| **Dynamic batch / balance scheduling** | Load or SLO-driven batching — `SLO_limits_for_dynamic_batch`, `enable_balance_scheduling` |

Tune batch limits to a **stable, non-OOM** point before chasing graph capture sizes or FC1.

### 6.4 Suggested tuning sequence (generic)

This is a **decision order**, not a checklist of every feature:

```text
Topology (TP/EP/PP, PD or not)
  → Capacity (max-num-batched-tokens, max-num-seqs, chunk/APC as needed)
    → Runtime (graph mode + cudagraph_capture_sizes [+ async-scheduling if CPU-bound])
      → Parallelism (long context: PCP/DCP; VL: SP Pass; else large-batch TP: FC1)
        → Bandwidth (quant, weight_nz_mode, weight prefetch only if throughput goal and profiled)
```

**Example — FC1 after stable graph and batch (illustrative):**

```bash
vllm serve Qwen/Qwen3-32B \
  --tensor-parallel-size 4 \
  --max-num-seqs 64 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [64]}' \
  --additional-config '{"enable_flashcomm1": true}'
```

Adjust `cudagraph_capture_sizes` to your real concurrency and keep values as multiples of TP when FC1 is on.

### 6.5 Further reading

| Topic | Document |
|-------|----------|
| All `additional_config` keys | [Additional Configuration](../../user_guide/configuration/additional_config.md) |
| Graph modes on Ascend | [Graph Mode Guide](../../user_guide/feature_guide/graph_mode.md) |
| FC1 vs SP | [Sequence Parallelism](../../user_guide/feature_guide/sequence_parallelism.md) |
| Long context | [Context Parallel](../../user_guide/feature_guide/context_parallel.md) |
| Benchmarking | [Performance Benchmark](performance_benchmark.md) |
| Accuracy / numerics debug | [msProbe Guide](msprobe_guide.md) |
