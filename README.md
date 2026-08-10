# dsv4-streaming

SSD expert streaming for DeepSeek-V4-Flash (284B sparse MoE) on a 64 GB M5 Pro MacBook.

DeepSeek-V4-Flash is 284B total / ~13.8B active parameters; at mixed 4/8-bit quantization the checkpoint is 165 GB, and no published quant fits a 64 GB Mac (smallest GGUF is 83 GB, smallest MLX 92.7 GB). This project keeps the ~9.4 GB of non-expert weights resident and streams the 155.8 GB of routed-expert weights from SSD on demand, with an LRU cache in unified memory — the approach [drumih/turbo-fieldfare](https://github.com/drumih/turbo-fieldfare) demonstrated for a 26B Gemma, scaled up 10x. It is built on the [PipeNetwork/deepseek-v4-mlx](https://github.com/PipeNetwork/deepseek-v4-mlx) from-scratch MLX port, because no mainstream runtime supports the `deepseek_v4` architecture (mlx-lm PR #1189 still open at time of writing). **Headline result:** byte-correct greedy decoding at 2.22–2.43 tok/s warm with a 0.8 s model load and 41.7 GB peak memory — versus 1.56–2.0 tok/s and a >10 minute cold start for the only llama.cpp configuration that works on this hardware at all (CPU-only). **Those figures are a repetitive-prompt benchmark, and a later diverse-prompt run shows they are the best case rather than the typical one:** a realistic prompt touches ~4,050 distinct experts (~57 GB), which is larger than any cache that fits on 64 GB hardware, so the cache churns and the same generation runs at roughly 0.87 tok/s. See *Diverse-workload results* below. This is a learning-project write-up: the numbers are real but narrow (batch-1 decode, two diverse prompts measured), and three failed optimization experiments are documented below with causes of death, because they say more about MLX and Metal than the success does.

## Quickstart

Requires the pipenetwork mixed 4/8-bit checkpoint downloaded locally, ~160 GB free SSD, and macOS with MLX.

```bash
# 1. Repack the checkpoint into resident weights + per-layer fixed-stride expert files.
#    Resumable; safe to run while the snapshot is still downloading (--layers to restrict).
python streaming/repack.py --snapshot /path/to/hf-snapshot --out repacked/

# 2. Run streaming inference.
python streaming/run_streaming.py --repacked repacked/ --cache-gb 32 \
    --prompt "The capital of France is" --max-new 48
```

Notes:

- Cache budgets above ~40 GB need a raised wired limit: `sudo sysctl iogpu.wired_limit_mb=57344`. This resets on reboot. Raising the limit is not the same as it being safe to use: a cache large enough to push the machine into swap costs 17x throughput (see *The swap cliff*), and `--cache-gb 32` above is a value that has been measured, not a recommended optimum — a sweep is in progress.
- Terminal output can be display-compressed; verify exact output from the written log files.
- `streaming/test_roundtrip.py` round-trips a synthetic quantized checkpoint through the repacker and store, bit-exact.

## Results

All measurements: M5 Pro, 64 GB, batch-1 greedy decode, 48 new tokens unless noted. Raw logs in `logs/`.

Unless a subsection says otherwise, the prompt is the 5-token `"The capital of France is"`, which generates `"The capital of X is Y"` indefinitely. That repetition is what makes its expert working set small, so the tok/s and hit-rate numbers below are best-case; *Diverse-workload results* gives the realistic figures.

### Baseline comparison

llama.cpp b10280 with the 83 GB UD-IQ1_S GGUF was the only runnable prior art on this machine. Every GPU path fails:

| Config | Result |
|---|---|
| llama.cpp, full GPU offload | Fails: Metal residency ceiling ~60 GB (`res=-3`) |
| llama.cpp, `-ot exps=CPU` | SIGBUS in `ggml_compute_forward_mul_mat_id` (KERN_PROTECTION_FAILURE; same class as stale-closed llama.cpp#24413, reproduced here on a second architecture) |
| llama.cpp, partial offload | Prefill works; generation fails `res=-3` somewhere between `-ngl 1` (works) and `-ngl 22` (fails) |
| **llama.cpp, CPU-only (`-ngl 0`)** | **Works: 1.56–2.0 tok/s decode, 0.48–1.1 tok/s prefill, >10 min cold start** |
| **This repo, v0 streaming (mixed 4/8-bit)** | **Works: 2.22 tok/s cold-cache, 2.43 tok/s warm, 0.8 s load, peak 41.7 GB (32 GB cache) — repetitive prompt; ~0.87 tok/s on a diverse prompt** |

Note the quantization levels differ (1-bit GGUF vs mixed 4/8-bit here), so this compares deployability more than it isolates speed — the streaming path runs a much higher-quality quant in two-thirds of the memory.

Two caveats on the speed half of that comparison. Both rows were measured on the repetitive prompt. The streaming path's throughput depends on how much of its expert working set fits in cache and so degrades to ~0.87 tok/s on a realistic prompt; llama.cpp CPU-only holds every weight resident and has no such dependence, so it has not been re-measured on diverse prompts but would not be expected to move. On prompts that are not degenerate, the streaming path is most likely *slower* than the CPU-only baseline, and its advantage is the 0.8 s load, the higher-quality quant, and fitting at all.

### Hit-rate vs cache size (repetitive prompt)

`ExpertStore` is a global LRU over 14.16 MB expert blobs, explicit `pread` (no mmap). 48-token run, ~13.4k expert visits:

| Cache budget | Hit rate | Notes |
|---|---|---|
| 8 GB | 47% | |
| 32 GB | 80% (86% at 256 tokens) | Working set exceeds budget; warm repeat is *slower* than run 1 (LRU thrash) |
| 44 GB | 80% run 1, **zero new misses run 2** | 48-token working set is 37.4 GB and fits entirely; warm pass 2.43 tok/s |

The 37.4 GB in that last row is a property of this prompt, not of the model: 2,641 distinct experts over 53 tokens. A diverse prompt touches ~4,050 over 62 tokens and does not fit in any cache this machine can hold — see below. Read the table as a cache-size curve, not as a claim that the working set fits.

SSD micro-benchmark (real layer-0 blobs, `F_NOCACHE` so cold means cold): 1.07 ms per 14.16 MB blob = 13.3 GB/s. SSD traffic is ~1.9 GB/token cold and falls as the cache warms. Per-layer working sets are heavily skewed — 58 to 140 distinct experts per layer over a 48-token run (`logs/hitrate_last_run.json`) — which is why global LRU beats any uniform per-layer allocation (see experiment 3).

### Diverse-workload results (partial: 2 of 8 prompts)

`streaming/multi_prompt_eval.py` runs 8 prompts from different domains through one persistent-cache process and records per-prompt *deltas* in hits, misses and bytes read. The run was stopped after two prompts finished, so there is no `diverse_eval.json` summary; `logs/diverse_run.txt` is the raw output of what completed. Two prompts is not a workload characterization. It is enough to overturn the "working set fits in cache" reading of the table above, because the two agree closely with each other and both touch ~50% more distinct experts than the repetitive prompt does.

48 new tokens each. The two diverse prompts ran back to back in one process with a 40 GB cache budget, saturated (`cache 40.0 GB` on both lines); the repetitive row is the 44 GB run from the table above, repeated here for comparison:

| Prompt | Tokens (prompt + new) | Distinct experts | GB read | Hit rate |
|---|---|---|---|---|
| repetitive — `"The capital of France is"` (44 GB cache) | 5 + 48 | 2,641 | 37.4 | 80.3% (zero new misses on a warm repeat) |
| factual — `"What is the tallest mountain in the world, and how tall is it?"` | 14 + 48 | 4,043 | 57.2 | 74.7% |
| coding — `"Write a Python function ... def fib(n):"` | 17 + 48 | 4,072 | 57.6 | 75.7% |

The counters close exactly, which confirms these are per-prompt deltas rather than cumulative totals: (14 + 48) tokens x 6 routed experts x 43 scored layers = 15,996 expert visits = 11,953 hits + 4,043 misses. Likewise 4,043 misses x 14.16 MB = 57.2 GB, the figure the harness reports for that prompt alone.

Three things follow.

**The working set that fit was a property of the prompt.** A realistic prompt touches ~4,050 distinct experts, ~57 GB — more than the largest cache that fits on 64 GB hardware (~44 GB, and only with a raised wired limit). The LRU therefore churns, and each new prompt re-pays most of its cold cost. "Zero new misses on the warm pass" describes `"The capital of X is Y"` repeated forever, not inference in general.

**The ~75% hit rate is intra-prompt reuse, not cross-prompt reuse.** Prompt 1 started with an empty cache and got 74.7%. Prompt 2 started with a full 40 GB cache warmed by prompt 1 and got 75.7% — about one point better, from a different domain, which is within noise of nothing. Essentially all the reuse is the same experts recurring across the tokens of a single generation. The persistent-cache design in the harness buys much less than it was built to test.

**The SSD is not what the extra misses cost.** 57 GB of misses at the measured 13.3 GB/s is ~4.3 s of I/O per 48-token generation, which would be affordable. The cost lands on the memory system instead:

#### The swap cliff

The same 48-token factual generation, doing byte-identical work — identical hit and miss counts, identical 57.2 GB read — took **55.0 s** on one run and **928.8 s** on another: 17x apart with no difference in the work performed. `sysctl vm.swapusage` during the slow run showed 1.5 GB of swap in use. (The 928.8 s run is the one in `logs/diverse_run.txt`; the 55.0 s run was an earlier invocation of the same harness whose log was not retained, so only the slow side has a committed log.)

A 40 GB expert cache plus 9.4 GB of resident weights plus MLX scratch sits close enough to the raised 57,344 MB wired limit that macOS begins swapping, and once it does, throughput collapses. No cache counter can see this happen — the hit rate, miss count and byte count are identical on the fast and slow runs. This is the same cliff experiment 3 hit from the other direction, and it means cache size has to be chosen by measured end-to-end throughput, not by how much capacity the wired limit will nominally permit.

### Where the time goes, and the ceiling

A diagnostic run with no-op experts (resident path only: attention + the 20-iteration Sinkhorn hyper-connections) gives 7.8 tok/s = 128 ms/token. The warm streaming path costs ~410 ms/token. The ~284 ms gap is 43 per-layer Python sync round-trips — each layer must `mx.eval` its routing indices before Python can look up cache slots. That is the quantified case for a compiled runtime (Swift/Metal, or an oMLX contribution); it is not addressable from Python, as experiment 3 shows.

### Experiment autopsies

Three optimization attempts, all closed, each with a specific cause of death:

| # | Experiment | Result | Cause of death |
|---|---|---|---|
| 1 | **v1 slot pools** — experts cached in per-layer stacked pool arrays, indexed in-graph | 0.11 tok/s (22x slower) | MLX arrays are functional: any `setitem` on a stacked pool copies the entire pool array (~1 GB/layer) per insert |
| 2 | **Stacked `gather_qmm`** — batch all of a layer's expert matmuls into one kernel | 1.94 tok/s (no change) | Correct, but no speedup: the expert path is not kernel-dispatch-bound, so reducing kernel count buys nothing |
| 3 | **v2 optimistic execution** (`streaming/optimistic.py`) — run the whole token lazily with one `mx.eval`, verify routed ids afterward, repair misses and redo with free KV rollback (functional array references make snapshot/restore trivial); per-layer slot plan from a measured profile | Structure worked: pass 2 zero misses, clean single-attempt tokens, byte-correct output. But a *clean* token costs ~7.5–7.9 s vs v0's 0.41 s | `gather_qmm` over 44 GB of wired resident pools at the Metal wired-limit ceiling is pathological. One-sync-per-token achieved its design goal and lost on the memory system. Bonus finding: uniform per-layer slots thrash badly (skewed working sets); the measured slot plan fixed the thrash but not the matmul cost |

Verdict: v0 global-LRU is the Python result — 2.43 tok/s warm on the repetitive prompt, ~0.87 tok/s on a diverse one. The three experiments were all aimed at the per-token compute cost; the diverse-workload numbers say the larger lever is now cache behaviour on prompts whose working set does not fit. Commits `1579e06`, `4539f5e`, `2a20924` carry the full narrative.

## Architecture

### Repack layout (`streaming/repack.py`)

The HF checkpoint stores expert tensors stacked `[n_experts, ...]` with the expert axis first, so each expert's slice of each tensor is already one contiguous byte range in the shard. The repacker splits the snapshot into:

- `resident-XXXX.safetensors` — everything except routed experts (~9.4 GB: embeddings, attention, shared experts, gates, norms, head)
- `experts/layer_NN.bin` — 256 fixed-stride expert blobs per scored layer (3.62 GB/layer, 43 layers, 155.8 GB total)
- `experts/layout.json` — blob stride (14,155,776 B) plus per-slice offsets/shapes/dtypes

One blob = one expert's nine slices (gate/up/down projections x weight/scales/biases) concatenated in fixed order, page-aligned (16 KB) so a blob read never splits a page with its neighbor. Shards are processed one at a time through numpy memmaps — peak memory is one resident tensor, never a full shard. Resumable: finished layer files are skipped by size check. Verified byte-exact against independent range fetches.

### Store and runtime (`streaming/expert_store.py`, `run_streaming.py`)

- Explicit `pread` into reusable buffers, never mmap page faults — both the TurboFieldfare write-up and this project's llama.cpp baseline (fault storms SIGBUS even under a raised wired limit) say fault-driven streaming loses.
- Global LRU keyed on `(layer, expert_id)`, whole blobs as the unit of transfer and caching; `get_many` issues parallel preads for miss batches.
- `run_streaming.py` builds the pipenetwork model, quantizes the module tree to match the checkpoint, then swaps each layer's `SwitchGLU` experts for a `StreamingExperts` drop-in: gate → routed indices → `store.get` → per-expert quantized FFN.
- Hit-rate and byte counters are first-class; the first milestone of the project was the hit-rate-vs-cache-size curve, not tok/s.

### Model geometry that makes this viable

- 43 scored layers x 256 experts; 6 routed + 1 shared active per token → 258 routed visits/token, ~3.7 GB touched per cold token out of 165 GB.
- The first 3 layers are hash-routed by token id (`tid2eid`) — perfectly prefetchable before the forward pass even starts.
- MLA-style attention keeps the KV cache around 1 GB at 32k context, so nearly the whole memory budget goes to the expert cache.

## Limitations

- **2.4 tok/s is the Python ceiling with a fully cached working set, and this is a Python prototype.** The 284 ms/token of sync overhead requires a compiled runtime to remove; the no-op-expert ceiling on this hardware is 7.8 tok/s. On a prompt whose working set does not fit in cache, the binding constraint is not this ceiling at all.
- **The headline throughput is a best case.** The 2.22–2.43 tok/s, 80% and 86% figures all come from one repetitive prompt whose 48-token working set (37.4 GB) happens to fit in cache. The varied-prompt honesty check has now partly run and says a realistic prompt does not fit: ~57 GB working set, ~75% hit rate, ~0.87 tok/s. A perplexity validation through the streaming path (target: match pipenetwork's 6.13 for mixed 4/8-bit) is still pending.
- **The diverse-prompt evidence is 2 prompts, not 8 — this is partial data.** `multi_prompt_eval.py` defines 8 domains; the run was interrupted after `factual` and `coding` completed, so there is no cross-domain spread, no per-domain variance, and no summary JSON. Two agreeing measurements are enough to retire the "fits in cache" claim and not enough to characterize the workload. The remaining six prompts have not been run.
- **Cache sizing is unresolved; a sweep is in progress.** 40 GB is demonstrably past the point where memory pressure costs more than the extra hit rate returns (see the swap cliff), but the optimum is not known. A sweep over cache budgets, measuring end-to-end throughput rather than hit rate, is running now. Until it lands, no cache size in this README should be read as recommended — including the 32 GB in the quickstart.
- **Prefill is not batched or chunked.** Decode-shaped (one token at a time); prefill throughput is poor and long prompts have not been measured.
- **No prefetch yet.** Reads are demand-driven (parallel within a miss batch only). The hash-routed first 3 layers are exactly prefetchable and unexploited.
- **Wired-limit fragility.** Cache budgets above ~40 GB depend on `iogpu.wired_limit_mb`, which resets on reboot; experiment 3 shows large wired pools carry their own performance cliff, and the swap cliff above shows a large cache does too, at 17x.
- Known model-level issue independent of this repo: V4-Flash leaks DSML markers / repeats on markup-heavy agentic prompts even via the official API (llama.cpp#26694).

## Lessons for MLX streaming

1. **MLX arrays are functional; in-place writes into big pools are copies.** Any `setitem` on a stacked pool array materializes a copy of the whole array. A 1 GB per-layer pool costs ~1 GB of copy per inserted expert batch. This killed slot pools (experiment 1) and taxed optimistic repair (experiment 3). Cache experts as *individual* arrays and let a Python-side dict/LRU own the indexing; keep indirection out of the graph.
2. **Functional semantics cut the other way too: rollback is free.** Because rebinding `cache.kv` to a new array leaves saved references naming the old one, snapshot/restore of KV state is just collecting and reassigning references. Optimistic execution with redo is structurally easy in MLX even though it lost here for other reasons.
3. **Count syncs, not kernels.** Batching expert matmuls with `gather_qmm` changed nothing; collapsing 43 per-layer `mx.eval` round-trips into one was worth ~284 ms/token on paper. Per-layer graph breaks are the dominant Python-side cost in a routed-MoE loop.
4. **The Metal wired limit is a cliff, not a line.** Operating `gather_qmm` over 44 GB of wired pools at the raised ceiling turned a 0.41 s token into a ~7.5 s token. Staying comfortably under the limit with many small cached arrays behaved far better than a few huge wired pools at the edge. Relatedly, llama.cpp's full-GPU path dies at a ~60 GB residency ceiling (`res=-3`) on 64 GB hardware.
5. **Size the cache for headroom, not for capacity — and end-to-end, not by hit rate.** Two runs of the same generation with identical hits, misses and bytes read differed 17x in wall time (55.0 s vs 928.8 s); the slow one had 1.5 GB of swap in use. A 40 GB expert cache on top of 9.4 GB of resident weights sits near enough to the 57,344 MB wired limit to tip the machine into swapping, and every cache metric reports the two runs as identical. Hit rate is not a proxy for throughput once the cache is large enough to compete with the OS for memory.
6. **A repetitive prompt flatters every cache metric.** The same code shows an 80% hit rate with a working set that fits in cache on `"The capital of France is"`, and a 75% hit rate with a working set 1.5x larger than any available cache on an ordinary question. Benchmark prompts have to be chosen for their expert diversity, not their convenience, or the cache curve measures the prompt.
7. **Global LRU beats per-layer partitioning.** Per-layer working sets in a real MoE are skewed (58–140 experts/layer here); any uniform per-layer allocation simultaneously starves hot layers and wastes slots on cold ones.
8. **Explicit reads beat mmap.** 13.3 GB/s from SSD with `pread` + `F_NOCACHE` on contiguous page-aligned blobs; fault-driven paths SIGBUS or stall. Fixed-stride blob layout is what makes the read path this simple and this fast.
