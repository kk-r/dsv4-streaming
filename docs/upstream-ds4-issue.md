# Issue for antirez/ds4 — final text (ASD-STE-100)

**Title:** M5 Pro 64 GB, --ssd-streaming: an 8 GB expert cache is faster than a 32 GB expert cache

---

## Test configuration

- Mac with Apple M5 Pro, 64 GB unified memory, internal SSD, macOS 26.5.2
- ds4 built from `main` at commit `84cc882`, Metal backend
- Model: `ds4f-q2` (`DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`)
- Each run used a new process. Each run used `caffeinate -i`, `--temp 0`, and `-n 100`.
- We recorded `vm.swapusage` before and after each run. Swap did not increase in any run.

```
./ds4 -m <model> --ssd-streaming --ssd-streaming-cache-experts <N>GB \
      --temp 0 -n 100 --prompt-file <file>
```

## Results

| run | cache-experts | prompt | prefill t/s | decode t/s | max RSS |
|---|---|---|---|---|---|
| cold | 32GB | short (10 tokens) | 2.73 | 8.13 | 30.8 GB |
| warm | 32GB | short | 3.90 | 9.45 | 30.8 GB |
| | **8GB** | short | **8.93** | **11.37** | **5.1 GB** |
| | 16GB | short | 8.79 | 11.71 | ~18 GB |
| | 32GB | long (312 tokens) | 28.52 | 10.90 | 30.8 GB |

We think this is the first published measurement for an M5-class Mac. The README gives an M1 Ultra 64 GB as the reference machine, at approximately 5 t/s decode. The M5 Pro result is approximately two times faster.

## Finding

The 8 GB cache was better than the 32 GB cache in all three measurements:

- Decode: 11.37 t/s against 8.13-9.45 t/s
- Short-prompt prefill: 8.93 t/s against 2.73-3.90 t/s
- Resident memory: 5.1 GB against 30.8 GB

The 16 GB cache had the same speed as the 8 GB cache, but used three times more memory. The automatic budget (80 percent of the working set) selects a value near 32 GB on this machine. That configuration was the slowest and used the most memory.

## Possible cause

We also built a Python/MLX expert-streaming test system for the same model (mixed 4/8-bit weights, explicit pread, an application-side LRU cache). Our measurements there show the same pattern. The per-token expert working set is larger than each cache size that fits in memory. Because of this, a large application cache holds copies of data that the unified buffer cache also holds. The wired memory then decreases the page cache that serves the cache misses. In our 19-run sweep, 8 GB was the best value. Speed decreased at each size above 16 GB, while the hit rate increased. The `madvise(DONTNEED)` calls in ds4 decrease the duplication. On this hardware, they did not change the order of the results.

We can supply this data if it is useful:

- The cache sweep procedure: a new process for each run, replicate runs, swap tracking. Single-run times on these machines can change by a factor of two under background load. Medians of replicates are necessary.
- Per-layer routed working-set data for V4-Flash: 53 to 156 different experts per layer across a 48-token generation.
- Expert re-use data across an 8-prompt multi-domain workload: 9,344 of 11,008 experts touched, approximately 7 reads per different expert.

## Limits of this test

- Only the 32 GB configuration had a replicate run. All runs used the q2 model. The default hotlist preload was active. We did not test `--ssd-streaming-cold`.
- Short-prompt prefill speed is controlled by fixed costs. The long-prompt row (28.52 t/s) is the applicable prefill number.

## Questions

1. Is the 80-percent automatic budget intended for unified-memory Macs? Is a lower limit for Apple Silicon in scope?
2. Is a full sweep CSV (4 to 48 GB, with replicates, cold and warm) useful as a pull request in the speed-bench format?
