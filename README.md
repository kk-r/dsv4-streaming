# dsv4-streaming

SSD expert streaming for DeepSeek-V4-Flash (284B sparse MoE) on a 64 GB M5 Pro MacBook.

dsv4-streaming runs a 284B-parameter model whose smallest published quant is 83 GB on a 64 GB laptop, by keeping non-expert weights resident and streaming routed-expert weights from SSD on demand. It contains two engines: a **fast engine** built on [antirez's ds4 runtime](https://github.com/antirez/ds4) (~11 tok/s decode on a q2 quant) and a **full-quality engine**, a from-scratch MLX/Python stack that runs the mixed 4/8-bit checkpoint with perplexity matching the published number to -0.02%. A local chat webapp fronts both engines with sessions, streaming, and an OpenAI-compatible API. The repo doubles as a measurement lab: cache-size sweeps, pre-registered quality gates, A/B falsifications and closed-experiment autopsies are documented below with raw logs.

<!-- Demo: drop a screen recording as assets/demo.gif and delete this comment -->
![Demo](assets/demo.gif)

| Headline | Number |
|---|---|
| Fast engine (ds4 `--ssd-streaming`, q2 quant) | **11.4 tok/s** decode in 5.1 GB RSS |
| Quality engine (this repo's MLX streaming path, mixed 4/8-bit) | perplexity **6.1250 vs the published 6.1262 (-0.02%)**, full wikitext-2 corpus |
| Cache law | an **8 GB** expert cache beats a 32 GB one — in both engines |
| Chat webapp | `caffeinate -i python3 streaming/serve.py` → http://localhost:8399 |

## Table of contents

- [Quickstart](#quickstart)
- [Chat app features](#chat-app-features)
- [Results](#results)
- [Experiments and autopsies](#experiments-and-autopsies)
- [Architecture](#architecture)
- [Lessons for MLX streaming](#lessons-for-mlx-streaming)
- [Limitations](#limitations)
- [Upstream engagement](#upstream-engagement)
- [Repo map](#repo-map)

## Quickstart

Requires the pipenetwork mixed 4/8-bit checkpoint downloaded locally, ~160 GB free SSD, and macOS with MLX.

```bash
# 1. Repack the checkpoint into resident weights + per-layer fixed-stride expert files.
#    Resumable; safe to run while the snapshot is still downloading (--layers to restrict).
python streaming/repack.py --snapshot /path/to/hf-snapshot --out repacked/

# 2. Run streaming inference.
python streaming/run_streaming.py --repacked repacked/ --cache-gb 8 \
    --prompt "What is the tallest mountain in the world, and how tall is it?" --max-new 24

# 3. Chat. The webapp serves both engines at http://localhost:8399:
caffeinate -i python3 streaming/serve.py
#    Quality engine only (no ds4 build needed):
caffeinate -i python3 streaming/serve.py --no-fast
#    Or the terminal REPL:
caffeinate -i python3 streaming/chat.py --cache-gb 8
```

Notes:

- **`--cache-gb 8` is the recommended default**, and it is also the tool's default. It was the fastest budget in a 19-run sweep ([Cache-size sweep](#cache-size-sweep-realistic-prompt--the-main-result) below); 4 GB is statistically indistinguishable from it, and everything from 16 GB up is measurably slower. Bigger is worse here, not better.
- **The fast engine is optional.** It needs the ds4 runtime built in `ds4-runtime/` (a gitignored clone) and the q2 GGUF; `--fast-model` is a glob that defaults to the Hugging Face cache location. The quality model loads in ~1 s; the ds4-server child becomes ready ~10–60 s later in the background. `--no-fast` skips it entirely.
- **No `sysctl` step is needed for the recommended configuration.** At `--cache-gb 8` peak memory is 17.7 GB, far under any limit. `sudo sysctl iogpu.wired_limit_mb=57344` (which resets on reboot) is only relevant if you want to reproduce the large-cache end of the sweep — 24 GB and 32 GB budgets peak at 33.7 GB and 41.7 GB. Those configurations are slower, and they ratchet the machine's swap usage upward without releasing it afterwards.
- Run anything long under `caffeinate` (e.g. `caffeinate -i python3 ...`). The extreme "sporadic stalls" this README used to treat as an anomaly turned out to be the machine going to sleep mid-run — `pmset log` confirmed it. See [Cache size, swap, and sporadic stalls](#cache-size-swap-and-sporadic-stalls).
- Exclude the repacked directory from Spotlight — `touch repacked/.metadata_never_index` — before the first run. This is done in this repo. Spotlight was once the lead suspect for the stalls; sleep turned out to be the cause, so this is now an unmeasured, secondary precaution.
- Terminal output can be display-compressed; verify exact output from the written log files.
- `streaming/test_roundtrip.py` round-trips a synthetic quantized checkpoint through the repacker and store, bit-exact.

## Chat app features

The build arc is complete. Works today: a chat webapp over two engines, a terminal REPL (`streaming/chat.py`) at ~1–2 tok/s for short-form use at full mixed 4/8-bit quality under 24 GB peak memory, and an OpenAI-compatible API — all with clean prose output through DeepSeek's official chat encoding.

### The webapp

`streaming/serve.py` serves `streaming/webui/` at `/` — vanilla JS, no CDN. Two engines behind one model picker, one global generation lock across both:

- **`dsv4-fast`** — ds4's first-class `ds4-server` binary (OpenAI-compatible, SSE, server-side KV prefix cache), spawned as a persistent child process on port 8398 and proxied. Multi-turn with a KV prefix hit: **1.8 s turns**; warm single questions 5–9 s at 10–12 tok/s; no per-turn reload.
- **`dsv4-quality`** — the in-process MLX streaming stack, unchanged: mixed 4/8-bit, ~1–2 tok/s short-form, all MLX work on one persistent generation thread.

Sessions persist server-side in sqlite (`data/chat.db`): create/rename/delete, message history, SSE reply streaming, stop. The UI has sidebar sessions with rename/delete, markdown with copy-button code blocks, a model picker with tradeoff tooltips, collapsible thoughts, an info panel, opt-in debug chips (default off), and dark/light themes. Browser-verified end to end.

Two implementation finds from the webapp work: ds4-server silently burns hidden high-effort thinking tokens by default — the fast engine sends `think:false`, and `/v1` exposes `think`; and stop needed a raw-socket shutdown, because ds4's `Connection: close` detaches the socket — stop now lands instantly and partial replies persist, flagged. Voice input is deferred (whisper-cpp not installed) and images are impossible (text-only model); both are noted in the app's info panel.

### Command-line chat

`streaming/chat.py` wraps the streaming runtime in DeepSeek's official chat encoding (`encoding_dsv4`). All 4 bundled test vectors pass byte-exact, so output is clean prose — no leaked scaffolding, and reasoning is parsed out and kept separate from the answer. (The earlier diverse-eval runs, which fed raw completions with no chat template, produced correct answers wrapped in JSON/HTML fragments; the encoding fixes that, and it was the top usability gap, not a model issue.)

```bash
# Interactive REPL — persistent warm expert cache across turns:
caffeinate -i python3 streaming/chat.py --cache-gb 8

# One-shot:
caffeinate -i python3 streaming/chat.py --cache-gb 8 --prompt "Why is the sky blue?"
```

- Thinking and chat modes are both supported; EOS and end-of-assistant stops are handled.
- Decoding is greedy by default; `--temp` / `--top-p` enable sampling.
- Use `caffeinate` for anything long. The machine sleeping mid-run was the actual cause of the "sporadic stall" anomalies discussed below.

Measured at `--cache-gb 8` under `caffeinate`, replicated on AC and battery:

| Workload | Prefill | Decode | Peak memory |
|---|---|---|---|
| Short question ("Why is the sky blue?") | 9.6 s | 0.96 tok/s over 200 tokens | 18.3 GB |
| 315-token prompt | 141–152 s (~2.2 tok/s — minutes, not hours) | see [the decode-after-prefill verdict](#the-verdict-the-collapse-is-io-bandwidth) | 23.9 GB |

Both produced correct, clean answers (markdown prose for the short question, a correct summary for the long prompt). Short-form chat is usable at roughly 1 tok/s; long-prompt chat works, but decode afterwards slows down — that slowdown was chased through three falsified hypotheses to an I/O-bandwidth verdict, documented in [Experiments and autopsies](#experiments-and-autopsies).

### OpenAI-compatible API

`streaming/serve.py` (FastAPI/uvicorn) also exposes the chat stack as an OpenAI-compatible endpoint — encoding, generation, sampling and stop handling are reused from `chat.py` verbatim. It serves `POST /v1/chat/completions` (both SSE streaming and non-streaming) and `GET /v1/models` on `127.0.0.1:8399`; the `model` field routes the engine (`dsv4-fast` → ds4, `dsv4-quality` → the streaming stack, and it is the default). The models load once and stay resident; generation serializes on a global lock, so concurrent requests queue and get a 429 past `--queue-timeout`. Thinking is off by default and `max_tokens` is capped at 512. Each request originally built a fresh KV cache; conversation-level KV reuse has since landed — see [Conversation KV reuse and batched suffix append](#conversation-kv-reuse-and-batched-suffix-append).

Validated end to end with curl (`logs/serve_smoke.txt`, commit `07d1f29`): a cold non-streaming request returned a correct short answer in 4.0 s; the SSE path streamed incremental deltas ~0.7–0.9 s apart and terminated with `finish_reason=stop`, a usage block and `[DONE]`; SIGTERM shut down cleanly with no orphan processes and the port freed; peak memory 18.7 GB. Two less flattering findings from the same validation — a repeated-request regression, and a latent `get_many` bug the server was the first workload to hit — are covered in [F_NOCACHE prefill](#f_nocache-prefill-rejected) and in [Lessons](#lessons-for-mlx-streaming) (no. 12) respectively.

A user-reported crash — the second chat message aborting the whole process — was traced to MLX's thread-local streams (lazy arrays are thread-affine) and fixed by restructuring all MLX work onto one persistent generation worker thread; the exact fatal sequence now completes, malformed requests return 4xx with the server alive, and shutdown is clean (peak 18.5 GB). The general rule is [Lessons](#lessons-for-mlx-streaming) no. 13.

### Conversation KV reuse and batched suffix append

**KV reuse.** After a reply, the cache already holds the conversation's KV; turn N+1 only processes its suffix. Token-prefix stability of the chat encoding was verified offline (junctions are guarded by special tokens, no cross-boundary BPE merges; thinking mode is structurally prefix-unstable — `drop_thinking` re-renders `<think>` junctions — so its guard never matches and it safely rebuilds). Measured, with byte-identical outputs vs `--no-kv-reuse` (`logs/kv_reuse_smoke.txt`):

| Turn | With reuse | Without | Speedup |
|---|---|---|---|
| 2 | 59.4 s (prefill 15.0 s) | 131.3 s (prefill 67.9 s) | 2.21x |
| 3 | 51.3 s (prefill 16.3 s) | 134.6 s (prefill 82.0 s) | 2.62x |

The speedup grows with conversation length. Bonus corroboration of the decode-after-prefill mechanism: with less prefill traffic, decode ran 1.08–1.11 tok/s vs the control's 0.74–0.76, and peak memory fell 21.5 → 18.3 GB. The server keeps one resident conversation slot under the generation lock; reuse is reported via `usage.prompt_tokens_details.cached_tokens`.

**Batched suffix append.** Conversation suffixes now append to the KV cache in batched teacher-forced passes (`streaming/kv_append.py`, adapted from `spec_generate.py`'s bitwise-exact multi-token forward): 43 per-layer syncs per chunk instead of per token, plus parallel `get_many` reads. Flag `DSV4_SUFFIX_APPEND` (batch is the default; `decode` is the old path); chunk size defaults to 64 after a monotonic 8/16/32/64 sweep. Byte-identity gates all pass: one-token suffix, 128-window ring wrap, mid-window 30-token append (logits bitwise identical, next-8-greedy identical), and a 3-turn conversation with every reply byte-equal to the decode path. Long-suffix bench (65–71 tok/turn): turn-2 time to first token 66.7 → 40.4 s, turn-3 61.8 → 33.8 s — quality-engine long turns 1.6–1.8x faster to first token. Short suffixes (~12 tok) are parity; decode-append was never the bottleneck there. Raw logs: `logs/quality_turns.txt`, `logs/qt_bench_seq.txt`.

## Results

All measurements: M5 Pro, 64 GB, batch-1 greedy decode, 48 new tokens unless noted. Raw logs in `logs/`.

**Quality-engine headline:** byte-correct greedy decoding at **2.04 tok/s on a realistic prompt** with an **8 GB expert cache**, 17.7 GB peak memory and a 0.8 s model load — against 1.56–2.0 tok/s and a >10 minute cold start for the only llama.cpp configuration that works on this hardware at all (CPU-only), which runs a 1.58-bit quant where this runs mixed 4/8-bit. The fast engine, on the same machine and the same SSD-streaming principle, decodes at 11.4 tok/s on a q2 quant ([The ds4 benchmark on this machine](#the-ds4-benchmark-on-this-machine)).

**The counter-intuitive part is that 8 GB is the optimum.** A 19-run sweep over cache budgets found throughput *falls* as the cache grows — 2.04 tok/s at 8 GB versus 1.23 tok/s at 32 GB — even though the hit rate climbs monotonically over the same range (0.507 → 0.669). Buying hits with unified memory is a bad trade on this machine, apparently because the expert files are not opened with `F_NOCACHE`, so "misses" are largely served by the macOS page cache at RAM speed, and a big MLX-side cache evicts exactly the file pages that were doing the work. See [Cache-size sweep](#cache-size-sweep-realistic-prompt--the-main-result) below.

An earlier configuration with a 40 GB cache ran this same prompt text at 0.87 tok/s, and a previous revision of this README reported that as the realistic-prompt figure. It was a cache-sizing artifact. This is a learning-project write-up: the numbers are real but narrow (batch-1 decode, one prompt swept, eight diverse prompts characterized at a single cache size), the failed optimization experiments are documented below with causes of death, and one causal claim that appeared here has since been retracted by the sweep — see [Cache size, swap, and sporadic stalls](#cache-size-swap-and-sporadic-stalls).

Unless a subsection says otherwise, the prompt is the 5-token `"The capital of France is"`, which generates `"The capital of X is Y"` indefinitely. That repetition is what makes its expert working set small, so the hit-rate numbers below are best-case; [Cache-size sweep](#cache-size-sweep-realistic-prompt--the-main-result) and [Diverse-workload results](#diverse-workload-results-8-prompts-8-gb-cache) use a realistic prompt.

The headline throughput number comes from the sweep, so read that subsection first — it changes how the older ones should be read.

### Baseline comparison

llama.cpp b10280 with the 83 GB UD-IQ1_S GGUF was the only runnable prior art on this machine. Every GPU path fails:

| Config | Result |
|---|---|
| llama.cpp, full GPU offload | Fails: Metal residency ceiling ~60 GB (`res=-3`) |
| llama.cpp, `-ot exps=CPU` | SIGBUS in `ggml_compute_forward_mul_mat_id` (KERN_PROTECTION_FAILURE; same class as stale-closed llama.cpp#24413, reproduced here on a second architecture) |
| llama.cpp, partial offload | Prefill works; generation fails `res=-3` somewhere between `-ngl 1` (works) and `-ngl 22` (fails) |
| **llama.cpp, CPU-only (`-ngl 0`)** | **Works: 1.56–2.0 tok/s decode, 0.48–1.1 tok/s prefill, >10 min cold start** |
| **This repo, v0 streaming (mixed 4/8-bit)** | **Works: 2.04 tok/s on a realistic prompt at the recommended 8 GB cache (0.8 s load, peak 17.7 GB); 2.43 tok/s warm on the repetitive prompt** |

Note the quantization levels differ (1.58-bit GGUF vs mixed 4/8-bit here), so this compares deployability more than it isolates speed — the streaming path runs a much higher-quality quant with a 17.7 GB peak against an 83 GB checkpoint the baseline has to page through.

On the speed half of that comparison: at 2.04 tok/s the streaming path now sits at or slightly above the top of the llama.cpp CPU-only band (1.56–2.0 tok/s), on a realistic prompt rather than a degenerate one. A previous revision of this README said the streaming path was "most likely *slower* than the CPU-only baseline" on non-degenerate prompts. **That is retracted.** It was based on the 0.87 tok/s figure from a 40 GB-cache run; the same prompt text on the same store runs at 2.04 tok/s once the cache is sized correctly. The two are not a perfectly controlled pair — the 0.87 figure came from `multi_prompt_eval.py` generating 48 tokens, the 2.04 from `run_streaming.py` generating 24 — but a 2.3x gap is far outside anything the token count explains, and the sweep's own 32 GB runs reproduce the direction of the effect under fully controlled conditions.

Two things still qualify the comparison. The two rows are not equal-quality: the baseline is a 1.58-bit quant, so the streaming path is doing more arithmetic per token for a better output distribution; the perplexity check through the streaming path is complete and directly comparable — 6.1250 vs the published 6.1262 across all 195 windows of wikitext-2 (see [Perplexity through the streaming path](#perplexity-through-the-streaming-path-wikitext-2)). And the streaming figure is one prompt at 24 tokens; llama.cpp CPU-only holds every weight resident and has no cache dependence at all, so its number should hold across prompts while this one is a workload-sensitive measurement — though the completed 8-prompt eval ([Diverse-workload results](#diverse-workload-results-8-prompts-8-gb-cache)) now bounds the cross-domain spread at 1.52–2.30 tok/s at this cache size.

### Cache-size sweep (realistic prompt) — the main result

Raw data: `logs/cache_sweep.txt`, `logs/cache_sweep.json`.

Prompt `"What is the tallest mountain in the world, and how tall is it?"`, 24 generated tokens, `--store lru`, one separate process per run so every run starts cold. 19 runs: three full passes (forward, reverse, forward) plus four replicates.

| Cache | tok/s (median) | Hit rate | GB read | Peak mem | n |
|---|---|---|---|---|---|
| 4 GB | 2.00 | 0.434 | 78.6 | 13.7 GB | 3 |
| **8 GB** | **2.04** | 0.507 | 68.5 | **17.7 GB** | 5 |
| 16 GB | 1.57 | 0.587 | 57.3 | 25.7 GB | 2 |
| 24 GB | 1.12 | 0.632 | 51.0 | 33.7 GB | 2 |
| 32 GB | 1.23 | 0.669 | 45.9 | 41.7 GB | 2 |

**Throughput moves opposite to hit rate.** The hit rate rises monotonically with cache size, 0.434 → 0.669, and tok/s falls: 8 GB is **1.67x faster than 32 GB while hitting 16 points less often**. 4 GB and 8 GB are within noise of each other (medians 2.00 vs 2.04, bests 2.01 vs 2.08), so the optimum is a plateau at 4–8 GB rather than a sharp peak; the clear, repeated signal is the decline from 16 GB upward.

**The comparison is unusually clean.** Hits, misses, `read_gb` and peak memory are *deterministic* per cache size — every run at a given budget did byte-identical work and emitted byte-identical output text. Only wall time varies. So none of the spread between cache sizes is a workload difference; all of it is a system effect.

**Two honesty notes on the medians.** They exclude `pass1-fwd`, which ran while the machine was still being driven into swap by its own 24 and 32 GB runs and was uniformly slow at every budget — that is why `n` is 2 or 3 for most rows rather than 3 or 4; the full per-run table is in `logs/cache_sweep.txt`. And the 8 GB median of 2.04 is taken over five settled runs one of which was a 0.03 tok/s stall (see below); the median is deliberately robust to it, the mean would not be.

**Interpretation (well-supported, but not directly instrumented).** `streaming/expert_store.py` opens layer files with a plain `open(path, "rb", buffering=0)` — no `F_NOCACHE`. Only `streaming/bench_layer0.py`, the micro-benchmark, sets `F_NOCACHE`. So in production a "miss" is not necessarily an SSD read: most of the time it is a `pread` served out of the macOS unified buffer cache at RAM speed. That reframes the whole cache-size question. A small MLX-side LRU leaves most of RAM free for the page cache, which ends up caching the expert blobs *more* efficiently than our own LRU does, because it sees the whole 155.8 GB file set and is not bounded by a budget. A large MLX-side cache does the opposite: it wires down GPU memory holding duplicates of what the page cache already had, evicts those file pages, pushes anonymous memory into swap, and degrades the very read path that was doing the work. We were paying wired memory to duplicate a free service. This has not been confirmed by direct instrumentation — measuring it would mean counting `pread` service times or comparing against an `F_NOCACHE` build of the store, and neither has been done. (The same small-cache law later replicated in ds4's compiled runtime — see [The ds4 benchmark on this machine](#the-ds4-benchmark-on-this-machine).)

#### Cache size, swap, and sporadic stalls

Two claims used to be bundled together here under the heading "the swap cliff". The sweep supports one of them and refutes the other.

**Supported: large caches are slower, and they ratchet swap upward irreversibly.** Swap sat at 1520 MB through every 4, 8 and 16 GB run in the first pass. The 24 GB run took it to 3591 MB, the 32 GB run to 6657 MB, and it never came back — it stayed near 7 GB for the remaining 14 runs of the sweep, hours later. Small-cache runs never grew swap and often shrank it slightly. Combined with the medians above, "don't size the cache past ~8 GB" is now a 19-run result rather than a two-run anecdote.

**Retracted: the extreme stalls are not caused by large caches.** A previous revision of this README, and `docs/evidence-swap-cliff.md`, attributed a 55.0 s vs 928.8 s difference on byte-identical work (17x) to a 40 GB cache driving the machine into swap. The sweep breaks that causal chain:

- The worst stall in the sweep — 726.4 s for work that takes ~11.7 s, a **62x** collapse — happened at **8 GB**, with a peak of 17.7 GB. That is nowhere near any memory ceiling.
- Three replicates at 8 GB run immediately afterwards returned 2.04, 2.03 and 2.08 tok/s. Nothing about the configuration had changed.
- Swap *fell* during the stalled run (7090 → 6898 MB). Swap pressure does not track the stalls.
- The 1.5 GB of swap cited as evidence in the original write-up turns out to be roughly this machine's idle baseline: the sweep began at 1519.62 MB used with no model running. It was not a signature of the 40 GB run at all.

So the extreme stalls can strike any cache size, and cache size does not predict them. What cache size does predict is the ordinary, reproducible ~1.7x spread in the medians.

**Resolved: the stalls were the machine sleeping mid-run.** `pmset log` confirmed sleep during the stalled runs — the wall clock kept counting while nothing executed, which is exactly the observed shape: enormous, sporadic, unrelated to memory pressure, never reproducing under attention. This retro-explains the 55.0 s vs 928.8 s pair that founded the original "swap cliff" story, the 726.4 s outlier at 8 GB, and the earlier livelock diagnoses' wall-clock numbers. The fix is operational, not code: run anything long under `caffeinate`. An earlier hypothesis blamed Spotlight indexing of `repacked/`; that remains unconfirmed and is now secondary — `repacked/.metadata_never_index` is kept as a precaution, with no measured effect. The `pass1-fwd` observation stands separately: it was uniformly slow at *every* budget including 4 GB (1.61 tok/s where later passes gave 1.97–2.01), consistent with a whole-machine state effect.

**The methodological point survives intact, and is stronger than before:** cache size must be chosen by measured end-to-end throughput, never by hit rate, and never by how much the wired limit will nominally permit. It just needs more replicates than we originally gave it, because single runs on this machine are capable of being 62x off — even if, in this case, the 62x turned out to be a sleeping laptop rather than anything exotic.

#### Untested implication: this may fit a 32 GB Mac

The recommended configuration peaks at 17.7 GB, which is a very different machine requirement from the 41.7 GB the earlier headline needed. That suggests a 32 GB Mac could run this, which would be a much more interesting claim than anything else in this README — a 284B model on a 32 GB laptop.

It has not been tested, and there are two reasons to be cautious. First, 17.7 GB is MLX's reported peak *allocator* memory, not process RSS; the real footprint including the Python runtime, the resident weights' bookkeeping and I/O buffers is larger, and only a measurement on the actual hardware settles it. Second, and more fundamental: if the mechanism above is right, throughput on this machine depends on having tens of gigabytes of free RAM for the page cache to hold expert blobs. A 32 GB machine has far less of it, so it would take many more true SSD reads and should be expected to be *slower* — possibly a lot slower — even if it fits. "Fits" and "is usable" are separate questions here. (The fast engine strengthens the "fits" half: ds4 at an 8 GB cache decodes 11.4 tok/s in **5.1 GB RSS** on its q2 quant — see [The ds4 benchmark on this machine](#the-ds4-benchmark-on-this-machine).)

### Hit-rate vs cache size (repetitive prompt)

`ExpertStore` is a global LRU over 14.16 MB expert blobs, explicit `pread` (no mmap). 48-token run, ~13.4k expert visits:

| Cache budget | Hit rate | Notes |
|---|---|---|
| 8 GB | 47% | |
| 32 GB | 80% (86% at 256 tokens) | Working set exceeds budget; warm repeat is *slower* than run 1 (LRU thrash) |
| 44 GB | 80% run 1, **zero new misses run 2** | 48-token working set is 37.4 GB and fits entirely; warm pass 2.43 tok/s |

The 37.4 GB in that last row is a property of this prompt, not of the model: 2,641 distinct experts over 53 tokens. A diverse prompt touches ~3,800–4,500 and does not fit in any cache this machine can hold — see below. Read the table as a cache-size curve, not as a claim that the working set fits.

**This table is now the least useful one in the README, and it is kept for the record.** It ranks cache sizes by hit rate, and the sweep above shows hit rate is anti-correlated with throughput over exactly this range. The 44 GB row is the best row here and one of the worst configurations to actually run.

SSD micro-benchmark (real layer-0 blobs, `F_NOCACHE` so cold means cold): 1.07 ms per 14.16 MB blob = 13.3 GB/s. Note that `F_NOCACHE` is set only in `streaming/bench_layer0.py`; the production store in `streaming/expert_store.py` does *not* set it, so this 13.3 GB/s is a floor on real miss cost rather than the cost actually paid — see the sweep's interpretation above. SSD traffic is ~1.9 GB/token cold and falls as the cache warms.

Per-layer working sets are heavily skewed, which is why global LRU beats any uniform per-layer allocation (see the failures table, experiment 3). The original citation for this was `logs/hitrate_last_run.json` at 58–140 distinct experts per layer over a 48-token run; **that file has since been overwritten by the last run of the sweep** and no longer holds the data it was cited for. The finding survives in the sweep's retained per-size files: `logs/hitrate_32gb.json` shows 53–156 misses per layer on the 24-token realistic prompt (a lower bound on distinct experts touched, since the 32 GB budget evicts little), a nearly 3x spread across layers.

### Diverse-workload results (8 prompts, 8 GB cache)

Raw data: `logs/diverse_eval.txt`, `logs/diverse_eval.json` (commit `a90aa44`).

`streaming/multi_prompt_eval.py` runs 8 prompts from different domains through one persistent-cache process and records per-prompt *deltas* in hits, misses and bytes read. The full eval completed end to end at the recommended 8 GB budget, 48 new tokens per prompt (translation self-terminated at 32). An earlier attempt at a 40 GB cache was abandoned after 2 of 8 prompts (`logs/diverse_run.txt`); its wall times are void — bad cache size, and its 928.8 s factual run is the same sleep-contaminated run dissected in [Cache size, swap, and sporadic stalls](#cache-size-swap-and-sporadic-stalls) — but its counter structure is still informative and is folded in below.

| # | Domain | tok/s | Hit rate | GB read | Distinct experts | New distinct | Cumulative distinct |
|---|---|---|---|---|---|---|---|
| 1 | factual | 2.29 | 52.9% | 106.7 | 3,905 | 3,905 | 3,905 |
| 2 | coding | 2.30 | 53.6% | 110.0 | 4,260 | 2,121 | 6,026 |
| 3 | math | 1.58 | 59.4% | 137.8 | 4,482 | 1,137 | 7,163 |
| 4 | creative | 1.74 | 47.4% | 124.9 | 3,988 | 887 | 8,050 |
| 5 | translation | 1.52 | 57.1% | 89.2 | 3,822 | 393 | 8,443 |
| 6 | science | 2.00 | 49.1% | 119.0 | 4,139 | 389 | 8,832 |
| 7 | history | 1.82 | 49.0% | 117.3 | 3,841 | 220 | 9,052 |
| 8 | casual | 1.59 | 52.6% | 121.2 | 4,499 | 292 | 9,344 |

Overall: 52.9% hit rate, 926 GB read, 9,344 distinct experts over all 8 prompts, 17.8 GB peak memory.

**Throughput holds across domains: 1.52–2.30 tok/s, median ~1.78.** Coding and factual are fastest (2.30 / 2.29); translation is slowest at 1.52 with math just above it at 1.58 — math read the most bytes (137.8 GB; it has the longest prompt). No domain collapses; the whole spread is about 1.5x.

**Cross-prompt warming is real and strong.** New distinct experts per prompt fall 3,905 → 2,121 → 1,137 → 887 → 393 → 389 → 220 → 292: from prompt 5 on, a fresh prompt from a fresh domain lands in roughly 90% already-seen territory. The partial run had made cross-prompt reuse look like nothing (see below); part of that was its bookkeeping — its per-prompt "distinct experts" were really per-prompt *miss* deltas, which cannot see hits on experts a previous prompt warmed. The completed run tracks true distinct sets, and half of coding's 4,260 distinct experts were already seen during factual.

**But there is no small hot set.** The 8 prompts together touch 9,344 of the model's 11,008 routed experts — 85% of the entire model — and the union would need 132 GB to cache. Every prompt individually uses ~3,800–4,500 distinct experts against the 565 blob slots an 8 GB cache holds, so the LRU hit rate sits near 53% no matter the domain, and each distinct expert is fetched from the store ~7x on average over the run (65,424 misses over 9,344 distinct experts). Per the sweep, those "misses" are mostly page-cache-served, which is how ~110 GB of reads per prompt coexists with 2.3 tok/s. This is the deep justification for the sweep's conclusion: reuse exists, but at a breadth only the OS page cache — which sees the whole 155.8 GB file set and uses all free RAM — can exploit. Growing the MLX-side LRU chases a working set that does not fit on this hardware in any configuration.

**The counters close exactly**, confirming the harness reports per-prompt deltas rather than cumulative totals. Routed visits per prompt are (prompt + generated tokens) x 6 routed experts x 43 scored layers; for the factual prompt, (14 + 48) x 6 x 43 = 15,996 visits — which the 40 GB partial run split as 11,953 hits + 4,043 misses, and the 8 GB run splits as 8,460 hits + 7,536 misses. Byte-identical work, redistributed between hits and misses by cache size. (The JSON's `prompt_tokens` runs one higher than the count that closes the arithmetic — 15 vs 14 for factual — consistent with one unscored leading token; every row closes as (`prompt_tokens` − 1 + generated) x 258.)

**The working set that fit was a property of the prompt.** The repetitive prompt's 2,641 distinct experts fit a 44 GB cache with zero new misses on a warm repeat; every realistic prompt here touches ~3,800–4,500 distinct experts (~54–64 GB) — more than the largest cache that fits on 64 GB hardware (~44 GB, and only with a raised wired limit). "Zero new misses on the warm pass" describes `"The capital of X is Y"` repeated forever, not inference in general. The partial-data revision wrote this as bad news, on the assumption that chasing the working set with a bigger cache was the goal; the sweep — and now the 85%-of-model footprint above — say the chase was the mistake. Neither a single prompt's working set nor the cross-prompt union can fit, so the right move is to stop competing with the OS for memory and let the page cache do the caching.

**The ~75% hit rate at 40 GB was intra-prompt reuse, and the completed run confirms it from the other side.** In the partial run, factual started cold and hit 74.7%; coding started on a 40 GB cache warmed by factual and hit 75.7% — one point, from a different domain. Warming changes *which* experts are new (the new-distinct column above), but hit rate barely moves because hits are dominated by the same experts recurring across the tokens of one generation — each prompt makes ~16,000 visits to ~4,000 distinct experts, ~4 visits each, at any cache size. The economics did change with cache size, in the counter-intuitive direction the sweep predicts: at 40 GB these prompts read ~57 GB each at 0.87 tok/s on the factual prompt's clean run; at 8 GB the same prompts read ~107–110 GB each — nearly double the bytes — and run at 2.3 tok/s. Paying more misses to a page-cache-backed read path beats wiring memory to avoid them; see [Cache-size sweep](#cache-size-sweep-realistic-prompt--the-main-result) for the mechanism.

**Output content is correct but document-scaffolded.** The factual prompt gives the right Everest answer inside a JSON fragment, coding wraps a correct Fibonacci note in HTML, math produces coherent step-by-step arithmetic behind a stray `</think>`, translation renders accurate French, history answers Augustus correctly mid-quotation. These runs predate `streaming/chat.py` and feed raw completions with no chat template, so the model continues whatever document it imagines it is in. Content was never the problem — see [Command-line chat](#command-line-chat) for the encoding that fixed the scaffolding.

#### The "swap cliff" (superseded)

This subsection previously argued that a 40 GB cache tips the machine into swap and that this is what produced a 17x wall-clock difference (55.0 s vs 928.8 s) on byte-identical work. The cache-size sweep tested that directly and only half of it held up. See [Cache size, swap, and sporadic stalls](#cache-size-swap-and-sporadic-stalls) above for the corrected version: large caches are reliably slower and do ratchet swap upward, but the extreme stalls occurred at small cache sizes too and were ultimately traced to the machine sleeping mid-run, not to cache size. The transcribed evidence is preserved in `docs/evidence-swap-cliff.md` with the same correction noted.

### Perplexity through the streaming path (wikitext-2)

**Final result: 6.1250 vs the published 6.1262 (-0.02%), over all 195 windows and 199,485 scored tokens** — the streaming path is numerically transparent. The run used the identical window construction and checkpoint as the published measurement, and was completed in two parts after an external interruption at window 90 (per-window teacher-forced NLLs are independent, so the combined aggregate is exact; a `--start` resume flag was added to `ppl_streaming.py`). Window 1 reproduced the earlier partial run to four decimal places, confirming determinism. Raw logs: `logs/ppl_full.txt`, `logs/ppl_full_part2.txt`.

The earlier partial run that established the method (raw log `logs/ppl_streaming.txt`; script `streaming/ppl_streaming.py`; commit `b9374ec`): teacher-forced perplexity through the full streaming path (`load_streaming`, repacked store, 8 GB cache), using the exact published window construction — the `ppl_corpus.py` pipeline reproduced end to end — wikitext-2-raw-v1 test set, non-empty chunks joined with `"\n\n"`, bundled V3 tokenizer, first 200k tokens → 195 windows x 1024 tokens, 1023 scored per window — with the NLL math copied verbatim from the reference `ppl_large.py`. Windows 1–8 were run first:

| Window | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| ppl | 4.08 | 3.03 | 2.54 | 3.09 | 3.10 | 4.97 | 6.60 | 6.02 |

Aggregate: **3.9492 over 8,184 scored tokens**, against the published full-corpus **6.1262** (195 windows). The two are not directly comparable — this is 8 of 195 windows, and evidently the easy end of the corpus. The verdict was nonetheless *consistent with numerical transparency*, for three reasons: the deviation is downward, and streaming/dequant numerics bugs raise NLL rather than lowering it systematically; windows 7–8 (6.60, 6.02) individually straddle the published mean; and the per-window curve is smooth, with no spikes or degenerate values. Together with the bit-exact repack round-trip and the batched-prefill regression check, there is no evidence of a streaming-path bug — and the full 195-window run above confirms it directly.

Throughput was 3.67 tok/s teacher-forced (253–307 s per 1024-token window), peak 24.6 GB. Side finding: at 1024-token batches the MLX-side LRU had **zero hits** — the per-window working set dwarfs the 8 GB budget, so all 1,016.75 GB of expert reads over the run were served by the page cache/SSD. The LRU is a pure passenger at large batch sizes; it earns its keep only at decode.

### The ds4 benchmark on this machine

Raw log: `logs/ds4_bench.txt`; the filed upstream version is `docs/upstream-ds4-issue.md`.

ds4 q2-imatrix GGUF, `--ssd-streaming`, greedy, 100 tokens, fresh process per run, `caffeinate`, swap tracked (it never grew):

| Config | Decode | Prefill | Max RSS |
|---|---|---|---|
| 32 GB cache (their auto-budget's recommendation) | 8.1–9.5 tok/s | 2.7–3.9 tok/s | 30.8 GB |
| **8 GB cache** | **11.4 tok/s** | 8.9 tok/s | **5.1 GB** |
| 16 GB cache | 11.7 tok/s | 8.8 tok/s | ~18 GB |
| 32 GB cache, 312-token prompt | 10.9 tok/s | 28.5 tok/s | 30.8 GB |

Three findings:

- **First rigorous 64 GB M5 datapoint for ds4.** Their reference machine is an M1 Ultra 64 GB at ~5 tok/s decode; the M5 Pro is roughly 2x that.
- **The small-cache law replicates in their compiled runtime.** 8 GB beat 32 GB on decode, on short-prompt prefill, and by 6x on resident memory — contradicting their 80%-of-working-set auto-budget on this machine class, and under `madvise(DONTNEED)`/`F_RDADVISE` page-cache cooperation our store lacks. This is the finding filed upstream ([Upstream engagement](#upstream-engagement)).
- **No long-context collapse.** At a 312-token prompt ds4 decodes 10.9 tok/s where our Python store drops to 0.62 — their async prefetch dissolves the bandwidth wall we proved structural for our store ([The verdict](#the-verdict-the-collapse-is-io-bandwidth)).

Caveats, as filed: only the 32 GB configuration had a replicate; all runs used the q2 model; the default hotlist preload was active; `--ssd-streaming-cold` was not tested; short-prompt prefill speed is dominated by fixed costs, so the 28.5 tok/s long-prompt row is the applicable prefill number. Harness lessons: ds4 loads `metal/flash_attn.metal` relative to CWD, and shell word-splitting of packed prompt arguments produced two false-failure rounds before the numbers above.

### Where the time goes, and the ceiling

A diagnostic run with no-op experts (resident path only: attention + the 20-iteration Sinkhorn hyper-connections) gives 7.8 tok/s = 128 ms/token. The warm streaming path costs ~410 ms/token. The ~284 ms gap is 43 per-layer Python sync round-trips — each layer must `mx.eval` its routing indices before Python can look up cache slots. That is the quantified case for a compiled runtime; it is not addressable from Python, as the failures table shows. (The recon that followed concluded the compiled runtime already exists — see [Upstream engagement](#upstream-engagement).)

The recommended 8 GB configuration runs the realistic prompt at ~490 ms/token, so the same accounting applies to it: roughly 128 ms of unavoidable resident-path work, ~284 ms of Python sync overhead, and only the remainder attributable to expert streaming. Now that cache sizing is no longer costing several hundred ms/token, sync overhead is once again the largest single line item. The context-scaling probe later confirmed the resident path barely scales with context (7.53 → 6.27 tok/s no-op, short vs long) — see [The verdict](#the-verdict-the-collapse-is-io-bandwidth).

## Experiments and autopsies

Every optimization idea this project tried is recorded here with its measured outcome — most were closed with a specific cause of death, one shipped by accident, and two shipped as opt-in flags. Raw logs in `logs/`.

### Three closed optimization rounds (the failures table)

| # | Experiment | Result | Cause of death |
|---|---|---|---|
| 1 | **v1 slot pools** — experts cached in per-layer stacked pool arrays, indexed in-graph | 0.11 tok/s (22x slower) | MLX arrays are functional: any `setitem` on a stacked pool copies the entire pool array (~1 GB/layer) per insert |
| 2 | **Stacked `gather_qmm`** — batch all of a layer's expert matmuls into one kernel | 1.94 tok/s (no change) | Correct, but no speedup: the expert path is not kernel-dispatch-bound, so reducing kernel count buys nothing |
| 3 | **v2 optimistic execution** (`streaming/optimistic.py`) — run the whole token lazily with one `mx.eval`, verify routed ids afterward, repair misses and redo with free KV rollback (functional array references make snapshot/restore trivial); per-layer slot plan from a measured profile | Structure worked: pass 2 zero misses, clean single-attempt tokens, byte-correct output. But a *clean* token costs ~7.5–7.9 s vs v0's 0.41 s | `gather_qmm` over 44 GB of wired resident pools at the Metal wired-limit ceiling is pathological. One-sync-per-token achieved its design goal and lost on the memory system. Bonus finding: uniform per-layer slots thrash badly (skewed working sets); the measured slot plan fixed the thrash but not the matmul cost |

**Caveat on experiment 3, added after the sweep.** That autopsy blamed `gather_qmm`-over-huge-pools — the operation. The sweep raises a competing explanation: the **44 GB itself** may have been the problem, independent of what was running on top of it. v0 at a 32 GB cache is already ~1.7x slower than v0 at 8 GB purely from memory pressure, with no pools and no `gather_qmm` involved, so a large share of v2's cost may have been the same effect rather than anything specific to the kernel. This does not overturn the conclusion above — v2 was ~18x slower than v0, far more than the ~1.7x the sweep attributes to memory pressure alone, so something specific to the optimistic path was also expensive. It does mean the experiment was never run in a regime where it could have succeeded. **v2 has not been re-tested at a small pool budget, and it should be** before "one-sync-per-token loses on MLX" is treated as settled.

**A later correction to experiment 3's design notes:** the "free KV rollback via functional array references" claim in `optimistic.py` turned out to be false — see the speculative-decoding findings below and [Lessons](#lessons-for-mlx-streaming) no. 2.

Verdict: v0 global-LRU is the Python result — 2.04 tok/s on a realistic prompt at an 8 GB cache, 2.43 tok/s warm on the repetitive one. The three experiments were all aimed at the per-token compute cost. The sweep says the larger lever was never compute or cache *policy* but cache *size*, in the opposite direction from the one everything here assumed. Commits `1579e06`, `4539f5e`, `2a20924` carry the full narrative.

### Speculative decoding: rejected in the regime that matters

Prompt-lookup drafting plus batched greedy verification (`streaming/spec_generate.py`), tested with a bitwise correctness gate. Decode tok/s, 96 tokens, cold process, 8 GB cache (`logs/spec_decode.txt`):

| Prompt | Plain | Exact k=4 | Exact k=8 | Acceptance |
|---|---|---|---|---|
| capitals | 2.13 | 1.12 | 0.84 | 0.41/0.22 |
| freeform | 2.58 | 1.39 | 1.00 | 0.00 |
| numbered list | 0.79 | 0.76 | — | 0.00 |
| JSON schema | 0.72 | 0.40 | 0.35 | 0.31/0.20 |

Spec decode cut forwards/token by up to 35% with identical output and **still lost throughput everywhere** (0.44–0.96x). The amortization hypothesis assumed per-pass cost is fixed; in the realistic IO-bound regime it is not — expert traffic scales with *drafted* tokens (capitals: 206 → 344 GB read for the same 96 emitted tokens), and rejected drafts pollute the LRU. Speedup is bounded by (accepted+1)/(k+1) ≤ 1 on the current store. The 128+284 ms accounting that motivated this was measured at 100% hits, where spec would win ~2.5x. Path to viability, if ever: batch blob decode/insert + overlap preads, then ~34% token acceptance at k=8 suffices; DSpark-MTP could deliver that. The bitwise `_verify_exact` is the substrate for that future test.

Two findings that outrank the numbers:

1. `optimistic.py`'s "free snapshot via functional arrays" lore is **false** — `mx.array.__setitem__` mutates the same object (repro in the log). Its rollback only worked because the redo rewrites identical values. Docstring corrected; real snapshots need `x[:]`.
2. No shape-changing verifier is byte-identical on this stack: batched MoE/hyper-connection kernels drift logit margins up to ~1.0 and deterministically flip near-tied argmaxes. Exact mode (per-token shapes, batched syncs only) is bitwise-identical and is what the gate passed with.

### Pre-gated expert prefetch: +8% short-form only, default off

The Mixtral-offloading / Pre-gated MoE technique on our store (`DSV4_PREFETCH=1`, `logs/pregate_prefetch.txt`): hash layers 0–2 prefetched *exactly* from the token id at token start; scored layer L+1 predicted by applying its resident gate to layer L's MoE input, folded into the existing per-layer sync (still one eval). Async workers do `pread` + numpy only; mx arrays are created on the main thread exclusively (the MLX thread-affinity crash class, Lessons no. 13); mispredictions never enter the LRU, so cache residency is provably byte-identical to a prefetch-free run.

- Byte-identity: pass everywhere (token-id-exact on long runs).
- Prediction accuracy on a real 284B MoE: hash 100%, scored 70–77% by band (low end of the papers' 70–90%); 53–68% of decode misses served from the prefetch buffer with waits rare — the reads genuinely land under compute.
- Speed: short-form decode 2.63/2.58 → 2.85/2.80 tok/s (+8.4%, replicated). Long-form: no gain, directionally worse — prefetch added +107% read traffic (including 5,435 wasted reads) to a bandwidth-saturated, page-cache-cold pipe.

Same law as the spec-decode rejection: in the IO-bound regime, extra bytes are pure loss, so 70% accuracy does not pay. The default stays off; the quantified prediction table is the contribution. The port hook (a 2-line `token_hook` in the model port's `model.py`) rides the gitignored clone and is captured in `patches/deepseek-v4-mlx-token-hook.patch`.

### AQLM 2-bit quality gate: KILL

A pre-registered week-1 gate on whether codebook (additive VQ) 2-bit quantization could halve expert bytes without wrecking quality. Three arms on layers 18–23 (the other 37 layers normal), wikitext windows 1–8, criteria fixed in advance (PASS: within 1–2% of control; KILL: >5%). Code in `aqlm/`, full pre-registration and progress in `logs/aqlm_gate.txt`:

| Arm | Configuration | ppl (windows 1–8) | vs baseline |
|---|---|---|---|
| A | container control (dequant → requant 4-bit) | 3.9483 | -0.02% |
| B | codebook 2-bit (additive VQ, 2x256, dim 8) | 4.3211 | +9.44% |
| C | affine 2-bit (known-bad comparator) | 4.3814 | +10.97% |

The test was powered (C clearly detected at 6 layers). The decisive finding: codebook 2-bit recovered only ~14% of affine's perplexity damage despite 25% lower weight MSE — the learned-codebook MSE advantage does not translate to perplexity on these experts. Naive compounding extrapolates full-model ppl to ~11.7 vs the published 6.125. Upper-bound caveat: the source was the already-4-bit repack (the original FP4 was deleted), but arm A proves that path costs ~zero, and the miss is 5–9x the pass band.

The bytes-halving track is closed unless activation-aware calibration or the original FP4/FP8 source re-opens it. Weeks of Metal-kernel work were saved by a pre-registered gate. Rebuilt-layer evidence lives in `aqlm/out_*` (untracked, ~65 GB, deletable).

### F_NOCACHE prefill: rejected

**The problem this and the next two sections chase: decode after a long prefill drops to ~0.25–0.38 tok/s**, against 2.04 tok/s in the short-prompt benchmark. Reproduced across runs with identical store counters. (Background: `StreamingExperts` dispatches multi-token prefill through the verified `gather_qmm` batch path — see [Engines](#engines) — and a regression check confirmed decode output is byte-identical to the pre-batching code.)

The obvious explanation was tested first, and rejected. The theory: a long prefill streams ~300 GB of misses and evicts the page-cache file pages that make decode misses cheap (see [Cache-size sweep](#cache-size-sweep-realistic-prompt--the-main-result) for why those pages matter), so bypassing the page cache for prefill reads should spare them and recover decode speed. Implementation: a second file descriptor opened with `F_NOCACHE`, used only by `get_many` (the >1-token batch path), decode reads untouched, behind the env flag `DSV4_PREFILL_NOCACHE`. The A/B — 316-token prompt, greedy, 100 new tokens, raw log in `logs/fnocache_ab.txt`:

| Run | Prefill reads | Prefill | Decode after |
|---|---|---|---|
| control | page-cached | 136.1 s | 0.37 tok/s |
| treatment 1 | `F_NOCACHE` | 237.8 s | 0.25 tok/s |
| treatment 2 | `F_NOCACHE` | 127.5 s | 0.38 tok/s |

Zero decode recovery, up to +75% prefill cost. Store counters were identical in every run (11,825 hits / 21,169 misses / 299.66 GB read) and generated text was byte-identical across all three, so the comparison is clean in the same sense the sweep was: only wall time varied. "Prefill evicts decode's pages" is therefore largely ruled out as the binding constraint — decode after a long prefill is cold either way, and decode itself streams ~100+ GB per 100 tokens, self-evicting whatever prefill spared. The feature is kept in the code behind `DSV4_PREFILL_NOCACHE`, **default off**; a short-prompt sanity run with it on decoded at 1.07 tok/s, so the flag is behavior-safe, just not useful.

One confounder was on record at the time: the machine carried 9–15 GB of ambient swap from unrelated workloads across all three runs, and run-to-run variance under that pressure is ~2x (the two treatment prefills, 237.8 s and 127.5 s, bracket the control).

**The server validation independently corroborated the problem from the request side.** Identical repeated requests to `streaming/serve.py` regressed 4.0 → 8.4 → 9.8 s on byte-identical work (`logs/serve_smoke.txt`): each identical repeat incurred ~1,830 fresh misses and ~26 GB of re-reads, with near-zero LRU hits. Per-prompt working sets exceed any affordable cache, so cross-request expert persistence buys little at prefill scale — the same root cause as the decode-after-prefill slowdown. (Not a server-level defect, and single timings on this machine vary ~2x; but the direction of the regression was consistent across the repeats.)

### LRU protection (prefill no-evict): rejected, with an accidental 2x prefill win

The second hypothesis came from the ds4 recon: ds4's prefill fills but never evicts its expert cache, so maybe prefill evicting *our LRU* (not the page cache) causes the decode collapse. **Rejected** (`logs/prefill_lru.txt`): with fill-no-evict, decode is unchanged (0.62 vs 0.63 tok/s; decode misses 13,813 vs 13,825) — decode's working set dwarfs 565 slots no matter what prefill leaves behind. Outputs byte-identical.

But the same A/B found something better: **prefill dropped from 76.3 s to 36–39 s (2x, replicated)**, because no-evict skips 6,608 pointless insert+evict cycles — decoding blobs into the LRU and immediately evicting them was costing prefill half its wall clock. Short-prompt sanity was unaffected (2.76 tok/s decode). The default flipped to on.

That reframed the residual mystery: long-context decode (0.6 tok/s at ~410 ctx) vs short-context (2.76 at ~110 ctx) shows similar decode GB/token (1.96 vs 1.83) — both cache-side explanations (`F_NOCACHE`, LRU protection) falsified, and per-token read volume barely differs. The remaining suspect was context-length-dependent compute in the Python port's attention path, which motivated the probe below.

### The verdict: the collapse is I/O bandwidth

The context-scaling probe (`logs/probe_noop_short.txt`, `logs/probe_noop_long.txt`) exonerated the attention path: with no-op experts the resident path runs 7.53 tok/s at short context vs 6.27 at long (1.2x) — it barely scales with context. With real experts: 2.76 vs 0.62 (4.4x). The collapse is in expert I/O: ~1.9 GB/token in both regimes, but ~5 GB/s effective after short prompts vs ~1.6 GB/s after long ones. A long prefill's ~300 GB flood plus decode's own ~2 GB/token self-flood leave decode structurally page-cache-cold.

This was the fourth and final falsification for this mystery (`F_NOCACHE`, swap, LRU eviction, attention compute) — the mechanism is bandwidth, not a bug. Fixes require fewer bytes per token or routing-ahead prefetch, which is ds4 territory: their async prefetch dissolves exactly this wall ([The ds4 benchmark on this machine](#the-ds4-benchmark-on-this-machine)).

### Top-k=4 routing: opt-in turbo

`DSV4_TOPK=4` serves top-4 of the trained top-6 routing (`logs/topk_experiment.txt`). Freeform 96-token decode: 1.17 → 1.97 tok/s (**+68%**; reads 167 → 117 GB). Quality on the same 4 wikitext windows as the top-6 baseline: ppl 3.14 → 3.73 (**+19%**) — costlier than the whole 4/8-vs-8-bit quant gap (+2.3%) and than REAP25 expert pruning (+4.5%). Speed beats naive bytes math because IO latency compounds. The flag stays opt-in; the top-5 curve point is untested.

## Architecture

DeepSeek-V4-Flash is 284B total / ~13.8B active parameters; at mixed 4/8-bit quantization the checkpoint is 165 GB, and no published quant fits a 64 GB Mac (smallest GGUF is 83 GB, smallest MLX 92.7 GB). This project keeps the ~9.4 GB of non-expert weights resident and streams the 155.8 GB of routed-expert weights from SSD on demand, with an LRU cache in unified memory — the approach [drumih/turbo-fieldfare](https://github.com/drumih/turbo-fieldfare) demonstrated for a 26B Gemma, scaled up 10x. It is built on the [PipeNetwork/deepseek-v4-mlx](https://github.com/PipeNetwork/deepseek-v4-mlx) from-scratch MLX port, because no mainstream runtime supports the `deepseek_v4` architecture (mlx-lm PR #1189 still open at time of writing).

### Repack layout (`streaming/repack.py`)

The HF checkpoint stores expert tensors stacked `[n_experts, ...]` with the expert axis first, so each expert's slice of each tensor is already one contiguous byte range in the shard. The repacker splits the snapshot into:

- `resident-XXXX.safetensors` — everything except routed experts (~9.4 GB: embeddings, attention, shared experts, gates, norms, head)
- `experts/layer_NN.bin` — 256 fixed-stride expert blobs per scored layer (3.62 GB/layer, 43 layers, 155.8 GB total)
- `experts/layout.json` — blob stride (14,155,776 B) plus per-slice offsets/shapes/dtypes

One blob = one expert's nine slices (gate/up/down projections x weight/scales/biases) concatenated in fixed order, page-aligned (16 KB) so a blob read never splits a page with its neighbor. Shards are processed one at a time through numpy memmaps — peak memory is one resident tensor, never a full shard. Resumable: finished layer files are skipped by size check. Verified byte-exact against independent range fetches.

### Store and runtime (`streaming/expert_store.py`, `run_streaming.py`)

- Explicit `pread` into reusable buffers, never mmap page faults — both the TurboFieldfare write-up and this project's llama.cpp baseline (fault storms SIGBUS even under a raised wired limit) say fault-driven streaming loses.
- Layer files are opened `open(path, "rb", buffering=0)` — **no `F_NOCACHE`**. This was incidental when written and turns out to be the most performance-relevant decision in the store: reads go through the macOS unified buffer cache, so a large fraction of LRU "misses" are RAM-speed, and leaving RAM free for that page cache beats enlarging our own. See [Cache-size sweep](#cache-size-sweep-realistic-prompt--the-main-result).
- Global LRU keyed on `(layer, expert_id)`, whole blobs as the unit of transfer and caching; `get_many` issues parallel preads for miss batches. Prefill inserts are fill-no-evict by default (the 2x prefill win from the [LRU-protection A/B](#lru-protection-prefill-no-evict-rejected-with-an-accidental-2x-prefill-win)).
- Behavior flags, all measured above: `DSV4_PREFETCH=1` (pre-gated prefetch, default off), `DSV4_TOPK=4` (opt-in turbo routing), `DSV4_PREFILL_NOCACHE` (default off, kept for reproduction), `DSV4_SUFFIX_APPEND` (batched conversation suffix append, batch is the default).
- `run_streaming.py` builds the pipenetwork model, quantizes the module tree to match the checkpoint, then swaps each layer's `SwitchGLU` experts for a `StreamingExperts` drop-in: gate → routed indices → `store.get` → per-expert quantized FFN.
- Hit-rate and byte counters are first-class; the first milestone of the project was the hit-rate-vs-cache-size curve, not tok/s.

### Engines

**Quality engine** — the in-process MLX stack above, exposed through `chat.py` and `serve.py`. `StreamingExperts` dispatches multi-token batches (prefill) through the verified `StackedStreamingExperts` `gather_qmm` path with parallel SSD reads; single-token decode is unchanged, and a regression check confirmed decode output is byte-identical to the pre-batching code. All MLX work — model load, seeding, prefill, decode — happens on one persistent generation-worker thread, because MLX streams are thread-local and lazy arrays are thread-affine (Lessons no. 13). The server keeps one resident conversation KV slot and appends suffixes in batched teacher-forced chunks ([Conversation KV reuse and batched suffix append](#conversation-kv-reuse-and-batched-suffix-append)).

**Fast engine** — antirez's ds4 runtime, unmodified, running the q2 GGUF with `--ssd-streaming --ssd-streaming-cache-experts 8GB` (the measured optimum). `serve.py` spawns its first-class `ds4-server` binary as a persistent child on a loopback port (8398) and proxies it; the child must run with `cwd=ds4-runtime/` because it loads `metal/*.metal` relative to CWD. It keeps the model resident between requests and reuses KV prefixes across turns. One global generation lock serializes across both engines — they contend for disk and page cache; memory coexistence is fine (~5 GB + ~18 GB).

### Model geometry that makes this viable

- 43 scored layers x 256 experts; 6 routed + 1 shared active per token → 258 routed visits/token, ~3.7 GB touched per cold token out of 165 GB.
- The first 3 layers are hash-routed by token id (`tid2eid`) — perfectly prefetchable before the forward pass even starts, and the pre-gated prefetch exploits exactly that (100% prediction accuracy on those layers).
- MLA-style attention keeps the KV cache around 1 GB at 32k context, so nearly the whole memory budget goes to the expert cache.

## Lessons for MLX streaming

1. **MLX arrays are functional; in-place writes into big pools are copies.** Any `setitem` on a stacked pool array materializes a copy of the whole array. A 1 GB per-layer pool costs ~1 GB of copy per inserted expert batch. This killed slot pools (experiment 1) and taxed optimistic repair (experiment 3). Cache experts as *individual* arrays and let a Python-side dict/LRU own the indexing; keep indirection out of the graph.
2. **Functional semantics cut the other way too: rollback is free — retracted, with a correction.** This lesson originally claimed that rebinding `cache.kv` to a new array leaves saved references naming the old one, making snapshot/restore of KV state just collecting and reassigning references. The spec-decode round proved the general form false: `mx.array.__setitem__` mutates the same object, so a saved reference does *not* protect you from in-place updates — optimistic rollback only worked because its redo rewrote identical values. Real snapshots need a copy (`x[:]`). Rebinding to a genuinely new array still behaves as described; the trap is assuming `setitem` creates one.
3. **Count syncs, not kernels.** Batching expert matmuls with `gather_qmm` changed nothing; collapsing 43 per-layer `mx.eval` round-trips into one was worth ~284 ms/token on paper. Per-layer graph breaks are the dominant Python-side cost in a routed-MoE loop.
4. **The Metal wired limit is a cliff, not a line — but check it is the cliff you actually hit.** Operating `gather_qmm` over 44 GB of wired pools at the raised ceiling turned a 0.41 s token into a ~7.5 s token, and staying under the limit with many small cached arrays behaved far better. The sweep adds a caveat to our own reading of that: plain v0 with no pools at all loses ~1.7x between 8 GB and 32 GB, so "the wired limit" and "having a lot of memory tied up" were confounded in every experiment here. Relatedly, llama.cpp's full-GPU path dies at a ~60 GB residency ceiling (`res=-3`) on 64 GB hardware.
5. **Hit rate is not a proxy for throughput — it can be anti-correlated with it.** Sweeping the expert cache from 4 GB to 32 GB raised the hit rate monotonically (0.434 → 0.669) while throughput peaked early and then fell (2.04 tok/s at 8 GB down to 1.23 at 32 GB). Every cache metric said the big cache was winning. Wall clock said it was losing by 1.67x. Size the cache by measured end-to-end throughput, on a realistic prompt, with replicates — nothing else.
6. **On macOS, an application-level file cache competes with a better one.** If reads go through `open()` without `F_NOCACHE`, the unified buffer cache is already caching the blobs, using all free RAM and with global visibility. An application LRU on top of that duplicates those bytes in wired memory, evicts the page cache that was serving the "misses", and pushes anonymous memory into swap. We spent the whole project trying to raise a hit-rate number that measured our redundant copy. The general form: before building a cache, find out whether the OS is already running one for you, and make the two mutually exclusive rather than stacked.
7. **Swap ratchets; it does not spring back.** Large-cache runs took swap from 1520 MB to 6657 MB and it stayed near 7 GB for hours and 14 subsequent runs. A single badly-sized run therefore contaminates every measurement after it. Bench in fresh processes, sweep in both directions, and treat the first pass through a new configuration as suspect — our forward pass was uniformly slow at every budget for exactly this reason.
8. **One timing measurement is not a measurement.** A run that should take 11.7 s took 726.4 s, at a small cache size, with no memory pressure, and never reproduced across three immediate replicates. We had previously built a whole causal story (the "swap cliff") on a single pair of runs that showed a 17x difference. Replicates would have prevented it. On a laptop, other things — indexers, backup daemons, whatever else the OS decided to do — are part of the experiment whether you model them or not. The eventual culprit here was more mundane than any of those guesses: the machine was asleep (`pmset log` confirmed) while the wall clock kept counting. `caffeinate` is part of the benchmark procedure now.
9. **A repetitive prompt flatters every cache metric.** The same code shows an 80% hit rate with a working set that fits in cache on `"The capital of France is"`, and a 75% hit rate with a working set 1.5x larger than any available cache on an ordinary question. Benchmark prompts have to be chosen for their expert diversity, not their convenience, or the cache curve measures the prompt.
10. **Global LRU beats per-layer partitioning.** Per-layer working sets in a real MoE are skewed (a nearly 3x spread across layers here); any uniform per-layer allocation simultaneously starves hot layers and wastes slots on cold ones.
11. **Explicit reads beat mmap.** 13.3 GB/s from SSD with `pread` on contiguous page-aligned blobs; fault-driven paths SIGBUS or stall. Fixed-stride blob layout is what makes the read path this simple and this fast. Note the 13.3 GB/s was measured with `F_NOCACHE` and the production store does not set it — which, per lesson 6, turned out to be the most consequential line in the store.
12. **The first caller with a genuinely warm cache found a bug three prior workloads had dodged.** `expert_store.get_many` inserted a batch's misses — triggering eviction — before touching the batch's own cached hits, so on a warm full cache it could evict a blob the same batch was about to use: silently omitted from the result, `KeyError` downstream. Cold prefills and the zero-hit 1024-token perplexity windows never exercised the path; the server's first warm repeat hit it immediately. Fix: `move_to_end` the batch's hits before insertion (commit `07d1f29`).
13. **MLX streams are thread-local, and lazy arrays are thread-affine.** An evaluated array crosses threads fine; an array whose lazy graph nodes were recorded on thread A cannot be evaluated from thread B once A is gone ("There is no Stream(gpu, N) in current thread" → uncaught C++ exception → abort). The server's resident KV slot held exactly such lazily-updated arrays between requests, and the second chat turn killed the process. Two working patterns: give one persistent thread ownership of all MLX work (the server's fix), and keep async helpers to `pread` + numpy with mx arrays created on the main thread only (the prefetch's rule).

## Limitations

- **2.4 tok/s is the Python ceiling with a fully cached working set, and this is a Python prototype.** The 284 ms/token of sync overhead requires a compiled runtime to remove; the no-op-expert ceiling on this hardware is 7.8 tok/s. Note that the realistic-prompt result (2.04 tok/s) is now within ~16% of the repetitive-prompt warm number, so sync overhead — not cache behaviour — is again the dominant cost to attack. The compiled runtime this argues for already exists — that is the fast engine.
- **The fast engine is a different quality point.** The 11.4 tok/s headline runs ds4's q2 (2-bit imatrix) GGUF. The mixed 4/8-bit quality engine is the one whose perplexity matches the published number to -0.02%; ds4 cannot load that checkpoint format.
- **The sweep is one prompt at 24 tokens.** Every number in [Cache-size sweep](#cache-size-sweep-realistic-prompt--the-main-result) comes from `"What is the tallest mountain in the world, and how tall is it?"` generating 24 tokens. The optimum cache size is a function of the workload, and a longer generation, a batch, or a long context (which grows the KV cache) could all move it. The shape of the result — throughput falling as the cache grows — is unlikely to invert, but the specific 8 GB figure should not be treated as universal.
- **The sweep's mechanism is inferred, not instrumented.** The page-cache explanation is consistent with the code (`expert_store.py` sets no `F_NOCACHE`) and with all 19 runs, but nothing measured `pread` service times or compared against an `F_NOCACHE` build of the store. An alternative explanation that also fits — MLX allocator or wired-memory overhead scaling with pool size — has not been ruled out. The law's replication in ds4 ([The ds4 benchmark on this machine](#the-ds4-benchmark-on-this-machine)) is corroboration, not instrumentation.
- **The extreme stalls are explained — the machine was sleeping.** `pmset log` confirmed sleep during the stalled runs; long runs must use `caffeinate`. (The Spotlight hypothesis remains untested and is now secondary.) The long-prefill decode slowdown is also now explained — two cache-side hypotheses falsified, then traced to I/O bandwidth ([The verdict](#the-verdict-the-collapse-is-io-bandwidth)). Any single timing measurement on this machine should still be replicated before it is believed.
- **The diverse eval is complete but single-pass.** All 8 domains ran end to end at the recommended 8 GB ([Diverse-workload results](#diverse-workload-results-8-prompts-8-gb-cache), commit `a90aa44`), but it is one run per prompt, 48 tokens each, in one fixed prompt order. The cross-prompt warming curve is a function of that order, the per-domain throughput figures have no replicates — and single timing measurements on this machine have been 62x off before — and the translation prompt self-terminated at 32 tokens, so its row is a shorter generation than the others.
- **Perplexity validation is complete.** Decoding is byte-correct against the reference implementation and output text was byte-identical across all 19 sweep runs; the streaming-path perplexity number is directly comparable and final: 6.1250 vs the published 6.1262 (-0.02%) over all 195 windows ([Perplexity through the streaming path](#perplexity-through-the-streaming-path-wikitext-2)). Chat output is additionally validated at the encoding level: all 4 bundled `encoding_dsv4` test vectors pass byte-exact.
- **Prefill is batched — and 2x faster since fill-no-evict — but decode after a long prefill is still slow.** Multi-token prefill goes through the `gather_qmm` batch path (~2.2 tok/s on a 315-token prompt — minutes, not hours; 76.3 s → 36–39 s after the no-evict change). Decode afterwards still drops to ~0.6 tok/s; the mechanism is identified as I/O bandwidth, not a bug ([The verdict](#the-verdict-the-collapse-is-io-bandwidth)), and the fixes it points to — fewer bytes per token, or routing-ahead prefetch with async staging — live in the compiled-runtime territory the fast engine occupies.
- **Prefetch exists but stays off by default.** Pre-gated prefetch (`DSV4_PREFETCH=1`) predicts the hash-routed layers at 100% and scored layers at 70–77%, and buys +8% short-form decode — but long-form it adds +107% read traffic for no gain ([Pre-gated expert prefetch](#pre-gated-expert-prefetch-8-short-form-only-default-off)). Demand-driven reads remain the default.
- **The webapp is single-user.** One global generation lock serializes across both engines, and there is one quality-engine KV slot, so interleaved quality conversations rebuild each other's cache. ds4-server has its own multi-prefix KV store and does not suffer this. Voice input and images are absent (no whisper-cpp; text-only model).
- **Wired-limit fragility — now avoidable, and verified.** Cache budgets above ~40 GB depend on `iogpu.wired_limit_mb`, which resets on reboot. The recommended 8 GB configuration does not touch it: tested directly with the limit at its macOS default (`iogpu.wired_limit_mb: 0`) — correct output, 9.2 s prefill, 0.81 tok/s decode, 18.2 GB peak, indistinguishable from the raised-limit runs. The sysctl is only relevant to reproducing the large-cache experiments.
- Known model-level issue independent of this repo: V4-Flash leaks DSML markers / repeats on markup-heavy agentic prompts even via the official API (llama.cpp#26694).

## Upstream engagement

**The recon that changed the plan** (`docs/ds4-recon.md`). This project's quantified case for a compiled runtime pointed at a Swift/Metal port — until reconnaissance found the endgame already exists: antirez's [ds4](https://github.com/antirez/ds4) (C/Metal/CUDA) ships `--ssd-streaming` at ~5 tok/s decode / ~110 tok/s prefill on a 64 GB M1 Ultra at q2 — 2.5x our decode and ~50x our prefill, on a lower-quality quant. A from-scratch port would re-derive its pread pools, async staging and slab cache from a standing start. The plan flipped to **contribute, don't port**: bring our measurement discipline upstream, and keep this repo as the measurement lab and the only runner of the mixed 4/8-bit checkpoint (ds4 cannot load our format). Three things worth stealing were noted — async expert staging under the shared-expert compute (their answer to our 43-sync gap), prefill-never-evicts (tested here, see [LRU protection](#lru-protection-prefill-no-evict-rejected-with-an-accidental-2x-prefill-win)), and hotness-with-decay eviction for our measured ~7x-refetch regime. Their 32 GB large-cache advice contradicted our 8 GB optimum, so the sweep was rerun under ds4 rather than assumed — and the 8 GB optimum held ([The ds4 benchmark on this machine](#the-ds4-benchmark-on-this-machine)). Their CUDA streaming cache is currently inert (upstream issue #639); the Metal path is the mature one.

**Contribution filed: [antirez/ds4#810](https://github.com/antirez/ds4/issues/810)** — the M5 Pro 64 GB `--ssd-streaming` datapoint and the small-cache finding, written in ASD-STE-100 per the project's posting style rules. The final issue text is archived in `docs/upstream-ds4-issue.md`, including the offer of the sweep methodology, the per-layer working-set skew data (53–156 distinct experts per layer), and the cross-domain reuse data (9,344 of 11,008 experts, ~7 reads per distinct expert) against their auto-budget planner.

**Blocked upstream: DSpark + `--ssd-streaming`.** ds4 at `84cc882` refuses the combination ("--ssd-streaming is not compatible with --mtp yet") while their README documents it as supported on Metal — a README/code mismatch, reported with the logs in `logs/ds4_dspark.txt`. The 64 GB streaming + speculation datapoint — the configuration our own spec-decode autopsy says could actually win — is unmeasurable until upstream wires it. Same-day controls were banked (9.00 tok/s short / 8.88 tok/s long at the 8 GB cache), consistent with the benchmark above.

## Repo map

| Path | What lives there |
|---|---|
| `streaming/` | The quality engine and everything around it: `repack.py` (checkpoint repacker), `expert_store.py` (LRU + pread store), `run_streaming.py`, `chat.py` (terminal REPL), `serve.py` + `webui/` (chat webapp and OpenAI-compatible API), `kv_append.py` (batched suffix append), `ppl_streaming.py` (perplexity harness), `multi_prompt_eval.py` (diverse eval), experiment code (`spec_generate.py`, `optimistic.py`, `bench_layer0.py`) and tests (`test_roundtrip.py`, `test_prefetch.py`) |
| `aqlm/` | The 2-bit codebook quality gate: `aqlm_lib.py`, `run_arm.py`, `validate_one.py` (`out_*` rebuilt-layer evidence is untracked, ~65 GB, deletable) |
| `docs/` | `ds4-recon.md` (the ds4 reconnaissance), `upstream-ds4-issue.md` (the filed issue text), `evidence-swap-cliff.md` (superseded swap-cliff evidence, kept with its correction) |
| `patches/` | `deepseek-v4-mlx-token-hook.patch` — the 2-line token hook the prefetch rides on, for the gitignored model-port clone |
| `logs/` | Raw measurement logs behind every number in this README (a curated set is tracked; scratch runs stay local) |

Local working directories that are gitignored: `repacked/` (the repacked checkpoint), `data/` (webapp sqlite), `ds4-runtime/` (the ds4 clone for the fast engine), `deepseek-v4-mlx/` (the model-port clone), and `bench/`.
