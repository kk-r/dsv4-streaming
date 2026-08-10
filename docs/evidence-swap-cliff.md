# Diverse eval, first attempt — transcribed evidence

> ## SUPERSEDED
>
> **The conclusion this note was written to support is wrong, and it is kept only
> as a record of the raw observations.** It argued that a 40 GB expert cache
> pushed the machine into swap and that this caused the 17x wall-clock
> difference below. The cache-size sweep (`logs/cache_sweep.txt`,
> `logs/cache_sweep.json`, 19 runs) tested that directly:
>
> - **The stalls are not caused by cache size.** The sweep's worst stall — 726.4 s
>   for work that takes ~11.7 s, a 62x collapse — occurred at an **8 GB** cache
>   with a 17.7 GB peak, nowhere near a memory ceiling. Three replicates at 8 GB
>   immediately afterwards returned 2.04 / 2.03 / 2.08 tok/s. Extreme stalls are
>   a sporadic system-level I/O phenomenon that can strike any cache size.
> - **The swap reading below is not evidence of anything.** The 1527.62 MB
>   measured after these runs is approximately this machine's idle baseline: the
>   sweep started at 1519.62 MB used with no model process running, and swap held
>   at that level through every 4, 8 and 16 GB run. It was not produced by the
>   40 GB runs.
> - **Swap did move for genuinely large caches, and permanently.** In the sweep,
>   the 24 GB run took swap 1520 → 3591 MB and the 32 GB run took it to 6657 MB,
>   after which it never recovered. That part of the original intuition holds.
> - **Large caches are reliably slower**, by ~1.67x from 8 GB to 32 GB, which the
>   sweep's medians establish across 19 runs rather than the two here.
>
> The surviving methodological point — size the cache by measured end-to-end
> throughput, not by hit rate or by what the wired limit permits — is unchanged
> and better supported. The causal story attached to it was not.
>
> Corrected discussion: README, *Cache size, swap, and sporadic stalls*.

**Provenance note:** these lines are transcribed from the interactive session
transcript of 2026-08-09, not captured from a file. The first attempt and the
retry both wrote to `logs/diverse_run.txt`, so the retry overwrote this run's
log. Recorded here because the pair of runs is the only evidence for the swap
cliff (identical work, 17x wall-clock difference). Anything downstream should
treat these as second-hand until re-measured.

## First attempt (killed after prompt 1), `--cache-gb 40`

```
[load] resident weights in 0.8s (9.4 GB active)
[1/8] factual     48 tok in 55.0s -> 0.87 tok/s | hit 0.747 (11953h/4043m, 57.2 GB) | cache 40.0 GB
```

## Retry (killed after prompt 2), `--cache-gb 40` — this one is in diverse_run.txt

```
[1/8] factual     48 tok in 928.8s -> 0.05 tok/s | hit 0.747 (11953h/4043m, 57.2 GB) | cache 40.0 GB
[2/8] coding      48 tok in 1933.7s -> 0.03 tok/s | hit 0.757 (12698h/4072m, 57.6 GB) | cache 40.0 GB
```

Prompt 1 is byte-identical between the two runs in every counter — same hits,
same misses, same bytes read — and differs 55.0s vs 928.8s in wall clock.

## System state, measured after both runs had exited

```
$ vm_stat  (derived)
free: 14.07 GB | compressor: 0.61 GB

$ sysctl vm.swapusage
vm.swapusage: total = 3072.00M  used = 1527.62M  free = 1544.38M  (encrypted)
```

~~Swap was in use with no model process running, i.e. the runs had driven the
machine into swap and it had not been reclaimed.~~ **Wrong** — see the
superseded banner at the top. 1527.62 MB is this machine's idle baseline, not a
residue of these runs; the sweep began at 1519.62 MB with nothing running. The
inference drawn from this number was unsound.

What remains from this note: two runs of byte-identical work, 55.0 s vs 928.8 s,
which is a real and unexplained 17x. The sweep reproduced the phenomenon (62x)
but at an 8 GB cache, so cache size is not the explanation. One untested
hypothesis is Spotlight indexing `repacked/`; `repacked/.metadata_never_index`
has since been added, without a controlled measurement of its effect.
