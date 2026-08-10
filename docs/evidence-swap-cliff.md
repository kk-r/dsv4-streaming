# Diverse eval, first attempt — transcribed evidence

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

Swap was in use with no model process running, i.e. the runs had driven the
machine into swap and it had not been reclaimed. This is the basis for the
"size the cache for OS headroom, not for capacity" conclusion. The cache-size
sweep (`logs/cache_sweep.*`) tests it directly and supersedes this note.
