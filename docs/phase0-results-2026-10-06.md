# Phase 0 results, 2026-10-06: DFlash2-G against incoai

Two Sparks, `PARALLEL=4`, measured with `tools/phase0_bench.py` from another machine on the network (one boot per
run). Prompts: 8 prose and 8 code, `max_tokens` 1024; "at once" 1 sends the first 4 one at a time, 4 sends all 8
through 4 workers.

| run | category | at once | agg tok/s | per request tok/s | tokens/round |
|---|---|---:|---:|---:|---:|
| g-w2048 | prose | 1 | 42.1 | 42.83 | 1.757 |
| g-w2048 | prose | 4 | 74.77 | 21.5 | 1.854 |
| g-w2048 | code | 1 | 62.99 | 64.01 | 3.237 |
| g-w2048 | code | 4 | 102.81 | 27.14 | 3.12 |
| g-w4096 | prose | 1 | 42.24 | 42.83 | 1.757 |
| g-w4096 | prose | 4 | 75.03 | 21.51 | 1.854 |
| g-w4096 | code | 1 | 63.01 | 63.82 | 3.237 |
| g-w4096 | code | 4 | 102.91 | 27.08 | 3.12 |
| incoai | prose | 1 | 46.28 | 46.87 | 2.0 |
| incoai | prose | 4 | 78.2 | 23.05 | 2.132 |
| incoai | code | 1 | 73.9 | 75.11 | 4.433 |
| incoai | code | 4 | 114.49 | 30.04 | 4.295 |

Identity (drafted == serial): 4 of 4

Gate (>= 90% of incoai at 4 requests, prose and code; replies identical):
- g-w2048: prose 95.6%, code 89.8% of incoai -> FAIL
- g-w4096: prose 95.9%, code 89.9% of incoai -> FAIL

**GO by the operator's decision** (the gate's NO-GO is 0.1 points short on code; G is Apache-2.0, incoai
non-commercial). `DFLASH_WINDOW` stays 2048.

## Notes

- The window did not matter here: every request's context stays under ~1,100 tokens, inside all the windows, so
  g-w2048 and g-w4096 are the same run twice (g-w8192 and g-full were skipped). What a window costs on long contexts
  is still unmeasured.
- The gap is G's acceptance: 3.12 tokens a round on code against incoai's 4.30, 1.85 against 2.13 on prose.
- Repeatable to ~0.3% across restarts. The published v1.8 image with incoai, same script, gave 46.21 / 78.31 /
  74.18 / 114.69 tok/s (prose 1, prose 4, code 1, code 4), the same as this branch's image: patches 0084 and 0085
  cost incoai nothing.
- These numbers are below the README's sparkDash ones (e.g. 108.8 tok/s prose at 4) for the method and prompts:
  aggregate here is all tokens over the batch's wall time, tail and prefill included.

## Long context (`--long`)

`tools/phase0_bench.py <label> --long`: this repo's README/CHANGELOG (prose) and scripts (code) cut to 7.7k-18.5k
tokens, each task asking about the start, beyond a 2048-token window. g-full is `PARALLEL=1 DFLASH_WINDOW=0
CONTEXT=131072`. Aggregate tok/s includes the long prompts' prefill; per request is decode only.

| run | category | at once | agg tok/s | per request tok/s | tokens/round |
|---|---|---:|---:|---:|---:|
| g-full | prose | 1 | 40.55 | 49.25 | 2.389 |
| g-full | code | 1 | 32.6 | 46.97 | 2.187 |
| g-w2048 | prose | 1 | 41.06 | 49.68 | 2.483 |
| g-w2048 | prose | 4 | 70.75 | 20.6 | 2.333 |
| g-w2048 | code | 1 | 32.73 | 47.78 | 2.312 |
| g-w2048 | code | 4 | 54.23 | 19.04 | 2.334 |
| g-w4096 | prose | 1 | 40.93 | 49.6 | 2.486 |
| g-w4096 | prose | 4 | 70.42 | 20.56 | 2.325 |
| g-w4096 | code | 1 | 32.3 | 47.44 | 2.294 |
| g-w4096 | code | 4 | 53.5 | 19.09 | 2.318 |
| incoai | prose | 1 | 48.11 | 60.04 | 3.326 |
| incoai | prose | 4 | 80.47 | 23.62 | 3.147 |
| incoai | code | 1 | 36.03 | 55.03 | 2.971 |
| incoai | code | 4 | 57.64 | 21.07 | 2.967 |

Gate: g-w2048 prose 87.9%, code 94.1% of incoai; g-w4096 prose 87.5%, code 92.8%.

- The window costs G nothing: 2048 and 4096 are within noise, and full attention is slightly worse (2.39 against
  2.48 tokens a round on prose; G was trained on <= 4,096-token completions). `DFLASH_WINDOW=2048` stands.
- On long contexts G drafts ~25% fewer tokens a round than incoai (which has its own 2048 window): ~12% slower on
  prose, ~6% on code at 4 requests. A larger window does not close that; draft-policy tuning (Phase 1) or a drafter
  trained on longer completions would have to.
