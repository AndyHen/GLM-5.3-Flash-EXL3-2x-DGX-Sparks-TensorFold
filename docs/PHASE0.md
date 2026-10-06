# Phase 0: DFlash2-G against incoai on two Sparks

Goal: G's aggregate decode at its best window >= 90% of incoai's at PARALLEL=4 (prose and code), and drafted
replies identical to serial ones. Spec: the planning repo's
`docs/superpowers/specs/2026-10-06-dflash2g-tensorfold-design.md`.

## Before

1. Stop whatever else uses the Sparks' GPUs (each rank needs ~110 GiB free).
2. On the head: `git clone https://github.com/AndyHen/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold && cd
   GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold && git checkout dflash2g`, and copy `scripts/local.sh` from your
   existing Mia checkout (WORKER, FABRIC_PEER, ...).
3. `scripts/prepare.sh` once: builds the image from our 84 patches (no published image yet), downloads G (6.2 GB)
   and copies it to the worker. The incoai run downloads incoai the same way.

## Run

    tools/phase0.sh                       # all five runs, ~1-1.5 h
    RUNS="incoai g-w2048" tools/phase0.sh # a subset

Results: `~/phase0-results/<date>-<time>/report.md` (+ one JSON per run, + each run's start log).

## What to look for

- Each G run prints `Loaded DFlash mask embedding for mask_token_id 154856 from mask_embedding.pt` and
  `the drafter's embed_tokens equal the target's` (from rank 0's container log).
- `tokens/round` (mean tokens a round commits): incoai's is the reference; G's should be within a few percent.
  Far lower (< 2) means a semantic mismatch (taps, mask) — stop and investigate before tuning.
- `g-full` vs `g-w2048` tokens/round: what the window costs G's acceptance.
- `Identity: N of N`. Any mismatch is a bug in the patches (drafts must never change replies).
- The report's last line: GO / NO-GO.

## After

Set `DFLASH_WINDOW`'s default in `scripts/config.sh` to the best passing window (and `dev/test_config.sh`'s expected
value); then Phase 1 (TP=3, PARALLEL=8, draft policy tuning) gets its own plan.
