# tfcap: admission control and delivery-abort patches

Two patches for the TensorFold serving stack. `0099` adds admission control to
`Scheduler`; `0100` makes the scheduler relay abort a stream when its delivery
callback fails. Both are measured against vLLM v1 behavior and are verified live
on a 3-rank TP3 deployment.

## Problem

**No admission control.** `Scheduler.submit` enqueues unconditionally. Under
saturation the waiting queue grows without signal: clients hold open connections
until their own timeouts, then retry and stack. Measured on this deployment:
`requests_waiting=3` with a 5th request served `HTTP 200` while four lanes ran
dead requests, followed by 13 client retries of ~360 s each.

**Delivery failures leak lanes.** The relay loop calls `emit(value)` — the
request's own delivery callback — as

```python
if not cancel[0] and emit(value):
    cancel[0] = True
```

An exception from `emit` (broken pipe, reset, timeout) escapes the relay thread,
escapes `submit`, and the stream remains admitted until generation ends —
minutes of decode into a queue nobody reads.

**Capacity refusals render inconsistently.** `CapacityError` is rendered as a
bare `503` with no `Retry-After` at two sites and as `400` at a third
(`CapacityError` is a `RequestError` subclass caught by a later branch).

## Reference behavior (vLLM v1)

- `AsyncLLM.check_admission`: every request passes an admission check; the limit
  is configuration (`max_num_queued_reqs`, `max_num_queued_tokens` default
  unlimited); refusals return a retryable status with a retry hint "so that load
  balancers and client SDKs retry".
- `AsyncLLM.generate`: `except (asyncio.CancelledError, GeneratorExit):
  await self.abort(request_id, internal=True)` reaches "OutputProcessor and
  EngineCore" — a request is aborted on every error path.
- `RequestOutputCollector.put` is non-blocking and merges outputs when the
  producer outruns the consumer: a connected-but-slow reader never blocks the
  engine and never frees its lane. No timeout is involved.

## Change

### `server/errors.py`

- `CapacityError` docstring: states the rendered status (429 + Retry-After).
- New `admission_response(exc) -> tuple[int, dict[str, str] | None]` — the
  status/headers pair for a capacity refusal, defined once:

  ```python
  return (429, {"Retry-After": "5"}) if isinstance(exc, CapacityError) else (400, None)
  ```

  Deviation from vLLM: 429 rather than 503. The deployment is a single instance
  on a LAN; 429 pressures the same client to back off. Both integrated clients
  were verified against it (opencode retries 429/≥500 honoring `Retry-After`;
  the Anthropic bridge maps 429 → 529 `overloaded_error` + `Retry-After`).

### `cuda/scheduler.py`

- `Waiting.foreground_count() -> int` — queued background streams under
  `self.mutex`; they are scheduled last and never take a lane from a foreground
  request, so only foreground waiters count against the cap. The existing
  `foreground()` (`-> bool`) is unchanged; its callers are unchanged.
- `Scheduler.__init__(..., max_in_system: int | None = None)` — the in-system
  cap (lanes + foreground-queued). `None`, the default, is the pre-patch
  behavior: every scheduler queues, as before. The mechanism is unconditional;
  the limit is configuration, matching vLLM where `check_admission` always runs
  and the caps default to unlimited.
- `Scheduler._check_admission(background)`:

  ```python
  if background or self.max_in_system is None:
      return
  if self.decoder.live() + self.waiting.foreground_count() >= self.max_in_system:
      raise CapacityError("all serving lanes are busy; retry shortly")
  ```

- `Scheduler._relay_token(emit, value, cancel)` — the delivery policy:

  ```python
  stop = False
  if not cancel[0]:
      try:
          stop = emit(value)
      except Exception as exc:
          print(f"[tensorfold] stream aborted: emit failed: {exc!r}", flush=True)
          stop = True
  if stop:
      cancel[0] = True
  ```

  Broad `except Exception` is deliberate: `emit` is a delivery callback; any
  exception means the token was not delivered. The abort fires at most once per
  stream (`cancel` latches; later tokens skip `emit`), so the log line cannot
  flood. `scheduler.py` previously printed nothing — this is its first line,
  and an aborted delivery is otherwise unobservable.
- `submit` calls `self._check_admission(background)`; the relay calls
  `self._relay_token(emit, value, cancel)`.

### `families/glm5_next/cuda/multi.py`

- `GlmScheduler.__init__` passes `max_in_system=max_streams` — the GLM
  multi-lane policy: no stacking beyond the lanes. This constructor argument is
  the entire family-scoped surface; the policy logic is inherited.
- `GlmScheduler.submit` calls `self._check_admission(background)`; the GLM
  relay calls `self._relay_token(emit, value, cancel)` (the class re-implements
  both, so each is a one-line call into the shared definitions).

### `server/http.py` and `cuda/http.py`

- `_send_json` / `_json` gain an optional `headers` kwarg (additive; existing
  call sites unchanged).
- The three `isinstance(exc, CapacityError)` render sites route through
  `admission_response(exc)`; `server/http.py`'s generate path adds
  `except CapacityError` before `except RequestError` (a `CapacityError` is a
  `RequestError` subclass; without the earlier branch it renders as 400).

## What is intentionally absent

- No timeout: a connected-but-stalled reader holds its lane exactly as in vLLM
  (coalescing on their side; the request thread blocks here while the scheduler
  keeps decoding). `timeout=0` in the patched tree.
- No submit lock: a burst arriving entirely inside the lane-fill window can
  admit one request past the cap (observed once; served normally, queue
  drained). vLLM does not race here because asyncio serializes
  `check_admission`. A lock is not worth the contention for a bounded,
  self-draining effect.
- No consolidation of the two HTTP layers — `server/http.py` and
  `cuda/http.py` duplicate their render paths. The policy lives in
  `errors.py` so the layers cannot drift; consolidation is a separate refactor.

## Measured results (3-rank TP3, `--parallel 8`, final build)

| Probe | Result |
|---|---|
| 9th request, 8 lanes busy — direct OpenAI wire | HTTP 429 in 1.6 ms, `Retry-After: 5` |
| 9th request — Anthropic bridge (non-streaming) | HTTP 529, `retry-after: 5`, `type: overloaded_error` |
| 12-request burst at idle | 8 lanes + 3×429 + 1 admitted inside the lane-fill window; queue drained to 0 |
| Post-burst drain | `requests_running 0, requests_waiting 0` |
| Dead client mid-decode | lane freed within one 15 s window (guard fires at the next token write) |
| Stalled reader | lane held to completion, as vLLM; `timeout=0` in the patched tree |

## Complexity and duplication impact

| Function | CC before | CC after |
|---|---|---|
| `Scheduler.submit` | 6 | 4 |
| `GlmScheduler.submit` | 7 | 5 |
| `do_POST` (cuda layer) | 42 | 40 |
| `make_handler` (server layer) | 93 | 95 |
| new `_check_admission` / `_relay_token` / `admission_response` | — | 4 / 4 / 2 |

Hardcoded capacity statuses: 3 sites → 0. `429` literals: 3, all in
`errors.py`. One definition per policy; call sites are single lines.

## Test

Static: both patches dry-apply on a pristine tree; `TFCAP-REJECT-v1` markers in
5 files, `TFCAP-0100` in 1; import graph passes.

Rollback: remove the two patch files, rebuild.
