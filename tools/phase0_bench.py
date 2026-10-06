#!/usr/bin/env python3
"""Phase 0 of DFlash2-G: prose and code prompts at 1 and N concurrent requests against the running server, each run's
tokens, seconds and the server's draft stats (the reply's "tensorfold" block) to <out>/<label>.json; --identity also
checks drafted replies against "draft": false ones (token_sha); --report <out> writes report.md and the go/no-go gate.

  tools/phase0_bench.py <label> [--concurrency 1,4] [--identity] [--out DIR]
  tools/phase0_bench.py --report DIR
API_URL as in tools/client.py."""
import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client import URL, open_url  # noqa: E402

MAX_TOKENS = 1024
SEED = 20261006
PROMPTS = {
    "prose": [
        "Write a short story about a lighthouse keeper who finds a message in a bottle.",
        "Explain to a curious teenager how vaccines train the immune system.",
        "Describe a busy morning market in a coastal town, with sounds and smells.",
        "Write a persuasive essay on why cities should plant more trees.",
        "Summarize the causes and consequences of the printing press for European society.",
        "Write a letter from a retired astronaut to their younger self.",
        "Compare living in a small village with living in a large city.",
        "Explain how a bill becomes law in a parliamentary democracy.",
    ],
    "code": [
        "Write a Python function that parses an ISO 8601 duration string into seconds, with tests.",
        "Implement an LRU cache class in Python with get and put in O(1), and explain the design.",
        "Write a TypeScript function that deep-merges two JSON objects, with unit tests.",
        "Write a Rust function that tokenizes arithmetic expressions and evaluates them.",
        "Implement Dijkstra's shortest path in Go for a graph given as an adjacency list.",
        "Write a bash script that rotates log files older than 7 days and compresses them.",
        "Write a SQL schema for a library system with books, members and loans, plus three queries.",
        "Write a Python asyncio crawler that fetches URLs with a concurrency limit of 5.",
    ],
}


def ask(prompt: str, *, draft: bool = True) -> dict:
    body = {"model": os.environ.get("MODEL", "GLM-5.3-Flash-EXL3"), "max_tokens": MAX_TOKENS, "seed": SEED,
            "messages": [{"role": "user", "content": prompt}]}
    if not draft:
        body["draft"] = False
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with open_url(req, timeout=1800) as resp:
        reply = json.load(resp)
    stats = reply.get("tensorfold") or {}
    return {"tokens": int(reply["usage"]["completion_tokens"]), "seconds": time.perf_counter() - t0,
            "decode_s": float(stats.get("decode_s") or 0.0), "tokens_per_round": float(stats.get("tokens_per_round") or 0.0),
            "drafted": int(stats.get("drafted") or 0), "accepted": int(stats.get("accepted") or 0),
            "token_sha": stats.get("token_sha")}


def batch(prompts: list[str], concurrency: int) -> tuple[list[dict], float]:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(ask, prompts))
    return rows, time.perf_counter() - t0


def summarize(rows: list[dict], wall: float) -> dict:
    tokens = sum(r["tokens"] for r in rows)
    rates = [r["tokens"] / r["decode_s"] for r in rows if r["decode_s"] > 0]
    rounds = [r["tokens_per_round"] for r in rows if r["tokens_per_round"] > 0]
    return {"requests": len(rows), "tokens": tokens, "wall_s": round(wall, 3),
            "agg_tok_s": round(tokens / wall, 2) if wall > 0 else 0.0,
            "per_req_tok_s": round(sum(rates) / len(rates), 2) if rates else 0.0,
            "tokens_per_round": round(sum(rounds) / len(rounds), 3) if rounds else 0.0}


def gate(runs: dict, reference: str = "incoai", threshold: float = 0.9) -> tuple[bool, list[str]]:
    """Go when some G run reaches threshold x the reference's aggregate tok/s at 4 concurrent requests, prose and code."""

    lines, passed = [], False
    ref = runs.get(reference)
    if ref is None:
        return False, [f"no '{reference}' run to compare with"]
    for label, run in runs.items():
        if label == reference:
            continue
        try:
            ratios = {c: run[c]["4"]["agg_tok_s"] / ref[c]["4"]["agg_tok_s"] for c in ("prose", "code")}
        except (KeyError, ZeroDivisionError):
            lines.append(f"{label}: skipped (no 4-request runs of both categories)")
            continue
        ok = all(r >= threshold for r in ratios.values())
        passed |= ok
        lines.append(f"{label}: prose {ratios['prose']:.1%}, code {ratios['code']:.1%} of {reference} -> "
                     f"{'PASS' if ok else 'FAIL'}")
    return passed, lines


def same_reply(a: str | None, b: str | None) -> bool:
    """Two replies' token hashes match; a reply without one (no "tensorfold" stats) never counts as the same."""

    return a is not None and b is not None and a == b


def identity(prompts: list[str]) -> list[dict]:
    out = []
    for p in prompts:
        drafted, serial = ask(p), ask(p, draft=False)
        out.append({"prompt": p[:60], "same": same_reply(drafted["token_sha"], serial["token_sha"]),
                    "drafted": drafted["token_sha"], "serial": serial["token_sha"]})
    return out


def report(out: Path) -> int:
    runs = {p.stem: json.loads(p.read_text()) for p in sorted(out.glob("*.json"))}
    md = ["| run | category | at once | agg tok/s | per request tok/s | tokens/round |", "|---|---|---:|---:|---:|---:|"]
    for label, run in runs.items():
        for cat in ("prose", "code"):
            for conc, s in sorted((run.get(cat) or {}).items()):
                md.append(f"| {label} | {cat} | {conc} | {s['agg_tok_s']} | {s['per_req_tok_s']} | {s['tokens_per_round']} |")
    ids = [(label, r) for label, run in runs.items() for r in run.get("identity", [])]
    if ids:
        md += ["", f"Identity (drafted == serial): {sum(r['same'] for _, r in ids)} of {len(ids)}"]
        md += [f"- {label}: MISMATCH on '{r['prompt']}'" for label, r in ids if not r["same"]]
    ok, lines = gate(runs)
    ok = ok and all(r["same"] for _, r in ids)
    md += ["", "Gate (>= 90% of incoai at 4 requests, prose and code; replies identical):"] + [f"- {l}" for l in lines]
    md += ["", f"**{'GO' if ok else 'NO-GO'}**"]
    (out / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("label", nargs="?")
    ap.add_argument("--concurrency", default="1,4")
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--out", default=os.path.expanduser("~/phase0-results/manual"))
    ap.add_argument("--report")
    a = ap.parse_args()
    if a.report:
        return report(Path(a.report))
    if not a.label:
        ap.error("a run label, or --report DIR")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    result: dict = {}
    for cat, prompts in PROMPTS.items():
        result[cat] = {}
        for conc in (int(c) for c in a.concurrency.split(",")):
            todo = prompts[:4] if conc == 1 else prompts
            rows, wall = batch(todo, conc)
            result[cat][str(conc)] = {**summarize(rows, wall), "rows": rows}
            print(f"{a.label} {cat} x{conc}: {result[cat][str(conc)]['agg_tok_s']} tok/s aggregate, "
                  f"{result[cat][str(conc)]['tokens_per_round']} tokens/round", flush=True)
    if a.identity:
        result["identity"] = identity(PROMPTS["prose"][:2] + PROMPTS["code"][:2])
        print(f"{a.label} identity: {sum(r['same'] for r in result['identity'])} of {len(result['identity'])} same")
    (out / f"{a.label}.json").write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
