import importlib.util
import sys

from conftest import ROOT

sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location("phase0_bench", ROOT / "tools" / "phase0_bench.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_summarize():
    rows = [{"tokens": 100, "seconds": 4.0, "decode_s": 2.0, "tokens_per_round": 3.0},
            {"tokens": 300, "seconds": 6.0, "decode_s": 3.0, "tokens_per_round": 4.0}]
    s = bench.summarize(rows, wall=8.0)
    assert s == {"requests": 2, "tokens": 400, "wall_s": 8.0, "agg_tok_s": 50.0, "per_req_tok_s": 75.0,
                 "tokens_per_round": 3.5}


def _run(prose, code):
    return {"prose": {"4": {"agg_tok_s": prose}}, "code": {"4": {"agg_tok_s": code}}}


def test_gate_passes_on_best_window():
    ok, lines = bench.gate({"incoai": _run(100, 200), "g-w2048": _run(85, 190), "g-w4096": _run(91, 181)})
    assert ok
    assert any("g-w4096" in line and "PASS" in line for line in lines)
    assert any("g-w2048" in line and "FAIL" in line for line in lines)


def test_gate_fails_when_one_category_short():
    ok, _ = bench.gate({"incoai": _run(100, 200), "g-w2048": _run(95, 170)})
    assert not ok


def test_gate_ignores_runs_without_concurrency_4():
    ok, lines = bench.gate({"incoai": _run(100, 200), "g-full": {"prose": {"1": {"agg_tok_s": 50}}}})
    assert not ok and any("g-full" in line and "skipped" in line for line in lines)


def test_long_prompts():
    long = bench.long_prompts(ROOT)
    assert set(long) == {"prose", "code"} and all(len(p) == 8 for p in long.values())
    assert long == bench.long_prompts(ROOT)
    for cat, prompts in long.items():
        sizes = [len(p) for p in prompts]
        # one at a time sends the first four: two of each size
        assert sorted(sizes[:4])[1] < sorted(sizes[:4])[2]
        for p, (name, task) in zip(prompts, bench.LONG[cat] * 2):
            assert p.startswith("```\n" + (ROOT / name).read_text(encoding="utf-8")[:200])
            assert p.endswith(task)
    assert max(len(p) for p in long["code"]) > 16384 * bench.CHARS_PER_TOKEN * 0.95


def test_identity_needs_both_hashes():
    assert bench.same_reply("ab12", "ab12")
    assert not bench.same_reply("ab12", "cd34")
    assert not bench.same_reply(None, None)
    assert not bench.same_reply("ab12", None)
