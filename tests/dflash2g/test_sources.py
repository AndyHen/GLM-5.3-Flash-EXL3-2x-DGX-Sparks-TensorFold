"""The GPU files cannot be imported on the dev box: check their sources instead."""
import re

from conftest import SRC

GLM = SRC / "tensorfold" / "families" / "glm5_next" / "cuda"
RAW = re.compile(r"""\[\s*["']sliding_window["']\s*\]|\.get\(\s*["']sliding_window["']""")


def test_no_raw_sliding_window_reads():
    files = [*GLM.glob("*.py"), SRC / "tensorfold" / "cuda" / "geometry.py"]
    hits = [f"{p.name}:{i}" for p in files for i, line in enumerate(p.read_text().splitlines(), 1) if RAW.search(line)]
    assert hits == []


def test_engine_agrees_on_window_and_refuses_full_attention_in_parallel():
    text = (GLM / "engine.py").read_text()
    assert "TF_GLM_DFLASH_WINDOW" in text                       # named in the ranks-differ error
    assert "dflash_window(read_config(drafter))" in text        # in the agreement list
    assert "full attention" in text and "--parallel 1" in text  # the refusal


def test_both_block_passes_apply_the_mask():
    for name in ("dflash2.py", "dflash2_multi.py"):
        text = (GLM / name).read_text()
        assert "apply_mask_embedding(x, " in text, name
    assert "load_mask_embedding(path, self.mask_id, self.D)" in (GLM / "dflash2.py").read_text()
