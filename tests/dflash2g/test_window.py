import pytest
import torch
from safetensors.torch import save_file

from tensorfold.cuda.dflash_compat import DEFAULT_WINDOW, dflash_window
from tensorfold.cuda.geometry import dflash2_geometry, dflash2_weights, draft_geometry, draft_ring_rows


@pytest.fixture(autouse=True)
def no_env(monkeypatch):
    monkeypatch.delenv("TF_GLM_DFLASH_WINDOW", raising=False)


def test_config_window_wins_over_env(monkeypatch, incoai_cfg):
    monkeypatch.setenv("TF_GLM_DFLASH_WINDOW", "8192")
    assert dflash_window(incoai_cfg) == 2048


def test_null_window_defaults_to_2048(g_cfg):
    assert g_cfg["sliding_window"] is None
    assert dflash_window(g_cfg) == DEFAULT_WINDOW == 2048
    assert dflash_window({}) == 2048


@pytest.mark.parametrize("raw, want", [("4096", 4096), (" 8192 ", 8192), ("0", 0), ("64", 64)])
def test_env_sets_null_window(monkeypatch, g_cfg, raw, want):
    monkeypatch.setenv("TF_GLM_DFLASH_WINDOW", raw)
    assert dflash_window(g_cfg) == want


@pytest.mark.parametrize("raw", ["abc", "-1", "8", "63", "2k"])
def test_bad_env_refused(monkeypatch, g_cfg, raw):
    monkeypatch.setenv("TF_GLM_DFLASH_WINDOW", raw)
    with pytest.raises(ValueError, match="TF_GLM_DFLASH_WINDOW"):
        dflash_window(g_cfg)


def _fixed(cfg, rows=64):
    return 16 * rows * (cfg["hidden_size"] + cfg["intermediate_size"]) * 4


def test_g_ring_geometry(g_cfg):
    g = dflash2_geometry(g_cfg, 2, 64, ring=True)
    rows = draft_ring_rows(2047, 8)
    assert rows == 2176
    assert g.bytes_at(1_048_576) == _fixed(g_cfg) + 2 * 8 * 4 * 128 * rows * 2


def test_g_full_attention_is_flat(monkeypatch, g_cfg):
    monkeypatch.setenv("TF_GLM_DFLASH_WINDOW", "0")
    g = dflash2_geometry(g_cfg, 2, 64, ring=True)
    assert g.bytes_at(131_072) == _fixed(g_cfg) + 2 * 8 * 4 * 128 * (131_072 + 8) * 2


def test_incoai_geometry_unchanged(incoai_cfg):
    g = dflash2_geometry(incoai_cfg, 2, 64, ring=True)
    assert g.bytes_at(1_048_576) == _fixed(incoai_cfg) + 2 * 5 * 4 * 128 * 2176 * 2


def test_draft_geometry_bounded_by_window(g_cfg):
    g = draft_geometry(g_cfg, 2, 64, bounded=True)
    assert g.bytes_at(10_000) == _fixed(g_cfg) + 2 * 2 * 8 * 4 * 128 * (2048 + 8) * 2


def _drafter(path, owned: bool):
    D, inter, vocab = 128, 256, 512
    t = {"fc.weight": (D, 3 * D), "hidden_norm.weight": (D,), "norm.weight": (D,),
         "candidate_selector.predecessor_codebook": (vocab, 16),
         "layers.0.self_attn.q_proj.weight": (D, D), "layers.0.self_attn.k_proj.weight": (D // 4, D),
         "layers.0.self_attn.v_proj.weight": (D // 4, D), "layers.0.self_attn.o_proj.weight": (D, D),
         "layers.0.mlp.gate_proj.weight": (inter, D), "layers.0.mlp.up_proj.weight": (inter, D),
         "layers.0.mlp.down_proj.weight": (D, inter),
         "layers.0.attention_conv.kernel_projection.weight": (64, D),
         "layers.0.mlp_conv.kernel_projection.weight": (64, D)}
    if owned:
        t["embed_tokens.weight"] = (vocab, D)
        t["lm_head.weight"] = (vocab, D)
    path.mkdir()
    save_file({k: torch.zeros(s, dtype=torch.bfloat16) for k, s in t.items()}, str(path / "model.safetensors"))
    return path


def test_weights_ignore_owned_copies(tmp_path):
    bare = dflash2_weights(_drafter(tmp_path / "bare", False), 2)
    owned = dflash2_weights(_drafter(tmp_path / "owned", True), 2)
    assert owned.resident == bare.resident
    assert owned.staging == bare.staging
