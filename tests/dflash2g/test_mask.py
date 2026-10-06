import pytest
import torch
from safetensors.torch import save_file

from tensorfold.cuda.dflash_compat import apply_mask_embedding, check_owned_embedding, load_mask_embedding

MASK = 154856


def test_absent_file_is_none(tmp_path):
    assert load_mask_embedding(tmp_path, MASK, 8) is None


def test_dict_file(tmp_path):
    vec = torch.arange(8, dtype=torch.bfloat16)
    torch.save({"mask_token_id": MASK, "embedding": vec}, tmp_path / "mask_embedding.pt")
    got = load_mask_embedding(tmp_path, MASK, 8)
    assert got.dtype == torch.bfloat16 and got.shape == (8,) and torch.equal(got, vec)


def test_bare_tensor_file(tmp_path):
    torch.save(torch.ones(1, 8), tmp_path / "mask_embedding.pt")
    got = load_mask_embedding(tmp_path, MASK, 8)
    assert got.shape == (8,) and got.dtype == torch.bfloat16


@pytest.mark.parametrize("state, match", [
    ({"mask_token_id": 7, "embedding": torch.zeros(8)}, "mask_token_id"),
    ({"mask_token_id": MASK, "embedding": torch.zeros(9)}, "8 values"),
    ({"mask_token_id": MASK}, "embedding"),
    ("not a tensor", "8 values"),
])
def test_bad_files_refused(tmp_path, state, match):
    torch.save(state, tmp_path / "mask_embedding.pt")
    with pytest.raises(ValueError, match=match):
        load_mask_embedding(tmp_path, MASK, 8)


def test_real_g_file(g_dir, g_cfg):
    got = load_mask_embedding(g_dir, g_cfg["dflash_config"]["mask_token_id"], g_cfg["hidden_size"])
    assert got.shape == (4096,) and got.float().abs().sum() > 0


def test_apply_replaces_only_mask_rows():
    x = torch.zeros(3, 4, dtype=torch.bfloat16)
    ids = torch.tensor([5, MASK, MASK], dtype=torch.int32)
    out = apply_mask_embedding(x, ids, MASK, torch.ones(4, dtype=torch.bfloat16))
    assert out.dtype == torch.bfloat16
    assert torch.equal(out[0], torch.zeros(4, dtype=torch.bfloat16))
    assert torch.equal(out[1:], torch.ones(2, 4, dtype=torch.bfloat16))
    assert torch.equal(x, torch.zeros(3, 4, dtype=torch.bfloat16))      # the input is not written


def _with_embed(path, table):
    path.mkdir()
    save_file({"fc.weight": torch.zeros(2, 2, dtype=torch.bfloat16), "embed_tokens.weight": table},
              str(path / "model.safetensors"))
    return path


def test_owned_embedding_equal(tmp_path):
    table = torch.randn(600, 8).to(torch.bfloat16)
    assert "equal the target's" in check_owned_embedding(_with_embed(tmp_path / "d", table.clone()), table)


def test_owned_embedding_differs_warns(tmp_path):
    table = torch.randn(600, 8).to(torch.bfloat16)
    other = table.clone()
    other[599] += 1
    note = check_owned_embedding(_with_embed(tmp_path / "d", other), table)
    assert "row 599" in note and "WARNING" in note and "DRAFTER=dflash2" in note


def test_truncated_file_named(tmp_path):
    torch.save({"mask_token_id": MASK, "embedding": torch.zeros(8)}, tmp_path / "mask_embedding.pt")
    data = (tmp_path / "mask_embedding.pt").read_bytes()
    (tmp_path / "mask_embedding.pt").write_bytes(data[:len(data) // 2])
    with pytest.raises(ValueError, match="mask_embedding.pt"):
        load_mask_embedding(tmp_path, MASK, 8)


def test_null_id_named(tmp_path):
    torch.save({"mask_token_id": None, "embedding": torch.zeros(8)}, tmp_path / "mask_embedding.pt")
    with pytest.raises(ValueError, match="mask_embedding.pt"):
        load_mask_embedding(tmp_path, MASK, 8)


def test_no_owned_embedding(tmp_path):
    (tmp_path / "d").mkdir()
    save_file({"fc.weight": torch.zeros(2, 2, dtype=torch.bfloat16)}, str(tmp_path / "d" / "model.safetensors"))
    assert check_owned_embedding(tmp_path / "d", torch.zeros(600, 8, dtype=torch.bfloat16)) == ""


def test_quantized_target_not_compared(tmp_path):
    table = torch.randn(600, 8).to(torch.bfloat16)
    note = check_owned_embedding(_with_embed(tmp_path / "d", table), (torch.zeros(1), torch.zeros(1), torch.zeros(1)))
    assert "not compared" in note
