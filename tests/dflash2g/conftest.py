"""CPU tests of patches 0084/0085 against .work/tf/src (dev/tree.sh) and the drafters' small files (dev/fixtures.sh)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / ".work" / "tf" / "src"
FIXTURES = ROOT / ".work" / "fixtures"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def g_dir() -> Path:
    path = FIXTURES / "G"
    if not (path / "mask_embedding.pt").exists():
        pytest.skip("run dev/fixtures.sh")
    return path


@pytest.fixture
def g_cfg(g_dir) -> dict:
    return json.loads((g_dir / "config.json").read_text())


@pytest.fixture
def incoai_cfg() -> dict:
    path = FIXTURES / "incoai" / "config.json"
    if not path.exists():
        pytest.skip("run dev/fixtures.sh")
    return json.loads(path.read_text())
