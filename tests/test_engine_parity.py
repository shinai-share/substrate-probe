"""JS エンジンと Python 参照実装の照合 — 二重定義のずれを黙って進行させない。

画面はブラウザで動くため、エンジンは JavaScript でもう一度書かれている。
二つある以上、必ずずれる。ここでは **語彙と定数** の一致を機械で照合し、
片方だけ直した状態がテストを通らないようにする。

(数値の完全一致まで JS を実行して確かめるのは Node 依存になるため、
 ここでは静的照合に留める。実行検証は tools/build_viewer.py 後の node -e で行う)
"""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from substrate_probe import substrate as sb
from substrate_probe import world as wd

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "viewer" / "engine.js"
INDEX = ROOT / "viewer" / "index.html"
TEMPLATE = ROOT / "viewer" / "app_template.html"


@pytest.fixture(scope="module")
def js():
    return ENGINE.read_text(encoding="utf-8")


def test_every_property_key_exists_in_js(js):
    for key in sb.PROPERTY_KEYS:
        assert f'"{key}"' in js, f"JS 側に性質 {key} が無い"


def test_every_experiment_exists_in_js(js):
    for exp in wd.EXPERIMENTS:
        assert f'"{exp}"' in js, f"JS 側に実験 {exp} が無い"


def test_detectability_tiers_match(js):
    """検出可能性の分類は製品の核である。両実装で食い違えば採点が割れる。"""
    for p in sb.PROPERTIES:
        # JS 側の { key: "discrete", ... tier: DETECTABLE } を突き合わせる
        m = re.search(r'key:\s*"' + re.escape(p.key) + r'".*?tier:\s*(\w+)', js, re.S)
        assert m, f"JS 側に {p.key} の tier が無い"
        assert m.group(1).lower() == p.detectability, (
            f"{p.key}: Python={p.detectability} / JS={m.group(1).lower()}")


def test_noise_and_gain_constants_match(js):
    """世界の物理定数。ずれれば同じ種でも違う測定値になる。"""
    assert f"BASE_NOISE = {wd.BASE_NOISE}" in js
    assert f"ANISOTROPY_GAIN = {wd.ANISOTROPY_GAIN}" in js


def test_index_html_embeds_engine_verbatim():
    """index.html は生成物である。engine.js の全文を一字違わず含む。

    片方だけ直して再生成を忘れた状態を、ここで止める。
    """
    engine = ENGINE.read_text(encoding="utf-8")
    index = INDEX.read_text(encoding="utf-8")
    assert engine in index, (
        "index.html が engine.js と食い違っている。"
        "python tools/build_viewer.py で再生成すること")


def test_index_html_declares_itself_generated():
    assert "生成物である。直接編集しない" in INDEX.read_text(encoding="utf-8")


def test_template_keeps_the_injection_mark():
    assert "<!--ENGINE-->" in TEMPLATE.read_text(encoding="utf-8")
