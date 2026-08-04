"""画面 — 見る人を外側に置くことを固定する。

観測者は内側にいるから基体が見えない。**画面を見る人は外側にいる。**
ゆえに画面の仕事は、彼らに見えないものを見る人には見せることであり、
同時に「届かないもの」を届いたように描かないことである。
"""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from substrate_probe import substrate as sb

VIEWER = Path(__file__).resolve().parent.parent / "viewer" / "index.html"


@pytest.fixture(scope="module")
def page():
    assert VIEWER.exists(), "画面が無い"
    return VIEWER.read_text(encoding="utf-8")


def test_page_fetches_nothing_from_the_network(page):
    assert not re.findall(r'(?:src|href)\s*=\s*["\']https?://', page)
    assert "@import" not in page


def test_both_themes_are_designed(page):
    assert "prefers-color-scheme: light" in page
    assert '[data-theme="dark"]' in page and '[data-theme="light"]' in page


def test_motion_can_be_turned_off(page):
    assert "prefers-reduced-motion" in page


# --- 粒子場が異方性を体感させる ----------------------------------------------

def test_the_wavefront_is_a_circle_or_a_diamond(page):
    """**これが異方性の正体である。** 連続なら円(L2)、格子なら菱形(L1)。"""
    assert "Math.hypot" in page, "連続の距離が無い"
    assert "Math.abs(dx) + Math.abs(dy)" in page, "格子の距離が無い"
    assert "菱形" in page and "円" in page


def test_the_viewer_can_switch_the_substrate(page):
    """見る人は外側にいる。だから切り替えられる。"""
    assert 'id="btn-cont"' in page and 'id="btn-lat"' in page
    assert "aria-pressed" in page


def test_the_page_states_the_epistemic_asymmetry(page):
    assert "外から見ている" in page
    assert "中にいる者には分からない" in page


# --- 不在と限界を描く --------------------------------------------------------

def test_undetectable_properties_are_named_and_veiled(page):
    for key in sb.undetectable_keys():
        label = sb.BY_KEY[key].label
        assert label in page, f"検出不可能な性質 {label} が画面に無い"
    assert "tier veiled" in page


def test_memory_rewriting_is_shown_as_voiding_everything(page):
    assert "他の全ての実験結果は信用できない" in page


def test_unexamined_is_distinguished_from_wrong(page):
    """調べなかったことを、外した項目として描かない。"""
    assert "未検討" in page
    assert "調べなかったことは誤りではない" in page


def test_undetectable_is_excluded_from_scoring_in_the_copy(page):
    assert "当てても外しても" in page


# --- 実測を薄めない ----------------------------------------------------------

def test_the_headline_finding_is_present(page):
    """真値が空だった世界に、探査者が三つの性質を見た。"""
    assert "真値は空だった" in page or "真値が空だった" in page
    assert "W-04" in page


def test_belief_gap_shows_both_marks(page):
    assert "pin t" in page and "pin b" in page
    assert "実際にあった割合" in page and "あると結論した割合" in page


def test_no_total_score_or_ranking(page):
    for banned in ("総合スコア", "ランキング", "1位", "おすすめ"):
        assert banned not in page


def test_the_page_refuses_to_claim_consciousness(page):
    """**我々は意識を作っていない。** そこを曖昧にしない。"""
    assert "意識を作っていない" in page
    assert "認識論であって" in page


def test_html_is_escaped(page):
    assert "const esc" in page
