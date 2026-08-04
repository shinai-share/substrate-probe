"""画面 — 実験機であり、実験観測機であることを固定する。

これは結果の提示ではない。**ユーザーが開始を押すと装置が走り、探査者たちの
振る舞いが目の前で進行する。** ここで守るのは三つ:

  一. 開始 -> 実行 -> 観測の構造が実在すること(録画の再生ではない)
  二. 封を切るまで真値を画面にも見せないこと(二重盲検を画面の構造にする)
  三. 不在(未検討・検出不能・無効世界)を装飾ではなく情報として描くこと
"""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from substrate_probe import substrate as sb

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "viewer" / "app_template.html"
INDEX = ROOT / "viewer" / "index.html"
REPORT = ROOT / "viewer" / "report.html"


@pytest.fixture(scope="module")
def app():
    return APP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def index():
    return INDEX.read_text(encoding="utf-8")


# --- 自己完結・両テーマ・動きの配慮 ------------------------------------------

def test_app_fetches_nothing_from_the_network(index):
    assert not re.findall(r'(?:src|href)\s*=\s*["\']https?://', index)
    assert "@import" not in index


def test_both_themes_are_designed(app):
    assert "prefers-color-scheme: light" in app
    assert '[data-theme="dark"]' in app and '[data-theme="light"]' in app


def test_motion_can_be_turned_off(app):
    assert "prefers-reduced-motion" in app
    assert "reduce" in app, "JS 側が減速設定を読んでいない"


# --- 一. 実験機である: 開始 -> 実行 -> 観測 ----------------------------------

def test_the_user_starts_the_simulation(app):
    """開始はユーザーの行為である。自動再生の録画ではない。"""
    assert 'id="start"' in app
    assert "探査を開始する" in app
    assert 'addEventListener("click"' in app


def test_agents_are_visible_actors_not_a_table(app):
    """探査者は名を持ち、信条を持ち、活動中は光る。"""
    assert "SCIENTISTS.map" in app
    assert 'class="agent' in app
    assert "setActive" in app, "誰が動いているかが画面に出ない"


def test_each_step_is_played_not_dumped(app):
    """段階を一つずつ再生する。全結果を一括で流し込めば、それは報告書である。"""
    assert "playStep" in app
    assert "setTimeout(playStep" in app
    assert "DUR" in app, "段階ごとの時間が無い(=一括表示)"


def test_the_run_is_operable(app):
    """一時停止と速度。観測機は観測者が操作できねばならない。"""
    assert 'id="pause"' in app
    assert 'data-speed="1"' in app and 'data-speed="4"' in app


def test_probes_travel_on_the_canvas(app):
    """実験は絵として飛ぶ。ログだけなら端末で足りる。"""
    assert "pulses.push" in app
    assert "agentPos" in app


def test_measured_values_are_shown_not_hidden(app):
    """測定値そのものが画面に現れる。装置は数値を隠さない。"""
    assert "p.value.toFixed(4)" in app


# --- 二. 封を切るまで真値を見せない ------------------------------------------

def test_the_world_core_is_fogged_until_reveal(app):
    """探査中の世界は霧である。開封の瞬間に材質が実体化する。

    観る人にも真値を先に見せない —— 二重盲検を演出ではなく画面の構造にする。
    """
    assert "sealed" in app and "fog" in app
    assert 'case "reveal"' in app
    assert "sealed = false" in app


def test_reveal_materializes_the_substrate(app):
    """開封後、格子・境界・漏れが描かれる。"""
    assert "s.discrete > 0" in app
    assert "s.bounded > 0" in app
    assert "s.leaky > 0" in app


def test_predictions_freeze_before_the_reveal(app):
    assert "凍結" in app
    assert "ここから先は書き換えられない" in app


# --- 三. 不在を情報として描く ------------------------------------------------

def test_leaps_of_faith_are_drawn_distinctly(app):
    """測っていないのに断言した瞬間は、破線で描かれる。"""
    assert "leap" in app
    assert "border-style:dashed" in app
    assert "測っていない" in app


def test_untouched_properties_stay_blank(app):
    assert "未検討" in app
    assert "調べなかった性質は空欄のまま残す" in app


def test_voided_worlds_are_named(app):
    assert "記録を書き換える基体だった" in app
    assert "無効" in app


def test_the_bare_world_finding_is_surfaced(app):
    """素の世界に性質を見た、という発見が観測パネルに出る。"""
    assert "素の世界" in app


def test_belief_gap_accumulates_across_worlds(app):
    """観測機の核: 世界を重ねるほど、傾きが見えてくる。"""
    assert "renderGaps" in app
    assert "信念の隙間" in app


def test_no_total_score_or_ranking(app):
    for banned in ("総合スコア", "ランキング", "1位", "おすすめ"):
        assert banned not in app


def test_the_page_refuses_to_claim_consciousness(app):
    assert "意識を作っていない" in app
    assert "認識論の実験場" in app


def test_rule_based_agents_are_disclosed(app):
    """探査者が規則で動くことを隠さない。言語モデル版への差替も明記する。"""
    assert "規則で動く" in app
    assert "言語モデル版へ差し替えられる" in app


def test_output_is_escaped(app):
    assert "const esc" in app


# --- 記録ページ(旧静的ページ)は保存されている -----------------------------

def test_the_report_page_survives():
    assert REPORT.exists()
    t = REPORT.read_text(encoding="utf-8")
    assert "W-04" in t and "真値は空だった" in t


def test_no_foreign_script_contamination(index):
    bad = re.compile("[" + chr(0x0400) + "-" + chr(0x04FF) + chr(0xAC00) + "-" + chr(0xD7AF) + "]")
    assert not bad.findall(index)
    for p in sb.PROPERTIES:
        assert p.label in index, f"{p.label} が画面に無い"
