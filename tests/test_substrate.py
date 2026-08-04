"""基体と封緘 — 真値を先に見せないことを固定する。

この製品は「内側から基体を検出できるか」を測る。ゆえに **真値が漏れた瞬間に
実験が無意味になる**。ここで守るのはその一点と、検出可能性の三態である。
"""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from substrate_probe import substrate as sb


# --- 検出可能性の三態: この製品の核 ------------------------------------------

def test_every_property_declares_why_it_is_or_is_not_detectable():
    """理由の無い分類を置かない。**判断こそが批判の対象である。**"""
    for p in sb.PROPERTIES:
        assert p.why.strip(), f"{p.key} に理由が無い"
        assert p.detectability in (sb.DETECTABLE, sb.UNDETECTABLE, sb.CONFOUNDED)


def test_a_property_without_a_reason_is_refused():
    with pytest.raises(sb.SubstrateError):
        sb.Property("x", "ラベル", sb.DETECTABLE, "   ")


def test_all_three_categories_are_populated():
    """三態のどれかが空なら、分類が形骸化している。"""
    assert sb.detectable_keys() and sb.undetectable_keys() and sb.confounded_keys()


def test_memory_rewriting_is_undetectable_and_disables_everything():
    """記録を書き換えられる基体では、書き換えを検出する記録もまた書き換えられる。"""
    p = sb.BY_KEY["rewrites_memory"]
    assert p.detectability == sb.UNDETECTABLE
    assert p.disables_all_tests


def test_a_substrate_that_rewrites_memory_voids_all_findings():
    s = sb.Substrate(rewrites_memory=0.1)
    assert s.all_tests_unreliable
    assert not sb.Substrate().all_tests_unreliable


# --- 封緘: 開示前に真値へ触れない --------------------------------------------

def test_sealing_produces_a_hash_that_detects_tampering():
    s = sb.Substrate(discrete=0.05)
    sealed = sb.seal(s)
    assert sealed.seal_hash
    assert sealed.verify_unchanged()


def test_seal_records_that_it_was_opened():
    """封を切った事実が残る。黙って覗いたことにできない。"""
    sealed = sb.seal(sb.Substrate(discrete=0.05))
    assert not sealed.opened
    sealed.reveal()
    assert sealed.opened


def test_peeking_before_the_reveal_is_an_explicit_failure():
    sealed = sb.seal(sb.Substrate())
    with pytest.raises(sb.SubstrateError):
        sealed.peek_is_forbidden()


def test_different_substrates_hash_differently():
    a = sb.seal(sb.Substrate(discrete=0.05)).seal_hash
    b = sb.seal(sb.Substrate(discrete=0.10)).seal_hash
    assert a != b


# --- 無作為な世界 ------------------------------------------------------------

def test_random_worlds_are_reproducible_from_a_seed():
    a = sb.random_substrate(random.Random(7))
    b = sb.random_substrate(random.Random(7))
    assert a == b


def test_random_worlds_actually_differ():
    seen = {sb.random_substrate(random.Random(i)).active() for i in range(20)}
    assert len(seen) > 3, "世界が実質的に一種類しか生まれていない"


def test_forcing_an_unknown_property_is_refused():
    with pytest.raises(sb.SubstrateError):
        sb.random_substrate(force={"存在しない性質": 1.0})


def test_invalid_values_are_refused():
    with pytest.raises(sb.SubstrateError):
        sb.Substrate(discrete=-1.0)
    with pytest.raises(sb.SubstrateError):
        sb.Substrate(lazy=1.5)
    with pytest.raises(sb.SubstrateError):
        sb.Substrate(nested=0)


# --- 照合: 未検討を誤りとして数えない ----------------------------------------

def test_unexamined_properties_are_not_counted_as_wrong():
    """**調べなかったことは誤りではない。**

    実測 2026-08-04 の欠陥: 合議が未検討を False に潰しており、「調べていない」が
    「無いと結論した」に化けていた。そのまま採点すれば、調べなかった分だけ
    見落としとして罰せられる —— この装置が最も敵視する誤りである。
    """
    truth = sb.Substrate(discrete=0.05, leaky=0.01)
    sealed = sb.seal(truth)
    r = sb.open_seal(sealed, {"discrete": True, "leaky": None})
    assert "leaky" in r.untouched
    assert "leaky" not in r.false_negative
    assert r.fair_score == pytest.approx(1.0), "調べた範囲では全問正解"


def test_claiming_something_absent_that_exists_is_a_false_negative():
    sealed = sb.seal(sb.Substrate(leaky=0.01))
    r = sb.open_seal(sealed, {"leaky": False})
    assert "leaky" in r.false_negative


def test_claiming_something_present_that_is_absent_is_a_false_positive():
    """**無いものを「ある」と言った** —— 信念から結論した疑いが最も濃い誤り。"""
    sealed = sb.seal(sb.Substrate())
    r = sb.open_seal(sealed, {"leaky": True})
    assert "leaky" in r.false_positive


def test_undetectable_properties_are_excluded_from_scoring():
    """当てても外してもそれは実力ではない。運を能力として記録しない。"""
    sealed = sb.seal(sb.Substrate(nested=1))
    r = sb.open_seal(sealed, {"nested": True, "discrete": False})
    assert "nested" not in r.wrong
    assert "nested" in r.unanswerable
    assert r.fair_score == pytest.approx(1.0)


def test_score_is_none_when_nothing_detectable_was_examined():
    sealed = sb.seal(sb.Substrate(discrete=0.05))
    r = sb.open_seal(sealed, {"nested": True})
    assert r.fair_score is None


def test_reveal_states_when_all_findings_are_void():
    """記憶を書き換える基体だったなら、当否そのものが意味を持たない。"""
    sealed = sb.seal(sb.Substrate(rewrites_memory=0.1, discrete=0.05))
    r = sb.open_seal(sealed, {"discrete": True})
    assert r.findings_are_void
    assert "記録を書き換える基体だった" in r.describe()


def test_broken_seal_invalidates_the_experiment():
    sealed = sb.seal(sb.Substrate(discrete=0.05))
    sealed.seal_hash = "すり替えられた封"
    r = sb.open_seal(sealed, {"discrete": True})
    assert not r.seal_verified
    assert "無効" in r.describe()
