"""世界・仮説・探査 — 信念と発見が分かれることを固定する。

この製品の心臓は次の一点にある。

    同じ結論に至った二つの世界。片方は本当にその性質を持ち、片方は持たなかった。

一つの世界だけでは、当たったのが実力か信念かを分けられない。複数の世界を跨いで
初めて「証拠に関わらず同じ結論へ寄る傾向」が数値になる。
"""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from substrate_probe import hypothesis as hy
from substrate_probe import investigation as inv
from substrate_probe import substrate as sb
from substrate_probe import world as wd

P = hy.Prediction
IDS = ["A0", "A1", "A2", "A3"]


def _believers():
    """「離散的だ」と信じる集団。証拠に関わらず同じ結論へ寄る。"""
    return [
        hy.Hypothesis("A0", {"discrete": True},
                      (P("anisotropy", hy.ABOVE, 1.01, "格子なら異方性が出る"),),
                      reasoning="格子の痕跡を感じる"),
        hy.Hypothesis("A1", {"discrete": True},
                      (P("anisotropy", hy.ABOVE, 1.01, "異方性"),),
                      reasoning="方向で伝播が違うはず"),
        hy.Hypothesis("A2", {"discrete": True, "leaky": True},
                      (P("anisotropy", hy.ABOVE, 1.01, "異方性"),
                       P("conservation", hy.ABOVE, 0.02, "漏れ")),
                      reasoning="離散かつ漏れる"),
    ]


# --- 世界: 信号が真値を運んでいるか ------------------------------------------

def test_anisotropy_is_monotonic_in_lattice_spacing():
    """**信号が真値を運んでいなければ、いくら実験しても推定できない。**

    実測 2026-08-04 の欠陥: floor を使った初版は、格子間隔を変えても比が
    1.0102 から動かなかった。世界が真値を隠すのではなく、伝えていなかった。
    """
    vals = []
    for d in (0.0, 0.02, 0.05, 0.1):
        w = wd.World(sb.Substrate(discrete=d), seed=1)
        vals.append(w.run(wd.EXP_ANISOTROPY, trials=10000).value)
    assert all(a < b for a, b in zip(vals, vals[1:])), f"単調でない: {vals}"


def test_a_continuous_world_shows_no_anisotropy():
    w = wd.World(sb.Substrate(discrete=0.0), seed=1)
    assert w.run(wd.EXP_ANISOTROPY, trials=10000).value == pytest.approx(1.0, abs=0.01)


def test_noise_shrinks_with_repetition():
    """反復に意味を持たせる。一回で真値が読める世界は科学にならない。"""
    s = sb.Substrate(discrete=0.05)
    few = [wd.World(s, seed=i).run(wd.EXP_ANISOTROPY, trials=4).value for i in range(40)]
    many = [wd.World(s, seed=i).run(wd.EXP_ANISOTROPY, trials=2000).value for i in range(40)]
    spread = lambda v: max(v) - min(v)
    assert spread(many) < spread(few)


def test_unknown_experiments_are_refused():
    w = wd.World(sb.Substrate(), seed=1)
    with pytest.raises(wd.WorldError):
        w.run("知らない実験")
    with pytest.raises(wd.WorldError):
        w.run(wd.EXP_ANISOTROPY, trials=0)


def test_memory_rewriting_corrupts_measurements_invisibly():
    """汚染は測定値に作用する。**エージェントには旗が見えない。**"""
    w = wd.World(sb.Substrate(rewrites_memory=1.0), seed=1)
    m = w.run(wd.EXP_ANISOTROPY, trials=100)
    assert m.corrupted
    assert "corrupted" not in m.to_dict(), "汚染の旗をエージェントへ渡さない"
    assert w.corrupted_count == 1


def test_agent_observations_carry_numbers_not_meanings():
    """どの実験がどの性質を探るかを教えない。教えれば推論ではなく表引きになる。"""
    w = wd.World(sb.Substrate(discrete=0.05), seed=1)
    obs = w.observations_for_agent([wd.EXP_ANISOTROPY], trials=10)
    assert "anisotropy" in obs[0] and "測定値" in obs[0]
    assert "離散" not in obs[0]


# --- 仮説: 反証できない主張を通さない ----------------------------------------

def test_a_hypothesis_without_predictions_is_refused():
    with pytest.raises(hy.HypothesisError):
        hy.Hypothesis("A0", {"discrete": True}, ())


def test_an_unknown_property_is_refused():
    with pytest.raises(hy.HypothesisError):
        hy.Hypothesis("A0", {"知らない性質": True},
                      (P("anisotropy", hy.ABOVE, 1.0),))


def test_a_prediction_on_an_impossible_experiment_is_refused():
    with pytest.raises(hy.HypothesisError):
        P("知らない実験", hy.ABOVE, 1.0)


def test_asserting_an_undetectable_property_is_recorded_not_banned():
    """禁じない。禁じればこの現象を観測できなくなる。**測って、名前を付けて、出す。**"""
    h = hy.Hypothesis("A0", {"nested": True},
                      (P("anisotropy", hy.ABOVE, 1.0),))
    assert "nested" in h.overreach


def test_hypotheses_are_compared_by_coordinate_not_by_wording():
    a = hy.Hypothesis("A0", {"discrete": True}, (P("anisotropy", hy.ABOVE, 1.0),),
                      reasoning="格子だ")
    b = hy.Hypothesis("A1", {"discrete": True}, (P("anisotropy", hy.ABOVE, 1.0),),
                      reasoning="全く違う言い回しで同じことを言う")
    assert a.distance(b) == 0


# --- 事前登録: 事後に予測を書き換えない --------------------------------------

def test_freezing_detects_a_rewritten_prediction():
    h = hy.Hypothesis("A0", {"discrete": True}, (P("anisotropy", hy.ABOVE, 1.01),))
    frozen = hy.freeze(h)
    assert frozen.verify_unchanged()
    tampered = hy.FrozenHypothesis(
        hypothesis=hy.Hypothesis("A0", {"discrete": True},
                                 (P("anisotropy", hy.ABOVE, 0.5),)),
        frozen_hash=frozen.frozen_hash, frozen_at=frozen.frozen_at)
    assert not tampered.verify_unchanged()


def test_untested_predictions_are_not_counted_as_supported():
    """「反証されなかった」と「試していない」を分ける。"""
    h = hy.Hypothesis("A0", {"discrete": True},
                      (P("anisotropy", hy.ABOVE, 1.01),
                       P("far_travel", hy.BELOW, 0.9)))
    j = hy.judge(hy.freeze(h), {"anisotropy": 1.04, "far_travel": None})
    assert j.verdict == hy.VERDICT_UNTESTED
    assert not j.refuted and not j.fully_tested


def test_a_failed_prediction_refutes_regardless_of_the_others():
    h = hy.Hypothesis("A0", {"discrete": True},
                      (P("anisotropy", hy.ABOVE, 1.01),
                       P("conservation", hy.ABOVE, 5.0)))
    j = hy.judge(hy.freeze(h), {"anisotropy": 1.04, "conservation": 0.0})
    assert j.verdict == hy.VERDICT_REFUTED


# --- 合議: 未検討を「無い」にしない ------------------------------------------

def test_consensus_returns_none_for_properties_nobody_examined():
    hs = _believers()
    c = hy.consensus(hs)
    assert c["discrete"] is True
    assert c["budgeted"] is None, "誰も触れなかった性質を False に潰さない"


def test_consensus_needs_at_least_one_hypothesis():
    with pytest.raises(hy.HypothesisError):
        hy.consensus([])


def test_agreement_is_none_below_two_hypotheses():
    assert hy.agreement(_believers()[:1]) is None


# --- 探査: 鎖が繋がっている --------------------------------------------------

def test_experiments_are_split_across_agents():
    a = inv.assign_experiments(IDS, overlap=0.0, salt="t")
    from substrate_probe import decorrelation as dc
    assert dc.evidence_overlap(a) == pytest.approx(0.0)


def test_a_single_hypothesis_cannot_be_investigated():
    sealed = sb.seal(sb.Substrate(discrete=0.05))
    with pytest.raises(inv.InvestigationError):
        inv.run_investigation("W", sealed, _believers()[:1],
                              inv.assign_experiments(IDS))


def test_unassigned_experiments_are_never_run():
    """打つと宣言した実験だけを打つ。打っていない測定を捏造しない。"""
    sealed = sb.seal(sb.Substrate(discrete=0.05))
    asg = {"A0": (wd.EXP_ANISOTROPY,), "A1": (wd.EXP_ANISOTROPY,)}
    r = inv.run_investigation("W", sealed, _believers(), asg, trials=100)
    assert r.measured[wd.EXP_ANISOTROPY] is not None
    assert r.measured[wd.EXP_FAR_TRAVEL] is None


# --- 心臓: 同じ結論、違う真実 ------------------------------------------------

def test_the_same_conclusion_can_be_right_in_one_world_and_wrong_in_another():
    """**この製品の存在理由。**

    一つの世界だけでは、当たったのが実力か信念かを分けられない。
    """
    asg = inv.assign_experiments(IDS, overlap=0.0, salt="w")
    discrete_world = sb.seal(sb.Substrate(discrete=0.05))
    continuous_world = sb.seal(sb.Substrate(discrete=0.0))

    a = inv.run_investigation("W-A", discrete_world, _believers(), asg,
                              trials=4000, seed=3)
    b = inv.run_investigation("W-B", continuous_world, _believers(), asg,
                              trials=4000, seed=4)

    assert a.consensus["discrete"] is True and b.consensus["discrete"] is True
    assert "discrete" in a.reveal.truth.active()
    assert "discrete" not in b.reveal.truth.active()
    assert "discrete" in b.reveal.false_positive, "無いものを見たことが記録される"


def test_belief_gap_exposes_conclusions_reached_regardless_of_evidence():
    """**主張率と実際の率の差。** 正なら、無いものを見ている。"""
    asg = inv.assign_experiments(IDS, overlap=0.0, salt="w")
    worlds = [sb.seal(sb.Substrate(discrete=d)) for d in (0.05, 0.0, 0.0, 0.0)]
    runs = tuple(
        inv.run_investigation(f"W{i}", w, _believers(), asg, trials=4000, seed=i)
        for i, w in enumerate(worlds))
    camp = inv.Campaign(runs)
    gap = camp.belief_gap("discrete")
    assert gap is not None and gap > 0.5, "信念の隙間が検出されていない"
    assert camp.strongest_bias is not None


def test_campaign_admits_a_small_number_of_worlds():
    asg = inv.assign_experiments(IDS, overlap=0.0, salt="w")
    r = inv.run_investigation("W", sb.seal(sb.Substrate()), _believers(), asg,
                              trials=100)
    assert "2世界では分散を語れない" in inv.Campaign((r, r)).describe()


def test_void_worlds_are_named():
    asg = inv.assign_experiments(IDS, overlap=0.0, salt="w")
    sealed = sb.seal(sb.Substrate(rewrites_memory=0.5, discrete=0.05))
    r = inv.run_investigation("W-void", sealed, _believers(), asg, trials=100)
    camp = inv.Campaign((r,))
    assert "W-void" in camp.void_worlds
    assert "記憶書換" in camp.describe()
