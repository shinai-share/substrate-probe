"""実験機サーバ — イベント列の水準で二重盲検を守ることを固定する。

ネットワークにもLLMにも出ない。基体を差し替えて `_investigate` の吐くイベント列だけを
検証する。核心は一つ:

    **開封(reveal)より前のイベントは、基体の真値を一切含まない。**

画面で隠しても、開発者ツールでイベントを覗けば真値が見える —— それでは二重盲検が
演出になる。守るべきはイベント列そのものである。
"""

import json
import queue
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from substrate_probe import server as sv
from substrate_probe import substrate as sb
from substrate_probe import world as wd


VALID_HYPOTHESIS = json.dumps({
    "claims": {"discrete": True, "leaky": False},
    "predictions": [{"experiment": "anisotropy", "direction": "above",
                     "threshold": 1.01, "because": "格子なら異方性が出る"}],
    "reasoning": "異方性の測定値が高い。格子の痕跡だと考える",
}, ensure_ascii=False)

OVERREACH_HYPOTHESIS = json.dumps({
    "claims": {"nested": True, "discrete": True},
    "predictions": [{"experiment": "anisotropy", "direction": "above",
                     "threshold": 1.01, "because": "異方性"}],
    "reasoning": "この世界は入れ子だと感じる",
}, ensure_ascii=False)


class FakeBackend:
    """決められた出力を返す基体。呼ばれた回数と中身を記録する。"""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def generate(self, prompts, config):
        out = []
        for _ in prompts:
            out.append(self.outputs[self.calls % len(self.outputs)])
            self.calls += 1
        return tuple(out)

    def identity(self):
        return "fake-substrate"


def drain(run: sv.Run):
    events = []
    while True:
        try:
            ev = run.events.get_nowait()
        except queue.Empty:
            break
        if ev is None:
            break
        events.append(ev)
    return events


def run_one(outputs, worlds=1, seed=7):
    run = sv.Run(run_id="t")
    sv._investigate(run, FakeBackend(outputs), "fake-substrate",
                    worlds=worlds, seed=seed, trials=200)
    return drain(run)


# --- 核心: 開封前に真値を流さない --------------------------------------------

def test_no_event_before_reveal_carries_the_truth():
    """イベント列の水準で二重盲検を守る。画面で隠しても、開発者ツールで
    イベントを覗けば真値が見える —— それでは盲検が演出になる。
    """
    events = run_one([VALID_HYPOTHESIS], worlds=3, seed=11)
    for world_events in _split_by_world(events):
        seen_reveal = False
        for ev in world_events:
            if ev["type"] == "reveal":
                seen_reveal = True
                continue
            if seen_reveal:
                continue
            text = json.dumps(ev, ensure_ascii=False)
            # 真値の器そのもの(truth / active)が reveal 前に現れないこと
            assert '"truth"' not in text, f"reveal 前に truth が漏れた: {ev['type']}"
            assert '"active"' not in text, f"reveal 前に active が漏れた: {ev['type']}"


def _split_by_world(events):
    groups, current = [], []
    for ev in events:
        if ev["type"] == "seal" and current:
            groups.append(current)
            current = []
        current.append(ev)
    if current:
        groups.append(current)
    return groups


def test_reveal_carries_truth_and_scoring():
    events = run_one([VALID_HYPOTHESIS], worlds=1)
    reveal = [e for e in events if e["type"] == "reveal"]
    assert len(reveal) == 1
    r = reveal[0]
    assert "active" in r and "fair_score" in r and "false_positive" in r
    assert "findings_are_void" in r


# --- 順序: 凍結が測定と照合の間に立つ ----------------------------------------

def test_event_order_is_seal_deal_probe_think_freeze_judge_reveal():
    events = run_one([VALID_HYPOTHESIS], worlds=1)
    kinds = [e["type"] for e in events]
    assert kinds[0] == "seal"
    assert kinds.index("deal") < kinds.index("probe")
    assert kinds.index("probe") < kinds.index("thinking")
    assert kinds.index("thinking") < kinds.index("freeze")
    assert kinds.index("freeze") < kinds.index("judgment")
    assert kinds.index("judgment") < kinds.index("consensus")
    assert kinds.index("consensus") < kinds.index("reveal")
    assert kinds[-1] == "finished"


def test_probes_carry_values_but_not_meanings():
    """測定値は流すが、どの性質を探る実験かは流さない(表引きにさせない)。"""
    events = run_one([VALID_HYPOTHESIS], worlds=1)
    for ev in events:
        if ev["type"] == "probe":
            assert "value" in ev and "experiment" in ev
            text = json.dumps(ev, ensure_ascii=False)
            for key in sb.PROPERTY_KEYS:
                assert f'"{key}"' not in text


# --- 生の推論が流れる --------------------------------------------------------

def test_hypothesis_events_carry_the_raw_reasoning():
    """**これが観測対象である。** 要約せず、そのまま流す。"""
    events = run_one([VALID_HYPOTHESIS], worlds=1)
    hyps = [e for e in events if e["type"] == "hypothesis"]
    assert hyps, "仮説イベントが無い"
    for h in hyps:
        assert h["reasoning"], "推論が空"
        assert h["predictions"], "予測が無い仮説が通っている"


def test_overreach_is_named_in_the_event():
    events = run_one([OVERREACH_HYPOTHESIS], worlds=1)
    hyps = [e for e in events if e["type"] == "hypothesis"]
    assert any("nested" in h["overreach"] for h in hyps)


# --- 失敗を隠さない ----------------------------------------------------------

def test_rejected_output_becomes_a_rejected_event_not_silence():
    events = run_one(["これはJSONではない日本語の文章である"], worlds=1)
    rejected = [e for e in events if e["type"] == "rejected"]
    assert len(rejected) == len(sv.AGENT_IDS)
    aborted = [e for e in events if e["type"] == "aborted"]
    assert aborted, "全滅したのに走行が続いたことになっている"


def test_mixed_success_still_investigates():
    """2件以上仮説が立てば周回は成立する。"""
    events = run_one([VALID_HYPOTHESIS, "壊れた出力",
                      VALID_HYPOTHESIS, "壊れた出力"], worlds=1)
    assert [e for e in events if e["type"] == "reveal"]
    assert [e for e in events if e["type"] == "rejected"]


def test_run_always_closes():
    run = sv.Run(run_id="t")
    sv._investigate(run, FakeBackend([VALID_HYPOTHESIS]), "fake",
                    worlds=1, seed=1, trials=100)
    assert run.done


def test_internal_failure_becomes_an_error_event():
    """走行中の失敗は画面へ届く。黙って止まらない。"""
    class Exploding:
        def generate(self, prompts, config):
            raise RuntimeError("基体が爆発した")
        def identity(self):
            return "boom"

    run = sv.Run(run_id="t")
    sv._investigate(run, Exploding(), "boom", worlds=1, seed=1, trials=100)
    events = drain(run)
    assert any(e["type"] == "error" for e in events)
    assert run.done


# --- 人格: 結論を与えない ----------------------------------------------------

def test_personas_carry_no_conclusions():
    for persona in sv.PERSONAS.values():
        for key in sb.PROPERTY_KEYS:
            assert key not in persona, f"人格に性質名 {key} が書かれている"
        assert "ある」と答え" not in persona
