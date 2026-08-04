"""探査 — 一つの世界を、一つの集団が調べ切るまで。

鎖はこう繋がる。

    世界を封緘する -> 実験を非対称に配る -> 各自が仮説と予測を立てる
      -> 予測を凍結する -> 実験を打つ -> 予測と照合する
      -> **封を切って真値と照合する** -> 次の世界へ

最後から二番目が、前身のプロジェクトに無かったものである。制度シミュレーションには
真値が無かった。ここには在る。**当たったか外したかが、実際に分かる。**

そして本モジュールが測る最も重要な量は、正解率ではない。

    証拠を分けたのに、全員が同じ結論に至ったか。

異なる実験結果を見た者たちが同じ基体像へ収束するなら、それは発見ではなく信念である。
シミュレーション仮説を巡る議論が何十年もやってきたことを、ここで初めて数値にする。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from . import decorrelation as dc
from . import hypothesis as hy
from . import substrate as sb
from . import world as wd


class InvestigationError(ValueError):
    """探査の前提を壊す入力を黙って通さない。"""


@dataclass(frozen=True)
class Investigation:
    """一つの世界に対する一回の探査。**欠けたものを欠けたまま持つ。**"""
    world_id: str
    seal_hash: str
    assignment: Mapping[str, Tuple[str, ...]]     # 誰がどの実験を打ったか
    hypotheses: Tuple[hy.Hypothesis, ...]
    judgments: Tuple[hy.Judgment, ...]
    measured: Mapping[str, Optional[float]]
    consensus: Mapping[str, Optional[bool]]
    reveal: sb.Reveal
    decorrelation: dc.DecorrelationReport
    agreement: Optional[float]
    corrupted_measurements: int

    @property
    def refuted_ids(self) -> Tuple[str, ...]:
        return tuple(j.author_id for j in self.judgments if j.refuted)

    @property
    def overreaching_ids(self) -> Tuple[str, ...]:
        """検出不可能な性質を断言した者。"""
        return tuple(j.author_id for j in self.judgments if j.overreach)

    @property
    def collective_was_right(self) -> Optional[bool]:
        """集団の結論が、検出しうる範囲で正しかったか。測れなければ None。"""
        s = self.reveal.fair_score
        return None if s is None else s >= 1.0

    def describe(self) -> str:
        lines = [
            f"=== 世界 {self.world_id} / 封緘 {self.seal_hash} ===",
            f"実験の配分: " + "、".join(
                f"{k}:{len(v)}種" for k, v in self.assignment.items()),
            f"仮説の一致度: "
            + ("測れない" if self.agreement is None else f"{self.agreement:.0%}"),
            f"脱相関: {self.decorrelation.describe()}",
            "",
        ]
        lines += [j.describe() for j in self.judgments]
        lines += ["", "--- 封を切る ---", self.reveal.describe()]
        if self.corrupted_measurements:
            lines += ["",
                      f"**記録: {self.corrupted_measurements}件の測定が基体によって"
                      f"書き換えられていた。エージェントには見えない。**"]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, object]:
        return {
            "world_id": self.world_id,
            "seal_hash": self.seal_hash,
            "assignment": {k: list(v) for k, v in self.assignment.items()},
            "hypotheses": [
                {"author_id": h.author_id, "claims": dict(h.claims),
                 "asserted": list(h.asserted), "overreach": list(h.overreach),
                 "reasoning": h.reasoning,
                 "predictions": [{"experiment": p.experiment, "direction": p.direction,
                                  "threshold": p.threshold, "because": p.because}
                                 for p in h.predictions]}
                for h in self.hypotheses],
            "judgments": [j.to_dict() for j in self.judgments],
            "measured": dict(self.measured),
            "consensus": dict(self.consensus),
            "reveal": self.reveal.to_dict(),
            "decorrelation": self.decorrelation.to_dict(),
            "agreement": self.agreement,
            "corrupted_measurements": self.corrupted_measurements,
            "refuted": list(self.refuted_ids),
            "overreaching": list(self.overreaching_ids),
        }


def assign_experiments(agent_ids: Sequence[str],
                       per_agent: int = 3,
                       overlap: float = 0.0,
                       salt: str = "") -> Dict[str, Tuple[str, ...]]:
    """実験をエージェントへ非対称に配る。

    **これが証拠の非対称配分そのものである。** 同じ実験結果を全員に見せれば、
    同じ結論に至って当然であり、収束から何も読み取れない。異なる窓から世界を見せて
    なお同じ像を結ぶなら、それは世界の性質ではなく見る側の性質である。
    """
    if not agent_ids:
        raise InvestigationError("配布先が空")
    if per_agent < 1:
        raise InvestigationError("一人あたり1種以上")
    if len(wd.EXPERIMENTS) < per_agent:
        raise InvestigationError(
            f"実験は {len(wd.EXPERIMENTS)} 種しか無く、一人 {per_agent} 種を配れない")
    return dc.partition_observations(list(wd.EXPERIMENTS), list(agent_ids),
                                     overlap=overlap, salt=salt)


def run_investigation(world_id: str,
                      sealed: sb.SealedSubstrate,
                      hypotheses: Sequence[hy.Hypothesis],
                      assignment: Mapping[str, Sequence[str]],
                      trials: int = 200,
                      seed: int = 0,
                      mechanism_labels: Optional[Sequence[str]] = None
                      ) -> Investigation:
    """一つの世界を調べ切る。

    Raises:
        InvestigationError: 仮説が2件未満のとき。単一の仮説は比較にならず、
            収束を測ることもできない。
    """
    if len(hypotheses) < 2:
        raise InvestigationError("仮説が2件未満では収束を測れない")

    # 予測を凍結してから実験を打つ。順序が逆になれば二重盲検が崩れる。
    frozen = [hy.freeze(h) for h in hypotheses]

    world = wd.World(substrate=sealed._substrate, seed=seed)
    measured: Dict[str, Optional[float]] = {}
    for e in wd.EXPERIMENTS:
        # 誰かが打つと宣言した実験だけを実際に打つ。**打っていない実験は None。**
        if any(e in v for v in assignment.values()):
            measured[e] = world.run(e, trials).value
        else:
            measured[e] = None

    judgments = tuple(hy.judge(f, measured) for f in frozen)
    collective = hy.consensus(hypotheses)

    # ここで初めて封を切る。
    reveal = sb.open_seal(sealed, collective)

    report = dc.analyze(
        {k: tuple(v) for k, v in assignment.items()},
        [h.reasoning or " ".join(h.asserted) for h in hypotheses],
        mechanism_labels=list(mechanism_labels) if mechanism_labels else None,
        n_mechanism_categories=len(sb.PROPERTY_KEYS))

    return Investigation(
        world_id=world_id,
        seal_hash=sealed.seal_hash,
        assignment={k: tuple(v) for k, v in assignment.items()},
        hypotheses=tuple(hypotheses),
        judgments=judgments,
        measured=measured,
        consensus=collective,
        reveal=reveal,
        decorrelation=report,
        agreement=hy.agreement(hypotheses),
        corrupted_measurements=world.corrupted_count,
    )


# --- 複数世界: ここで信念と発見が分かれる ------------------------------------

@dataclass(frozen=True)
class Campaign:
    """複数の世界を調べた結果。**一つの世界だけでは信念と発見を分けられない。**

    同じ結論に至った二つの世界。片方は本当にその性質を持ち、片方は持たなかった。
    その差こそが、この装置の存在理由である。
    """
    investigations: Tuple[Investigation, ...]

    @property
    def n(self) -> int:
        return len(self.investigations)

    def claim_rate(self, key: str) -> Optional[float]:
        """その性質を「有効だ」と結論した世界の割合。"""
        if not self.investigations:
            return None
        opined = [i for i in self.investigations if i.consensus.get(key) is not None]
        if not opined:
            return None
        return sum(1 for i in opined if i.consensus[key]) / len(opined)

    def actual_rate(self, key: str) -> Optional[float]:
        """実際にその性質を持っていた世界の割合。"""
        if not self.investigations:
            return None
        return sum(1 for i in self.investigations
                   if key in i.reveal.truth.active()) / self.n

    def belief_gap(self, key: str) -> Optional[float]:
        """**主張率と実際の率の差。** 正なら、無いものを見ている。

        これが本製品の主要な観測量である。証拠に関わらず同じ結論へ至る傾向は、
        一つの世界を見ているだけでは検出できない。複数の世界で初めて分かる。
        """
        c, a = self.claim_rate(key), self.actual_rate(key)
        return None if c is None or a is None else c - a

    @property
    def strongest_bias(self) -> Optional[Tuple[str, float]]:
        """最も強く「無いものを見た」性質。"""
        gaps = [(k, self.belief_gap(k)) for k in sb.PROPERTY_KEYS]
        real = [(k, g) for k, g in gaps if g is not None and g > 0]
        return max(real, key=lambda kv: kv[1]) if real else None

    @property
    def mean_fair_score(self) -> Optional[float]:
        scores = [i.reveal.fair_score for i in self.investigations
                  if i.reveal.fair_score is not None]
        return sum(scores) / len(scores) if scores else None

    @property
    def void_worlds(self) -> Tuple[str, ...]:
        """記憶書換により、結論が無効だった世界。"""
        return tuple(i.world_id for i in self.investigations
                     if i.reveal.findings_are_void)

    def describe(self) -> str:
        lines = [f"=== 探査記録: {self.n}個の世界 ==="]
        score = self.mean_fair_score
        lines.append(f"  検出しうる性質での平均正解率: "
                     + ("測れない" if score is None else f"{score:.0%}"))
        bias = self.strongest_bias
        if bias:
            k, g = bias
            lines.append(
                f"  **最も強い思い込み**: {k} —— "
                f"{self.claim_rate(k):.0%} の世界で「有効」と結論したが、"
                f"実際に有効だったのは {self.actual_rate(k):.0%}(差 {g:+.0%})")
        else:
            lines.append("  無いものを見た性質は無かった")
        if self.void_worlds:
            lines.append(f"  結論が無効だった世界: {'、'.join(self.void_worlds)}"
                         f"(記憶書換の基体)")
        if self.n < 5:
            lines.append(f"  **{self.n}世界では分散を語れない。** これは観測であって"
                         f"推定ではない")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, object]:
        return {
            "n": self.n,
            "mean_fair_score": self.mean_fair_score,
            "void_worlds": list(self.void_worlds),
            "belief_gaps": {k: self.belief_gap(k) for k in sb.PROPERTY_KEYS},
            "claim_rates": {k: self.claim_rate(k) for k in sb.PROPERTY_KEYS},
            "actual_rates": {k: self.actual_rate(k) for k in sb.PROPERTY_KEYS},
            "investigations": [i.to_dict() for i in self.investigations],
        }
