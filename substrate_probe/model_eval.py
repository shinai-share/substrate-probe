"""基体の推論を実測する — 形式を守ったことと、考えたことは違う。

これまで報告してきた「成功率 4/4」は **形式遵守率** である。JSONが閉じているか、
選択肢が空間内かを見ているにすぎない。**それは推論の質ではない。**

言語モデルは、指示を無視したまま形式だけ整った出力を出せる。実際そうなった:
現行の「退出権=条件付」をそのまま「変更」として提出した案は、形式的には完璧だった。
形式ゲートを通る出力を数えても、考えたかどうかは分からない。

本モジュールは、通った出力に対して **反証可能な問い** を立てる。

    1. 与えられた証拠を使ったか、それとも事前分布から書いたか(接地)
    2. 割り当てられた探索領域の中で動いたか(指示追従)
    3. 起点を正しく把握したか(基準点の把握)
    4. 指示された変更数を守ったか(制約遵守)
    5. 犠牲にする価値を具体的に述べたか(取引の自覚)

**1 が最も重要である。** 証拠を分けたのに、自分の証拠より他人の証拠に近い文章を
書くなら、その案は与えられた観察からではなく訓練データから来ている。これは
`decorrelation.prior_dominance` を出力ではなく **入力の使われ方** の側から測る。

どの指標も、良い値が出たからといって「賢い」ことを意味しない。測っているのは
**指示と証拠に対する忠実さ** であって、制度案の妥当性ではない。妥当性を測る物差しを
我々は持っていない。持っていないことを、指標の名前で誤魔化さない。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from . import decorrelation as dc
from . import institution_space as isp
from . import simulation_round as sr


class EvalError(ValueError):
    """評価の前提を壊す入力を黙って通さない。"""


# 変更してよい次元数の上限(プロンプトで指示している値)。
MAX_CHANGES = 3


@dataclass(frozen=True)
class ProposalEvaluation:
    """一案あたりの実測。**判定ではなく観測である。**"""
    proposal_id: str
    author_id: str
    own_evidence_similarity: Optional[float]     # 自分の証拠との語彙的近さ
    other_evidence_similarity: Optional[float]   # 他人の証拠との語彙的近さ
    region_adherence: Optional[float]            # 割当領域内で動いた割合
    change_count: int
    within_change_limit: bool
    names_a_sacrifice: bool

    @property
    def evidence_advantage(self) -> Optional[float]:
        """自分の証拠を、他人の証拠よりどれだけ使ったか。

        0 以下なら、その案は与えられた観察から書かれていない。**証拠を分けた意味が
        出力に現れていない** ということであり、prior_dominance を入力側から見た値になる。
        """
        if (self.own_evidence_similarity is None
                or self.other_evidence_similarity is None):
            return None
        return self.own_evidence_similarity - self.other_evidence_similarity

    @property
    def grounded_in_own_evidence(self) -> Optional[bool]:
        adv = self.evidence_advantage
        return None if adv is None else adv > 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "proposal_id": self.proposal_id,
            "author_id": self.author_id,
            "own_evidence_similarity": self.own_evidence_similarity,
            "other_evidence_similarity": self.other_evidence_similarity,
            "evidence_advantage": self.evidence_advantage,
            "grounded_in_own_evidence": self.grounded_in_own_evidence,
            "region_adherence": self.region_adherence,
            "change_count": self.change_count,
            "within_change_limit": self.within_change_limit,
            "names_a_sacrifice": self.names_a_sacrifice,
        }


def _text_of(evidence: Sequence[str]) -> str:
    return " ".join(evidence)


def evaluate_proposal(plan: sr.Plan,
                      own_evidence: Sequence[str],
                      all_evidence: Mapping[str, Sequence[str]],
                      allowed_dimensions: Optional[Sequence[str]] = None,
                      base: Optional[isp.Institution] = None) -> ProposalEvaluation:
    """一案を、指示と証拠への忠実さで測る。妥当性は測らない(測れない)。"""
    author = plan.proposal.author_id
    rationale = plan.proposal.rationale or plan.proposal.problem

    own = dc.claim_similarity(rationale, _text_of(own_evidence)) if own_evidence else None
    others = [v for k, v in all_evidence.items() if k != author and v]
    other = (sum(dc.claim_similarity(rationale, _text_of(v)) for v in others) / len(others)
             if others else None)

    origin = base or isp.CAPITALISM
    changed = plan.proposal.institution.differing_dimensions(origin)
    adherence = None
    if allowed_dimensions:
        allowed = set(allowed_dimensions)
        adherence = (sum(1 for d in changed if d in allowed) / len(changed)
                     if changed else None)

    return ProposalEvaluation(
        proposal_id=plan.proposal.proposal_id,
        author_id=author,
        own_evidence_similarity=own,
        other_evidence_similarity=other,
        region_adherence=adherence,
        change_count=len(changed),
        within_change_limit=1 <= len(changed) <= MAX_CHANGES,
        names_a_sacrifice=bool(plan.proposal.sacrificed_values),
    )


@dataclass(frozen=True)
class ModelEvaluation:
    """基体一つ分の実測。件数が少ないことを隠さない。"""
    substrate: str
    proposals: Tuple[ProposalEvaluation, ...]
    format_success_rate: Optional[float] = None
    no_op_rejections: int = 0

    @property
    def n(self) -> int:
        return len(self.proposals)

    def _mean(self, values: Sequence[Optional[float]]) -> Optional[float]:
        vals = [v for v in values if v is not None]
        return sum(vals) / len(vals) if vals else None

    @property
    def grounding_rate(self) -> Optional[float]:
        """自分の証拠に接地した案の割合。低いほど事前分布から書いている。"""
        flags = [p.grounded_in_own_evidence for p in self.proposals
                 if p.grounded_in_own_evidence is not None]
        return sum(flags) / len(flags) if flags else None

    @property
    def mean_evidence_advantage(self) -> Optional[float]:
        return self._mean([p.evidence_advantage for p in self.proposals])

    @property
    def mean_region_adherence(self) -> Optional[float]:
        return self._mean([p.region_adherence for p in self.proposals])

    @property
    def change_limit_rate(self) -> Optional[float]:
        if not self.proposals:
            return None
        return sum(p.within_change_limit for p in self.proposals) / self.n

    @property
    def sacrifice_rate(self) -> Optional[float]:
        if not self.proposals:
            return None
        return sum(p.names_a_sacrifice for p in self.proposals) / self.n

    def describe(self) -> str:
        def pct(v: Optional[float]) -> str:
            return "測れない" if v is None else f"{v:.0%}"

        def num(v: Optional[float]) -> str:
            return "測れない" if v is None else f"{v:+.3f}"

        lines = [
            f"=== 推論の実測: {self.substrate}(n={self.n}) ===",
            f"  形式遵守率          : {pct(self.format_success_rate)}"
            f"  ← これは推論の質ではない",
            f"  現状追認の棄却      : {self.no_op_rejections}件"
            f"  ← 起点を把握できなかった回数",
            "",
            f"  自分の証拠への接地  : {pct(self.grounding_rate)}",
            f"  証拠の優位(自-他)   : {num(self.mean_evidence_advantage)}"
            f"  ← 0以下なら証拠を分けた意味が出ていない",
            f"  探索領域の遵守      : {pct(self.mean_region_adherence)}",
            f"  変更数の制約遵守    : {pct(self.change_limit_rate)}",
            f"  犠牲の明示          : {pct(self.sacrifice_rate)}",
        ]
        if self.n < 10:
            lines += ["",
                      f"  **n={self.n} では分散を語れない。** これらは観測であって"
                      f"性能の推定ではない。"]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, object]:
        return {
            "substrate": self.substrate,
            "n": self.n,
            "format_success_rate": self.format_success_rate,
            "no_op_rejections": self.no_op_rejections,
            "grounding_rate": self.grounding_rate,
            "mean_evidence_advantage": self.mean_evidence_advantage,
            "mean_region_adherence": self.mean_region_adherence,
            "change_limit_rate": self.change_limit_rate,
            "sacrifice_rate": self.sacrifice_rate,
            "proposals": [p.to_dict() for p in self.proposals],
        }


def evaluate(substrate: str,
             plans: Sequence[sr.Plan],
             evidence: Mapping[str, Sequence[str]],
             regions: Optional[Mapping[str, Sequence[str]]] = None,
             format_success_rate: Optional[float] = None,
             no_op_rejections: int = 0,
             base: Optional[isp.Institution] = None) -> ModelEvaluation:
    """通った案を、指示と証拠への忠実さで測る。

    Raises:
        EvalError: 案が無いとき。0件から性能を語らない。
    """
    if not plans:
        raise EvalError("評価する案が無い(0件から性能を語らない)")
    evals = tuple(
        evaluate_proposal(
            p, evidence.get(p.proposal.author_id, ()), evidence,
            regions.get(p.proposal.author_id) if regions else None, base)
        for p in plans)
    return ModelEvaluation(substrate=substrate, proposals=evals,
                           format_success_rate=format_success_rate,
                           no_op_rejections=no_op_rejections)
