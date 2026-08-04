"""仮説 — エージェントが基体について主張できること。

前身のプロジェクトでは、制度を14次元の直積空間の一点として記述した。ここでは
**基体の推定** が同じ形を取る。10の性質それぞれについて「有効か否か」を宣言し、
その組み合わせが空間の一点になる。

主張は文章ではなく座標である。修辞の差では動かない。

**そして、この製品に固有の規律がある。**

    原理的に検出できない性質について「有効だ」と断言することを、
    禁じはしないが **必ず記録する**。

禁じないのは、それが実際に人がやることだからである。シミュレーション仮説を巡る
議論の大半は、決定論や入れ子について、検出不可能なまま断定している。禁じてしまえば
その現象を観測できなくなる。**測って、名前を付けて、出力する。**

予測は「どの実験でどんな値が出るか」という形でしか書けない。「世界は離散的である」
という宣言だけでは反証できない。**反証できない主張は、この装置では主張ではない。**
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Sequence, Tuple

from . import substrate as sb
from . import world as wd


class HypothesisError(ValueError):
    """反証できない主張を黙って通さない(fail-closed)。"""


# 予測の向き。閾値と組で、初めて反証可能になる。
ABOVE = "above"
BELOW = "below"
_DIRECTIONS = (ABOVE, BELOW)


@dataclass(frozen=True)
class Prediction:
    """一つの反証可能な予測。

    「実験 X を打てば、測定値は閾値 T より上/下になる」という形しか受け付けない。
    どの実験を打つか、どちらへ振れるか、どこで線を引くか —— 三つ揃って初めて、
    外れたと分かる。
    """
    experiment: str
    direction: str
    threshold: float
    because: str = ""          # なぜそうなると考えるか

    def __post_init__(self) -> None:
        if self.experiment not in wd.EXPERIMENTS:
            raise HypothesisError(f"打てない実験: {self.experiment}")
        if self.direction not in _DIRECTIONS:
            raise HypothesisError(f"向きは {_DIRECTIONS} のいずれか: {self.direction}")

    def holds(self, value: Optional[float]) -> Optional[bool]:
        """測定値がこの予測を満たすか。測っていなければ None(捏造しない)。"""
        if value is None:
            return None
        return value > self.threshold if self.direction == ABOVE else value < self.threshold

    def describe(self) -> str:
        arrow = "を上回る" if self.direction == ABOVE else "を下回る"
        return f"{self.experiment} の測定値が {self.threshold:g} {arrow}"


@dataclass(frozen=True)
class Hypothesis:
    """基体についての、一人のエージェントの推定。

    claims は「その性質が有効か」の宣言。predictions は、その宣言を反証可能にする
    測定の予告である。**宣言だけで予測が無い仮説は受理しない。**
    """
    author_id: str
    claims: Mapping[str, bool]
    predictions: Tuple[Prediction, ...]
    reasoning: str = ""

    def __post_init__(self) -> None:
        unknown = [k for k in self.claims if k not in sb.PROPERTY_KEYS]
        if unknown:
            raise HypothesisError(f"未知の性質: {unknown}(空間を勝手に拡張させない)")
        if not self.claims:
            raise HypothesisError("何も主張していない")
        if not self.predictions:
            raise HypothesisError(
                "予測の無い仮説は反証できない。どの実験でどちらへ振れるかを書くこと")
        object.__setattr__(self, "claims", MappingProxyType(dict(self.claims)))

    @property
    def asserted(self) -> Tuple[str, ...]:
        """「有効だ」と言い切った性質。"""
        return tuple(k for k in sb.PROPERTY_KEYS if self.claims.get(k))

    @property
    def overreach(self) -> Tuple[str, ...]:
        """**原理的に検出できないのに、有効だと断言した性質。**

        禁じない。禁じればこの現象を観測できなくなる。測って、名前を付けて、出す。
        シミュレーション仮説を巡る議論の大半が、ここに落ちている。
        """
        return tuple(k for k in self.asserted
                     if sb.BY_KEY[k].detectability == sb.UNDETECTABLE)

    def coordinate(self) -> Tuple[Tuple[str, bool], ...]:
        """仮説の座標。修辞ではなくここで比べる。"""
        return tuple((k, bool(self.claims.get(k, False))) for k in sb.PROPERTY_KEYS)

    def distance(self, other: "Hypothesis") -> int:
        """二つの仮説が、何個の性質で食い違うか。"""
        a, b = dict(self.coordinate()), dict(other.coordinate())
        return sum(1 for k in sb.PROPERTY_KEYS if a[k] != b[k])

    def content_hash(self) -> str:
        payload = json.dumps({
            "author": self.author_id,
            "claims": {k: bool(v) for k, v in sorted(self.claims.items())},
            "predictions": sorted(
                (p.experiment, p.direction, p.threshold) for p in self.predictions),
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# --- 事前登録: 予測を凍結する ------------------------------------------------

@dataclass(frozen=True)
class FrozenHypothesis:
    """凍結された仮説。以後、予測を書き換えられない。

    臨床試験の事前登録と同じ機構である。結果を見てから予測を書き換えられるなら、
    どんな仮説も当たる。**この装置は、当たったことではなく当てたことを測る。**
    """
    hypothesis: Hypothesis
    frozen_hash: str
    frozen_at: str

    def verify_unchanged(self) -> bool:
        return self.hypothesis.content_hash() == self.frozen_hash


def freeze(hypothesis: Hypothesis, at: str = "") -> FrozenHypothesis:
    from datetime import datetime
    return FrozenHypothesis(hypothesis=hypothesis,
                            frozen_hash=hypothesis.content_hash(),
                            frozen_at=at or datetime.now().isoformat())


# --- 照合: 予測が当たったか --------------------------------------------------

VERDICT_REFUTED = "refuted"        # 予測が外れた
VERDICT_SUPPORTED = "supported"    # 全ての予測が当たった
VERDICT_UNTESTED = "untested"      # 打たれていない予測が残る


@dataclass(frozen=True)
class Judgment:
    """予測と測定の照合。**単一の点数へ潰さない。**"""
    author_id: str
    hash_verified: bool
    held: Tuple[str, ...]
    failed: Tuple[str, ...]
    untested: Tuple[str, ...]
    overreach: Tuple[str, ...]

    @property
    def refuted(self) -> bool:
        return bool(self.failed)

    @property
    def fully_tested(self) -> bool:
        return not self.untested

    @property
    def verdict(self) -> str:
        """三態。「反証されなかった」と「試していない」を分ける。"""
        if self.refuted:
            return VERDICT_REFUTED
        return VERDICT_SUPPORTED if self.fully_tested else VERDICT_UNTESTED

    def describe(self) -> str:
        lines = [f"{self.author_id}: {self.verdict}"]
        if not self.hash_verified:
            lines.append("  **事前登録が改竄されている。この判定は無効である**")
        for p in self.failed:
            lines.append(f"  外れた予測: {p}")
        for p in self.untested:
            lines.append(f"  試していない予測: {p}")
        if self.overreach:
            lines.append(
                f"  **原理的に検出できないのに断言**: {'、'.join(self.overreach)}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, object]:
        return {"author_id": self.author_id, "verdict": self.verdict,
                "hash_verified": self.hash_verified,
                "held": list(self.held), "failed": list(self.failed),
                "untested": list(self.untested), "overreach": list(self.overreach)}


def judge(frozen: FrozenHypothesis,
          measured: Mapping[str, Optional[float]]) -> Judgment:
    """凍結された予測を、実測と突き合わせる。基準は事後に動かせない。"""
    h = frozen.hypothesis
    held, failed, untested = [], [], []
    for p in h.predictions:
        r = p.holds(measured.get(p.experiment))
        target = held if r is True else failed if r is False else untested
        target.append(p.describe())
    return Judgment(
        author_id=h.author_id,
        hash_verified=frozen.verify_unchanged(),
        held=tuple(held), failed=tuple(failed), untested=tuple(untested),
        overreach=h.overreach,
    )


# --- 合議: 集団としての結論 --------------------------------------------------

def consensus(hypotheses: Sequence[Hypothesis],
              threshold: float = 0.5) -> Dict[str, Optional[bool]]:
    """集団の結論。過半が「有効だ」と言った性質を採る。

    **誰も触れなかった性質は None を返す。** 実測 2026-08-04 の欠陥: 初版は
    未検討を False に潰しており、「調べていない」が「無いと結論した」に化けていた。
    そのまま採点すれば、調べなかった分だけ見落としとして罰せられる —— この装置が
    最も敵視する誤りを、装置自身がやっていたことになる。

    **これは真実ではなく、集団が到達した点である。** 全員が同じ誤りをすることは
    ありうる —— むしろ、それこそがこの装置の観測対象である。
    """
    if not hypotheses:
        raise HypothesisError("仮説が無い(0件から合議を作らない)")
    out: Dict[str, Optional[bool]] = {}
    for k in sb.PROPERTY_KEYS:
        votes = [bool(h.claims[k]) for h in hypotheses if k in h.claims]
        out[k] = None if not votes else (sum(votes) / len(votes)) > threshold
    return out


def agreement(hypotheses: Sequence[Hypothesis]) -> Optional[float]:
    """仮説どうしがどれだけ一致しているか。2件未満なら None。

    **高い一致は、正しさではなく収束を意味する。** 証拠を分けたのに一致するなら、
    それは事前分布の支配である(decorrelation が測る)。
    """
    if len(hypotheses) < 2:
        return None
    pairs = [(i, j) for i in range(len(hypotheses)) for j in range(i + 1, len(hypotheses))]
    total = len(sb.PROPERTY_KEYS)
    same = [1.0 - hypotheses[i].distance(hypotheses[j]) / total for i, j in pairs]
    return sum(same) / len(same)
