"""時間と不確実性 — 遠未来を断定しない義務(CONCEPT.md 7節)。

要件16は「百年後から千年後について、断定的な未来ではなく条件に応じた複数の分岐を示すこと」を
求める。これを心構えではなく機構として強制する。

不確実性は時間とともに単調増加する。したがって:

  - 遠未来は単一軌道ではなくアンサンブル(初期条件・パラメータを摂動した多数走)で計算する
  - アンサンブルの分散が閾値を超えた地平より先では、単一の数値を出すことを *禁止* し、
    「分岐」として複数の可能性を表示する
  - 各時点の表示には必ず不確実性の幅を伴わせる

これは、AGI 側で実装した鮮度表示と同じ規律である。分からないことを分からないと表示する
機構がなければ、システムは必ず断定する。正直さは善意ではなく機構によってしか担保されない。
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

# 時間解像度の階層(CONCEPT.md 7.1)。遠くなるほど粗く、最後は分岐でしか語らない。
TIME_BANDS: Tuple[Tuple[str, int, int, int], ...] = (
    ("移行期", 0, 30, 1),        # 移行摩擦・既得権の抵抗・財源・不信・産業転換
    ("世代交代期", 30, 100, 5),  # 教育・所有観・成功の定義の変化
    ("定着期", 100, 300, 25),    # 文化的定着・硬直化・新階級・改革運動
    ("文明期", 300, 1000, 0),    # 分岐でのみ語る(刻み 0 = 単一軌道を出さない)
)

# 分散がこれを超えたら単一値の提示を禁じる(点推定が意味を失う地平)。
DEFAULT_SPREAD_LIMIT = 0.25


class HorizonError(ValueError):
    """不確実性の扱いを壊す入力を黙って通さない(fail-closed)。"""


def band_for(year: int) -> str:
    """その年がどの時間帯に属するか。範囲外は捏造せず例外。"""
    for name, lo, hi, _ in TIME_BANDS:
        if lo <= year < hi:
            return name
    if year == TIME_BANDS[-1][2]:
        return TIME_BANDS[-1][0]
    raise HorizonError(f"扱う時間範囲の外: {year}年")


def step_for(year: int) -> int:
    """その時間帯の刻み幅。0 は『単一軌道を出さない』の意。"""
    for name, lo, hi, step in TIME_BANDS:
        if lo <= year < hi:
            return step
    if year == TIME_BANDS[-1][2]:
        return TIME_BANDS[-1][3]
    raise HorizonError(f"扱う時間範囲の外: {year}年")


def ensemble_spread(values: Sequence[float]) -> Optional[float]:
    """アンサンブルのばらつき(標準偏差)。2走未満なら None(捏造しない)。

    1走しかないのに「不確実性 0」と報告するのは、最も危険な断定である。
    """
    if len(values) < 2:
        return None
    return statistics.pstdev(values)


def refuse_point_estimate(values: Sequence[float],
                          spread_limit: float = DEFAULT_SPREAD_LIMIT) -> bool:
    """単一の数値を出すことを拒むべきか。

    True のとき、呼び出し側は点推定を表示してはならない(分岐として示す)。
    測定不能(1走以下)も拒む —— 不確実性が分からないなら断定はできない。
    """
    s = ensemble_spread(values)
    return True if s is None else s > spread_limit


@dataclass(frozen=True)
class HorizonEstimate:
    """ある年次の推定。点推定は、許されるときだけ与えられる。"""
    year: int
    band: str
    samples: Tuple[float, ...]
    spread: Optional[float]
    point_estimate: Optional[float]   # 拒まれたときは None
    low: Optional[float] = None
    high: Optional[float] = None

    @property
    def is_branching(self) -> bool:
        """断定を拒み、分岐として示すべき地平か。"""
        return self.point_estimate is None

    def describe(self) -> str:
        """表示用。断定できないときは、できないと言う。"""
        if self.is_branching:
            n = len(self.samples)
            if self.spread is None:
                return f"{self.year}年({self.band}): 不確実性を測れていない(走数 {n})"
            return (f"{self.year}年({self.band}): 分岐 —— "
                    f"{self.low:.2f}〜{self.high:.2f} のばらつき(標準偏差 {self.spread:.2f})")
        return (f"{self.year}年({self.band}): {self.point_estimate:.2f} "
                f"(幅 {self.low:.2f}〜{self.high:.2f})")


def estimate_at(year: int, samples: Sequence[float],
                spread_limit: float = DEFAULT_SPREAD_LIMIT) -> HorizonEstimate:
    """アンサンブルから、その年次の推定を作る。断定が許されるかは機構が決める。"""
    if not samples:
        raise HorizonError("標本が空(何も測っていない)")
    vals = tuple(float(v) for v in samples)
    spread = ensemble_spread(vals)
    refuse = refuse_point_estimate(vals, spread_limit)
    band = band_for(year)
    # 文明期は、ばらつきに関わらず単一軌道を出さない(要件16の下限保証)。
    if step_for(year) == 0:
        refuse = True
    return HorizonEstimate(
        year=year, band=band, samples=vals, spread=spread,
        point_estimate=None if refuse else statistics.fmean(vals),
        low=min(vals), high=max(vals),
    )


def uncertainty_is_monotonic(spreads: Sequence[Optional[float]],
                             tolerance: float = 1e-9) -> Optional[bool]:
    """不確実性が時間とともに増加しているか(構造的期待の実測)。

    減っているなら、それはモデルが未来を過信している兆候である(摂動が効いていない、
    または収束を仮定している)。測れない要素があれば None。
    """
    if any(s is None for s in spreads) or len(spreads) < 2:
        return None
    return all(b >= a - tolerance for a, b in zip(spreads, spreads[1:]))


@dataclass(frozen=True)
class BranchReport:
    """遠未来の複数可能性。単一の未来を描かない。"""
    year: int
    branches: Tuple[Tuple[str, float], ...]   # (分岐の名前, その帰結値)

    @property
    def branch_count(self) -> int:
        return len(self.branches)

    def to_dict(self) -> Dict[str, object]:
        return {"year": self.year,
                "branches": [list(b) for b in self.branches],
                "branch_count": self.branch_count}


def branch_horizon(estimates: Sequence[HorizonEstimate]) -> Optional[int]:
    """これより先は断定してはならない、という最初の年次。

    全て断定可能なら None(= 分岐地平に到達していない)。
    """
    for e in sorted(estimates, key=lambda x: x.year):
        if e.is_branching:
            return e.year
    return None
