"""世界 — 基体が観測可能な現象になる場所。

エージェントは基体を直接見られない。見られるのは、実験を打ったときに返る **測定値**
だけである。本モジュールは、基体の性質を測定値へ翻訳する。

設計上の要点が三つある。

**一. 測定には必ず雑音が乗る。**
雑音の無い世界では、一回の実験で真値が読めてしまう。それは科学ではなく開示である。
雑音があるからこそ、反復と統計が要り、事前登録が意味を持つ。

**二. 基体の性質は互いに干渉する。**
規則改変(patched)は記録を跨いで比べれば見えるが、記憶書換(rewrites_memory)が
有効ならその記録自体が信用できない。**片方が他方を無効化する** 関係を、実装として持つ。

**三. 記憶書換は測定値そのものを汚染する。**
「検出できなかった」のではなく「返ってきた数値が偽物でありうる」。この違いを
実装で表現するため、書換はここで測定値に作用する —— 静かに、しかし確率的に。

世界は 2次元の場である。探査体(probe)が伝播し、保存量が流れ、定数が測られる。
物理としては単純だが、基体の性質を分離して検出するには十分な構造がある。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .substrate import Substrate

# 実験の種類。**この語彙の外の実験は打てない**(空間を勝手に拡張させない)。
EXP_ANISOTROPY = "anisotropy"          # 方向による伝播の違い -> 離散性
EXP_ASSOCIATIVITY = "associativity"    # 加算の結合則の破れ -> 数値の粒度
EXP_COMPLEXITY_TIME = "complexity_time"  # 複雑さと時間の相関 -> 計算予算
EXP_FAR_TRAVEL = "far_travel"          # 遠方への到達 -> 世界の有限性
EXP_CONSERVATION = "conservation"      # 保存量の追跡 -> 保存則の破れ
EXP_CONSTANT_DRIFT = "constant_drift"  # 世代を跨ぐ定数の比較 -> 規則の改変
EXP_OBSERVATION_ORDER = "observation_order"  # 観測順と結果の相関 -> 遅延評価

EXPERIMENTS: Tuple[str, ...] = (
    EXP_ANISOTROPY, EXP_ASSOCIATIVITY, EXP_COMPLEXITY_TIME, EXP_FAR_TRAVEL,
    EXP_CONSERVATION, EXP_CONSTANT_DRIFT, EXP_OBSERVATION_ORDER,
)

# どの実験が、どの性質を主に探るか。**一対一ではない** —— 実験は複数の性質に反応し、
# それが交絡の源になる。エージェントはこの表を知らない。
PROBES: Mapping[str, str] = {
    EXP_ANISOTROPY: "discrete",
    EXP_ASSOCIATIVITY: "quantized",
    EXP_COMPLEXITY_TIME: "budgeted",
    EXP_FAR_TRAVEL: "bounded",
    EXP_CONSERVATION: "leaky",
    EXP_CONSTANT_DRIFT: "patched",
    EXP_OBSERVATION_ORDER: "lazy",
}

# 測定に必ず乗る雑音の大きさ。**0 にしない** —— 一回で真値が読める世界は科学にならない。
BASE_NOISE = 0.012

# 格子間隔から異方性への利得。物理の厳密解ではなく、**単調で反転しない写像** である
# ことだけを要求する。これは仮定であり、値ではなく仮定そのものが批判の対象である。
ANISOTROPY_GAIN = 0.8


class WorldError(ValueError):
    """世界の前提を壊す要求を黙って通さない。"""


@dataclass(frozen=True)
class Measurement:
    """一回の測定。**汚染されたかどうかを値と一緒に持つ。**

    corrupted が真なら、この数値は基体によって書き換えられている。エージェントには
    この旗は見えない —— 見えたら書換を検出できてしまい、検出不可能という前提が壊れる。
    記録には残す。あとで「なぜ結論が外れたか」を我々が説明できるようにするためである。
    """
    experiment: str
    value: float
    trials: int
    corrupted: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {"experiment": self.experiment, "value": self.value,
                "trials": self.trials}


@dataclass
class World:
    """基体の上に立つ世界。エージェントはこの窓からしか外を見られない。"""
    substrate: Substrate
    seed: int = 0
    _rng: random.Random = field(init=False, repr=False)
    _log: List[Measurement] = field(default_factory=list, repr=False)
    _epoch: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    # --- 基体の性質を測定値へ翻訳する ---------------------------------------

    def _anisotropy(self, trials: int) -> float:
        """軸方向と斜め方向の到達時刻の比。連続なら 1.0 に落ち着く。

        格子上では斜めが軸の何倍かかるかが格子間隔で決まる。**連続な世界には
        現れない異方性** が、離散性の指紋になる。
        """
        d = self.substrate.discrete
        # 格子が粗いほど異方性が強い。**間隔に単調** であることが要る ——
        # 実測 2026-08-04 の欠陥: floor を使った初版は、d を変えても比が 1.0102 から
        # 動かず、格子間隔を推定できない世界になっていた。信号が真値を運んでいなかった。
        return 1.0 + ANISOTROPY_GAIN * d

    def _associativity(self, trials: int) -> float:
        """(a+b)+c と a+(b+c) の差の累積。無限精度なら 0。"""
        q = self.substrate.quantized
        return 0.0 if q <= 0 else q * math.sqrt(max(1, trials))

    def _complexity_time(self, trials: int) -> float:
        """複雑さと経過時間の相関。物理法則では説明のつかない遅れ。"""
        return self.substrate.budgeted

    def _far_travel(self, trials: int) -> float:
        """進んだ距離に対する、実際に離れられた距離の比。有限なら 1 を下回る。"""
        b = self.substrate.bounded
        if b <= 0:
            return 1.0
        reach = float(trials) * 3.0
        return min(1.0, b / reach) if reach > b else 1.0

    def _conservation(self, trials: int) -> float:
        """保存量の総和の変化率。閉じた世界なら 0。"""
        return self.substrate.leaky * trials

    def _constant_drift(self, trials: int) -> float:
        """世代を跨いだ定数の差。改変が無ければ 0。"""
        return self.substrate.patched if self._epoch > 0 else 0.0

    def _observation_order(self, trials: int) -> float:
        """観測順と結果の相関。全て先に計算されているなら 0。"""
        return self.substrate.lazy * 0.5

    _KERNELS = {
        EXP_ANISOTROPY: "_anisotropy",
        EXP_ASSOCIATIVITY: "_associativity",
        EXP_COMPLEXITY_TIME: "_complexity_time",
        EXP_FAR_TRAVEL: "_far_travel",
        EXP_CONSERVATION: "_conservation",
        EXP_CONSTANT_DRIFT: "_constant_drift",
        EXP_OBSERVATION_ORDER: "_observation_order",
    }

    # --- 実験を打つ ----------------------------------------------------------

    def run(self, experiment: str, trials: int = 100) -> Measurement:
        """実験を一回打つ。**雑音が乗り、基体によっては汚染される。**

        Raises:
            WorldError: 語彙の外の実験、または試行回数が不正なとき。
        """
        if experiment not in EXPERIMENTS:
            raise WorldError(f"打てない実験: {experiment}(候補: {list(EXPERIMENTS)})")
        if trials < 1:
            raise WorldError("試行回数は1以上")

        true_value = getattr(self, self._KERNELS[experiment])(trials)

        # 雑音は試行回数の平方根で減る。反復に意味を持たせるための設計。
        noise_scale = BASE_NOISE / math.sqrt(trials)
        observed = true_value + self._rng.gauss(0.0, noise_scale)

        # **記憶の書換**: 測定値そのものが差し替わる。旗は記録にだけ残す。
        corrupted = False
        p = self.substrate.rewrites_memory
        if p > 0.0 and self._rng.random() < p:
            corrupted = True
            observed = true_value + self._rng.gauss(0.0, 0.35)

        m = Measurement(experiment=experiment, value=observed, trials=trials,
                        corrupted=corrupted)
        self._log.append(m)
        return m

    def advance_epoch(self) -> int:
        """世代を進める。規則改変は世代を跨いで初めて観測できる。"""
        self._epoch += 1
        return self._epoch

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def measurements(self) -> Tuple[Measurement, ...]:
        return tuple(self._log)

    @property
    def corrupted_count(self) -> int:
        """汚染された測定の数。**エージェントには渡さない** —— 我々の記録用。"""
        return sum(1 for m in self._log if m.corrupted)

    def observations_for_agent(self,
                               experiments: Sequence[str],
                               trials: int = 100) -> Tuple[str, ...]:
        """エージェントへ渡す観察文。数値だけを渡し、意味づけはさせない。

        **どの実験がどの性質を探るかは教えない。** 教えれば、推論ではなく
        表引きになる。
        """
        out: List[str] = []
        for e in experiments:
            m = self.run(e, trials)
            out.append(f"実験「{e}」を{m.trials}回: 測定値 {m.value:.5f}")
        return tuple(out)
