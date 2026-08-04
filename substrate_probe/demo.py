"""複数の世界を調べ、信念と発見を分ける — GPUもAPIも要らない入口。

    python -m substrate_probe.demo [--worlds N]

仮説は固定の探査者集団が立てる(言語モデルを使わない)。ここで見せたいのは
「良い仮説が立った」ことではなく、**装置が働くこと** である ——

    同じ結論に至った二つの世界。片方は本当にその性質を持ち、片方は持たなかった。

一つの世界だけを見ていると、当たったのが実力か信念かは分からない。複数の世界を
跨いで初めて、証拠に関わらず同じ結論へ寄る傾向が数値になる。

言語モデルを繋ぐと、この固定集団が生成された探査者に置き換わる。鎖の形は変わらない。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from typing import List, Optional, Sequence, Tuple

from . import hypothesis as hy
from . import investigation as inv
from . import substrate as sb
from . import world as wd

P = hy.Prediction

AGENT_IDS: Tuple[str, ...] = ("A0", "A1", "A2", "A3")

# 探査者の人格。**結論は与えない** —— 何を重く見るかだけが違う。
PERSONAS = {
    "A0": "測定の異常をまず疑う探査者。装置の癖と世界の性質を区別したがる",
    "A1": "保存則を重んじる探査者。総量が合わないことに敏感である",
    "A2": "世界の果てを気にする探査者。どこまで行けるかを知りたがる",
    "A3": "時間の流れ方に注目する探査者。速さの違いに理由を求める",
}


def build_hypotheses(measured_hint: Optional[float] = None) -> List[hy.Hypothesis]:
    """固定の探査者集団。

    **意図的に、A0とA2は「離散である」に寄せてある。** 証拠に関わらず同じ結論へ
    寄る集団を作ることで、複数世界を跨いだときに信念の隙間が現れる。
    現実の科学者集団にも同じ偏りがあり、それを可視化するのが本装置の目的である。
    """
    return [
        hy.Hypothesis("A0", {"discrete": True},
                      (P(wd.EXP_ANISOTROPY, hy.ABOVE, 1.01,
                         "格子なら方向で伝播が変わる"),),
                      reasoning="伝播の異方性は格子の痕跡だと考える"),
        hy.Hypothesis("A1", {"leaky": True, "discrete": False},
                      (P(wd.EXP_CONSERVATION, hy.ABOVE, 0.5,
                         "漏れがあれば総量が減る"),),
                      reasoning="保存量の減りに注目した"),
        hy.Hypothesis("A2", {"discrete": True, "nested": True},
                      (P(wd.EXP_ANISOTROPY, hy.ABOVE, 1.005, "異方性"),),
                      reasoning="格子であり、さらに入れ子だと感じる"),
        hy.Hypothesis("A3", {"budgeted": True},
                      (P(wd.EXP_COMPLEXITY_TIME, hy.ABOVE, 0.02,
                         "複雑な領域ほど遅れる"),),
                      reasoning="複雑さと時間の相関を見た"),
    ]


def run(worlds: int = 6, seed: int = 20260804,
        trials: int = 4000, verbose: bool = True) -> inv.Campaign:
    """複数の世界を無作為に生成し、同じ集団に調べさせる。"""
    if worlds < 2:
        raise inv.InvestigationError(
            "1世界では信念と発見を分けられない(2以上が要る)")

    rng = random.Random(seed)
    assignment = inv.assign_experiments(AGENT_IDS, overlap=0.0, salt="demo")
    runs: List[inv.Investigation] = []

    for i in range(worlds):
        sealed = sb.seal(sb.random_substrate(rng))
        r = inv.run_investigation(
            world_id=f"W-{i + 1:02d}", sealed=sealed,
            hypotheses=build_hypotheses(), assignment=assignment,
            trials=trials, seed=rng.randrange(10 ** 6))
        runs.append(r)
        if verbose:
            print(r.describe())
            print()

    return inv.Campaign(tuple(runs))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="複数の世界を調べ、信念と発見を分ける")
    parser.add_argument("--worlds", type=int, default=6, help="調べる世界の数")
    parser.add_argument("--trials", type=int, default=4000, help="一実験あたりの試行")
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--json", action="store_true", help="結果をJSONで出す")
    args = parser.parse_args(list(argv) if argv is not None else None)

    campaign = run(worlds=args.worlds, seed=args.seed, trials=args.trials,
                   verbose=not args.json)
    if args.json:
        print(json.dumps(campaign.to_dict(), ensure_ascii=False, indent=2))
        return 0

    print("=" * 62)
    print(campaign.describe())
    print()
    print("  この装置が言えるのは、当てたかどうかまでである。")
    print("  **なぜ当てたのか** —— 証拠からか、信念からか —— は、")
    print("  一つの世界を見ているだけでは決して分からない。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
