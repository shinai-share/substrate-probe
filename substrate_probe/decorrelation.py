"""構造的脱相関 — 同一基体からの疑似多様性を実測し、証拠で破る。

実測(2026-08-02, Qwen3.5-9B / 4体):
  低所得の生活者・企業経営者・未来世代・行政 という異なる観察位置を与えたにもかかわらず、
  採用された3件の主張はほぼ同一だった(r > g / 相続上限の欠如 / 複利による固定化)。
  企業経営者が低所得生活者とほぼ同じ資本集中批判を述べた。現実にはこの二つは鋭く分岐する。

  人格プロンプトは観点の違いを作れなかった。

理屈は単純である。出力 = f(重み, 入力) であり、重みは全エージェントで共有される。
動かせる唯一の梃子は入力である。人格テキストも入力だが弱い —— *証拠を変えていない* からだ。
ゆえに証拠そのものを非対称に配る。

そしてこの配分が、単なる対処以上のものを可能にする:

    配った証拠の重なり(設計値)と、出てきた主張の似かより(実測)を比べれば、
    モデルの事前分布が証拠をどれだけ押しのけているかが数値になる。

証拠を完全に分けたのに主張が一致するなら、それは合意ではなく **事前分布の支配** である。
訓練データが議論を決めているという、見えない連鎖の実測になる。

なお情報の非対称は現実の社会審議そのものでもある。人が意見を違えるのは、部分的には
違うものを見ているからである。原則と現実性がここでも一致する。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple


class DecorrelationError(ValueError):
    """脱相関の前提を壊す入力を黙って通さない(fail-closed)。"""


# --- 証拠の非対称配分 --------------------------------------------------------

def partition_observations(observations: Sequence[str],
                           agent_ids: Sequence[str],
                           overlap: float = 0.0,
                           salt: str = "") -> Dict[str, Tuple[str, ...]]:
    """観察をエージェントへ非対称に配る。決定論的で再現可能。

    Args:
        observations: 全観察。
        agent_ids: 配布先。
        overlap: 0.0 で完全分割(誰も同じ証拠を持たない)、1.0 で全員に全部
            (= 従来の状態)。中間は共有分と専有分の混合。
        salt: 配分を変えるための種(再現性を保ったまま別配分を試せる)。

    Returns:
        {agent_id: 観察の部分集合}

    Raises:
        DecorrelationError: 観察が足りず、誰かが証拠ゼロになるとき。
            証拠なしで主張させるのは、根拠のない断定を強いることであり、
            StructuredClaim が要求する根拠を満たせない。
    """
    if not observations:
        raise DecorrelationError("観察が空(配る証拠が無い)")
    if not agent_ids:
        raise DecorrelationError("配布先が空")
    if len(set(agent_ids)) != len(agent_ids):
        raise DecorrelationError("agent_id が重複している")
    if not 0.0 <= overlap <= 1.0:
        raise DecorrelationError("overlap は 0.0-1.0")

    n_obs, n_ag = len(observations), len(agent_ids)
    n_shared = int(round(n_obs * overlap))
    shared = tuple(observations[:n_shared])
    exclusive = list(observations[n_shared:])

    if not shared and len(exclusive) < n_ag:
        raise DecorrelationError(
            f"観察 {n_obs} 件をエージェント {n_ag} 名へ分けると証拠ゼロの者が出る。"
            f"証拠なしで主張させない(overlap を上げるか観察を増やす)")

    # 専有分は「配り切る」。剰余で散らすと衝突する —— 実測 2026-08-04: 観察12件を4名へ
    # overlap=0.0 で配ったとき、**4名中3名が完全に同一の証拠** を受け取り、12件中6件は
    # 誰にも配られなかった(実測の重なり 0.50)。開始位置を hash で決めて等間隔に拾う方式は、
    # 開始位置が同じ剰余類に落ちた者どうしが同一の集合を引く。
    # 分割は「散らす」のではなく「重複なく分配する」ことで保証する。
    order = sorted(exclusive,
                   key=lambda o: hashlib.sha256(f"{o}|{salt}".encode("utf-8")).hexdigest())
    out: Dict[str, Tuple[str, ...]] = {aid: () for aid in agent_ids}
    for idx, obs in enumerate(order):
        aid = agent_ids[idx % n_ag]
        out[aid] = out[aid] + (obs,)
    return {aid: tuple(dict.fromkeys(shared + out[aid])) for aid in agent_ids}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def evidence_overlap(assignment: Mapping[str, Sequence[str]]) -> Optional[float]:
    """配った証拠の平均重なり(0=完全に別、1=全員同じ)。2名未満なら None。"""
    ids = list(assignment)
    if len(ids) < 2:
        return None
    sets = {i: set(assignment[i]) for i in ids}
    vals = [_jaccard(sets[ids[i]], sets[ids[j]])
            for i in range(len(ids)) for j in range(i + 1, len(ids))]
    return sum(vals) / len(vals)


# --- 出てきた主張の似かより --------------------------------------------------

# 日本語の内容語をおおまかに拾う。
# 注意(実測 2026-08-03): 日本語は空白で区切られないため、漢字とひらがなを同じ文字クラスに
# 入れると正規表現が文全体を1トークンとして飲み込み、類似度が常に0になる。ひらがな(主に助詞・
# 活用語尾)を区切りとして扱い、漢字・カタカナの連なりを内容語として拾う。
_TOKEN = re.compile(r"[一-鿿]{1,}|[ァ-ヿー]{2,}|[A-Za-z]{3,}")
_STOP = frozenset({"こと", "もの", "ため", "よう", "これ", "それ", "ある", "いる",
                   "する", "なる", "れる", "られ", "この", "その", "および"})


def content_tokens(text: str) -> Set[str]:
    """内容語の集合。修辞の差ではなく語る対象の差を見る。"""
    return {t for t in _TOKEN.findall(text) if t not in _STOP}


def claim_similarity(a: str, b: str) -> float:
    """二つの主張の **語彙的** な近さ(0=語が重ならない, 1=同じ語で語る)。

    重要な非対称性(実測 2026-08-03):
        高い値は収束の証拠になる。低い値は脱相関の証拠に **ならない**。

    実測された反例:
        A1「資産の私人所有と相続自由化 -> 過剰集中 -> 低所得層の生存保障の崩壊」
        A3「資本の資産の集積 -> 非正規雇用拡大 -> 生活の不安定化」
      この二つは同じ機構(資産集中 -> 生活の不安定化)を述べているが、共通トークンは
      「資産」のみで類似度 0.036 だった。日本語の複合語が「流動性封鎖」「流動性」の
      ように別トークンへ固まるためである(文字bigramでも 0.05 にしかならず救済不能)。

    ゆえに本関数は片側検出器である。この非対称性は DecorrelationReport.verdict に
    そのまま反映される(低い語彙類似だけでは decorrelated と判定しない)。
    """
    return _jaccard(content_tokens(a), content_tokens(b))


def output_similarity(claims: Sequence[str]) -> Optional[float]:
    """主張群の平均的な語彙の似かより。2件未満なら None(捏造しない)。"""
    if len(claims) < 2:
        return None
    vals = [claim_similarity(claims[i], claims[j])
            for i in range(len(claims)) for j in range(i + 1, len(claims))]
    return sum(vals) / len(vals)


# 分類器が機構を読み取れなかったことを表すラベル。**これは機構名ではない。**
MECHANISM_UNCLASSIFIED = "分類不能"


def mechanism_overlap(labels: Sequence[str]) -> Optional[float]:
    """主張が帰属した **因果機構** の重なり(0=全員別の機構, 1=全員同じ機構)。

    語彙は意味を測れないので、意味は語彙の外で与える。labels は「この主張はどの機構に
    原因を帰属しているか」の分類であり、生成者とは別の工程が付ける(自己確認を避ける)。

    **分類不能を機構として数えない。** 実測(2026-08-04)で、3主張のうち1件が分類不能
    だったとき、旧実装はそれを「他の2件とは別の機構」と数え、機構の重なりを 0.00 に
    押し下げた。分類できなかったことを違いの証拠に使うのは、未検査を安全と読むのと
    同じ誤りである。分類不能を含む対は母数から外し、残りだけで測る。

    Returns:
        分類できた対が1つも無ければ None(測れないものを測れたことにしない)。
    """
    if len(labels) < 2:
        return None
    pairs = [(a, b)
             for i, a in enumerate(labels) for b in labels[i + 1:]
             if a != MECHANISM_UNCLASSIFIED and b != MECHANISM_UNCLASSIFIED]
    if not pairs:
        return None
    return sum(1 for a, b in pairs if a == b) / len(pairs)


# 機構による判定を許す条件。
#
# 問題は標本が小さいことではなく、**分類器が取りこぼしたこと** である。分類できなかった
# 主張が何を言っていたかは分からない以上、残りが一致していても「全員が別の機構を語った」
# とは言えない。取りこぼしが一定割合を超えたら、その測定は判定に使わない。
#
# 実測 2026-08-04: 3案のうち1件(33%)が分類不能。残る1組だけで decorrelated と宣言していた。
MAX_UNCLASSIFIED_FOR_MECHANISM = 0.25
# 対を作るのに最低限必要な数(これを下回ると重なりが定義できない)。
MIN_CLASSIFIED_FOR_MECHANISM = 2


def classified_count(labels: Sequence[str]) -> int:
    """機構を読み取れた主張の数。判定の土台の厚みそのものである。"""
    return sum(1 for x in labels if x != MECHANISM_UNCLASSIFIED)


def comparable_pairs(labels: Sequence[str]) -> int:
    """機構を比べられた対の数。少ないほど推定は揺れる —— それを隠さない。"""
    ok = [x for x in labels if x != MECHANISM_UNCLASSIFIED]
    return len(ok) * (len(ok) - 1) // 2


def chance_overlap(n_categories: Optional[int]) -> Optional[float]:
    """無作為にラベルを貼ったときに期待される重なり。

    **これを示さずに重なりを語るのは不誠実である。** 機構の語彙が8種なら、何も考えずに
    貼っても 1/8 = 0.125 は一致する。観測値がこの水準を超えているかどうかが、
    「収束した」と言えるかの下限であり、0 との比較ではない。
    """
    if not n_categories or n_categories < 1:
        return None
    return 1.0 / n_categories


def unclassified_ratio(labels: Sequence[str]) -> Optional[float]:
    """分類できなかった主張の割合。高いほど機構による判定の根拠が薄い。"""
    if not labels:
        return None
    return sum(1 for x in labels if x == MECHANISM_UNCLASSIFIED) / len(labels)


# --- 事前分布の支配度 --------------------------------------------------------

VERDICT_DECORRELATED = "decorrelated"        # 証拠の差が主張の差になった
VERDICT_PRIOR_DOMINATED = "prior_dominated"  # 証拠を分けたのに主張が一致した
VERDICT_UNDETERMINED = "undetermined"        # 判定に足る測定が無い
# 語彙は割れたが、機構が同じかは未検査。ここを decorrelated と呼ばないことが本質である。
VERDICT_SEMANTICALLY_UNCHECKED = "semantically_unchecked"

# 語彙類似がこの値を下回ると、収束の有無を語彙だけでは判断できない(片側検出器の死角)。
LEXICAL_BLIND_BELOW = 0.30


@dataclass(frozen=True)
class DecorrelationReport:
    """脱相関が効いたかの実測。効いたことにしない。

    prior_dominance は「証拠を分けた度合いに対して、主張がどれだけ似たままか」である。
    高いほど、モデルの事前分布(訓練データ)が与えた証拠を押しのけている。
    """
    evidence_overlap: Optional[float]
    output_similarity: Optional[float]
    n_agents: int
    n_claims: int
    mechanism_overlap: Optional[float] = None
    unclassified_ratio: Optional[float] = None
    classified_count: int = 0
    comparable_pairs: int = 0
    chance_overlap: Optional[float] = None

    @property
    def mechanism_is_usable(self) -> bool:
        """機構の重なりを判定に使えるだけの土台があるか。

        分類器が取りこぼした主張が何を言っていたかは分からない。取りこぼしが
        MAX_UNCLASSIFIED_FOR_MECHANISM を超えるとき、残りが一致していても
        「全員が別の機構を語った」とは言えない —— 値はあっても判定には使わない。
        """
        if (self.mechanism_overlap is None
                or self.classified_count < MIN_CLASSIFIED_FOR_MECHANISM):
            return False
        return (self.unclassified_ratio or 0.0) <= MAX_UNCLASSIFIED_FOR_MECHANISM

    @property
    def convergence(self) -> Optional[float]:
        """主張がどれだけ収束したか。機構の重なりが使えるならそちらを使う。

        語彙は意味を測れない(実測 2026-08-03)。機構ラベルを優先するのは、近似をより良い
        測定で置き換える行為であり、都合の良い方を選ぶことではない。ただし土台が薄ければ
        優先しない —— より良い測定であることと、足りていることは別である。
        """
        return (self.mechanism_overlap if self.mechanism_is_usable
                else self.output_similarity)

    @property
    def prior_dominance(self) -> Optional[float]:
        """事前分布の支配度(0.0-1.0)。測れなければ None。

        証拠の重なりが小さいのに主張が収束しているほど大きくなる。
        evidence_overlap=0(完全に別の証拠)で収束=1(同じ主張)なら 1.0。
        """
        c = self.convergence
        if self.evidence_overlap is None or c is None:
            return None
        return max(0.0, c - self.evidence_overlap)

    @property
    def verdict(self) -> str:
        """三態ではなく四態。『語彙が割れた』を『脱相関できた』と読ませない。

        claim_similarity は片側検出器である —— 高い値は収束の証拠だが、低い値は
        脱相関の証拠ではない。機構ラベルが無いまま語彙類似が低いときは、収束を
        見落としている可能性が残る。そこを未検査と呼ぶ。
        """
        pd = self.prior_dominance
        if pd is None:
            return VERDICT_UNDETERMINED
        if pd > 0.30:
            return VERDICT_PRIOR_DOMINATED
        if not self.mechanism_is_usable and (
                self.output_similarity is None
                or self.output_similarity < LEXICAL_BLIND_BELOW):
            return VERDICT_SEMANTICALLY_UNCHECKED
        return VERDICT_DECORRELATED

    @property
    def exceeds_chance(self) -> Optional[bool]:
        """機構の重なりが偶然水準を超えているか。測れなければ None。

        超えていなければ、それは「同じ機構へ収束した」ではなく「無作為と区別できない」
        である。両者を同じ言葉で呼ばない。
        """
        if self.mechanism_overlap is None or self.chance_overlap is None:
            return None
        return self.mechanism_overlap > self.chance_overlap

    def describe(self) -> str:
        if self.prior_dominance is None:
            return (f"判定不能: 証拠{self.n_agents}名/主張{self.n_claims}件では"
                    f"脱相関を測れない(2以上が要る)")
        mech = ("未検査" if self.mechanism_overlap is None
                else f"{self.mechanism_overlap:.2f}")
        if self.unclassified_ratio:
            mech += f"(分類不能 {self.unclassified_ratio:.0%} を除外)"
        if self.chance_overlap is not None and self.mechanism_overlap is not None:
            mech += (f"(偶然水準 {self.chance_overlap:.2f} / "
                     f"比較できた対 {self.comparable_pairs}組)")
        if self.mechanism_overlap is not None and not self.mechanism_is_usable:
            mech += (f"[取りこぼしが多く判定には使わない"
                     f"(分類できたのは{self.classified_count}件)]")
        note = ("(語彙は割れたが機構が同じかは未検査 —— 脱相関できたとは言えない)"
                if self.verdict == VERDICT_SEMANTICALLY_UNCHECKED else "")
        return (f"証拠の重なり {self.evidence_overlap:.2f} / "
                f"語彙の似かより {self.output_similarity:.2f} / "
                f"機構の重なり {mech} / "
                f"事前分布の支配度 {self.prior_dominance:.2f} -> {self.verdict}{note}")

    def to_dict(self) -> Dict[str, object]:
        return {
            "evidence_overlap": self.evidence_overlap,
            "output_similarity": self.output_similarity,
            "mechanism_overlap": self.mechanism_overlap,
            "unclassified_ratio": self.unclassified_ratio,
            "classified_count": self.classified_count,
            "comparable_pairs": self.comparable_pairs,
            "chance_overlap": self.chance_overlap,
            "exceeds_chance": self.exceeds_chance,
            "mechanism_is_usable": self.mechanism_is_usable,
            "convergence": self.convergence,
            "prior_dominance": self.prior_dominance,
            "verdict": self.verdict,
            "n_agents": self.n_agents,
            "n_claims": self.n_claims,
        }


def analyze(assignment: Mapping[str, Sequence[str]],
            claims: Sequence[str],
            mechanism_labels: Optional[Sequence[str]] = None,
            n_mechanism_categories: Optional[int] = None
            ) -> DecorrelationReport:
    """配った証拠と出てきた主張から、脱相関が効いたかを実測する。

    証拠を分けたのに主張が一致するなら、それは合意ではなく事前分布の支配である。
    訓練データが議論を決めているという、見えない連鎖の測定になる。

    Args:
        mechanism_labels: 各主張が原因を帰属した機構の分類。省略すると語彙だけで
            判断することになり、収束を見落とす可能性が残るため、判定は
            VERDICT_SEMANTICALLY_UNCHECKED に留まる(できたことにしない)。
        n_mechanism_categories: 機構語彙の種類数。渡すと偶然水準を併記する。
            **重なりを 0 と比べるのは誤りである** —— 8種の語彙なら無作為でも 0.125 出る。

    Raises:
        DecorrelationError: ラベル数が主張数と合わないとき(取り違えた対応で
            測定すると、測っているつもりで別物を測ることになる)。
    """
    if mechanism_labels is not None and len(mechanism_labels) != len(claims):
        raise DecorrelationError(
            f"機構ラベル {len(mechanism_labels)} 件と主張 {len(claims)} 件が対応しない")
    return DecorrelationReport(
        evidence_overlap=evidence_overlap(assignment),
        output_similarity=output_similarity(claims),
        mechanism_overlap=(mechanism_overlap(mechanism_labels)
                           if mechanism_labels is not None else None),
        unclassified_ratio=(unclassified_ratio(mechanism_labels)
                            if mechanism_labels is not None else None),
        classified_count=(classified_count(mechanism_labels)
                          if mechanism_labels is not None else 0),
        comparable_pairs=(comparable_pairs(mechanism_labels)
                          if mechanism_labels is not None else 0),
        chance_overlap=chance_overlap(n_mechanism_categories),
        n_agents=len(assignment),
        n_claims=len(claims),
    )


# --- 制度空間の分割探索 ------------------------------------------------------

def assign_exploration_regions(agent_ids: Sequence[str],
                               dimensions: Sequence[str],
                               per_agent: int = 3,
                               salt: str = "") -> Dict[str, Tuple[str, ...]]:
    """各エージェントに『変更を検討する制度次元』を割り当てる。

    証拠の非対称に加え、探索領域も分けることで、構造的に異なる案が出る確率を上げる。
    これは結論を指定するものではない —— どの軸を *見るか* を分けるだけであり、
    その軸をどう変えるか(あるいは変えないか)はエージェントが決める。
    """
    if not agent_ids:
        raise DecorrelationError("配布先が空")
    if per_agent < 1:
        raise DecorrelationError("per_agent は 1 以上")
    if len(dimensions) < per_agent:
        raise DecorrelationError(
            f"次元 {len(dimensions)} 件では 1名あたり {per_agent} 件を配れない")
    n = len(dimensions)
    out: Dict[str, Tuple[str, ...]] = {}
    for i, aid in enumerate(agent_ids):
        h = int(hashlib.sha256(f"{aid}|{salt}".encode("utf-8")).hexdigest()[:8], 16)
        start = (h + i * per_agent) % n
        out[aid] = tuple(dict.fromkeys(
            dimensions[(start + k) % n] for k in range(per_agent)))
    return out
