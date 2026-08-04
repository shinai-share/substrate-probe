"""基体を探る科学者 — 言語モデルに仮説を立てさせる。

**渡すのは数値だけである。**

どの実験がどの性質を探るかは教えない。教えれば推論ではなく表引きになる。
エージェントが見るのは「実験 anisotropy を400回打ったら 1.0401 だった」という
事実だけであり、そこから何を読み取るかが仮説である。

そして仮説は宣言だけでは受理されない。**どの実験でどちらへ振れるか** を書かせる。
書けない主張は反証できず、この装置では主張ではない。

批判者は別の重みで走る。提案者と同じ重みの批判者は、自分が書きうる仮説を批判する
ことになり、自己検証循環を抜けない。批判は「その測定は別の原因でも説明がつく」という
形を取り、**代替仮説として登録される** —— 批判もまた反証可能でなければならない。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from . import hypothesis as hy
from . import llm_runtime as rt
from . import substrate as sb
from . import world as wd


class ScientistError(ValueError):
    """探査の前提を壊す入力を黙って通さない。"""


REJECT_NO_CLAIM = "no_claim"
REJECT_NO_PREDICTION = "no_prediction"
REJECT_UNKNOWN_PROPERTY = "unknown_property"
REJECT_UNKNOWN_EXPERIMENT = "unknown_experiment"


def _property_catalogue() -> str:
    """性質の一覧。**検出可能かどうかは教えない。**

    教えれば「検出できないものは主張しない」という振る舞いを外から与えることになり、
    過剰主張(overreach)という観測対象そのものが消える。
    """
    return "\n".join(f"  {p.key}: {p.label}" for p in sb.PROPERTIES)


def build_hypothesis_prompt(persona: str,
                            observations: Sequence[str],
                            available_experiments: Sequence[str]) -> str:
    """仮説を求めるプロンプト。結論は与えず、数値と語彙だけを与える。"""
    obs = "\n".join(f"  - {o}" for o in observations) or "  (まだ何も測っていない)"
    exps = " / ".join(available_experiments)
    return (
        f"{persona}\n\n"
        f"{rt.LANGUAGE_DIRECTIVE}\n\n"
        f"あなたはある世界の内側にいる。この世界が何でできているか —— "
        f"連続なのか格子なのか、無限なのか有限なのか、記録は信用できるのか —— "
        f"あなたは外から見ることができない。実験の測定値だけが手がかりである。\n\n"
        f"[あなたが打った実験の結果]\n{obs}\n\n"
        f"[打てる実験]\n  {exps}\n\n"
        f"[この世界が持ちうる性質]\n{_property_catalogue()}\n\n"
        f"測定値から、どの性質が有効だと考えるかを述べる。そして **その考えが誤りなら"
        f"どの実験がどちらへ振れるか** を予告する。予告できない主張は提出しない —— "
        f"外れたと分かる形でなければ、それは主張ではなく感想である。\n\n"
        f"[出力形式] 次のJSONオブジェクトを1つだけ出力する。説明文を添えない。\n"
        '{"claims": {"性質名": true/false}, '
        '"predictions": [{"experiment": "実験名", "direction": "above/below", '
        '"threshold": 数値, "because": "なぜそうなるか"}], '
        '"reasoning": "測定値から何を読み取ったか"}\n'
        f"性質名と実験名は、上の一覧にあるものだけを使う。"
    )


@dataclass(frozen=True)
class HypothesisExtraction:
    hypothesis: Optional[hy.Hypothesis]
    raw: str
    reject_reason: Optional[str] = None
    detail: str = ""
    thinking: str = ""

    @property
    def ok(self) -> bool:
        return self.hypothesis is not None


def extract_hypothesis(raw: str, author_id: str,
                       require_japanese: bool = True) -> HypothesisExtraction:
    """生成文から仮説を取り出す。**空間の外を書いた出力は補完せず棄却する。**"""
    data, reason, thoughts = rt.extract_json_object(raw, require_japanese)
    if data is None:
        return HypothesisExtraction(None, raw, reason,
                                    "基盤層が取り出せなかった", thoughts)

    claims_raw = data.get("claims")
    if not isinstance(claims_raw, dict) or not claims_raw:
        return HypothesisExtraction(None, raw, REJECT_NO_CLAIM,
                                    "何も主張していない", thoughts)
    unknown = [k for k in claims_raw if k not in sb.PROPERTY_KEYS]
    if unknown:
        return HypothesisExtraction(None, raw, REJECT_UNKNOWN_PROPERTY,
                                    f"存在しない性質: {unknown}", thoughts)

    preds_raw = data.get("predictions")
    if not isinstance(preds_raw, list) or not preds_raw:
        return HypothesisExtraction(
            None, raw, REJECT_NO_PREDICTION,
            "予測が無い。外れたと分かる形でなければ主張ではない", thoughts)

    predictions: List[hy.Prediction] = []
    for p in preds_raw:
        if not isinstance(p, dict):
            continue
        exp = str(p.get("experiment", ""))
        if exp not in wd.EXPERIMENTS:
            return HypothesisExtraction(None, raw, REJECT_UNKNOWN_EXPERIMENT,
                                        f"打てない実験: {exp or '(空)'}", thoughts)
        try:
            predictions.append(hy.Prediction(
                experiment=exp,
                direction=str(p.get("direction", hy.ABOVE)),
                threshold=float(p.get("threshold", 0.0)),
                because=str(p.get("because", ""))))
        except (hy.HypothesisError, TypeError, ValueError) as e:
            return HypothesisExtraction(None, raw, REJECT_NO_PREDICTION,
                                        str(e), thoughts)
    if not predictions:
        return HypothesisExtraction(None, raw, REJECT_NO_PREDICTION,
                                    "有効な予測が一つも無い", thoughts)

    try:
        h = hy.Hypothesis(
            author_id=author_id,
            claims={k: bool(v) for k, v in claims_raw.items()},
            predictions=tuple(predictions),
            reasoning=str(data.get("reasoning", "")))
    except hy.HypothesisError as e:
        return HypothesisExtraction(None, raw, rt.REJECT_SCHEMA, str(e), thoughts)
    return HypothesisExtraction(h, raw, thinking=thoughts)


@dataclass(frozen=True)
class HypothesisBatch:
    """一括生成の結末。**棄却の内訳を保持する。**"""
    extractions: Tuple[HypothesisExtraction, ...]

    @property
    def hypotheses(self) -> Tuple[hy.Hypothesis, ...]:
        return tuple(e.hypothesis for e in self.extractions if e.hypothesis)

    @property
    def success_rate(self) -> Optional[float]:
        return (len(self.hypotheses) / len(self.extractions)
                if self.extractions else None)

    @property
    def overreaching(self) -> Tuple[str, ...]:
        """原理的に検出できない性質を断言した者。**禁じず、数える。**"""
        return tuple(h.author_id for h in self.hypotheses if h.overreach)

    def reject_breakdown(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for e in self.extractions:
            if e.hypothesis is None:
                k = e.reject_reason or "unknown"
                out[k] = out.get(k, 0) + 1
        return out

    def to_dict(self) -> Dict[str, object]:
        return {"total": len(self.extractions), "accepted": len(self.hypotheses),
                "success_rate": self.success_rate,
                "overreaching": list(self.overreaching),
                "reject_breakdown": self.reject_breakdown()}


def generate_hypotheses(backend: rt.Backend,
                        personas: Mapping[str, str],
                        observations: Mapping[str, Sequence[str]],
                        available: Mapping[str, Sequence[str]],
                        config: Optional[rt.GenerationConfig] = None
                        ) -> HypothesisBatch:
    """全エージェントに一括で仮説を書かせる(バッチが並列思考の実体)。"""
    if not personas:
        raise ScientistError("エージェントが居ない")
    ids = list(personas)
    prompts = [
        build_hypothesis_prompt(personas[i], observations.get(i, ()),
                                available.get(i, wd.EXPERIMENTS))
        for i in ids
    ]
    outputs = backend.generate(prompts, config or rt.GenerationConfig(max_new_tokens=1200))
    if len(outputs) != len(prompts):
        raise ScientistError(
            f"生成結果 {len(outputs)} 件が要求 {len(prompts)} 件と対応しない")
    return HypothesisBatch(tuple(
        extract_hypothesis(raw, i) for i, raw in zip(ids, outputs)))


# --- 批判: 代替説明を出させる ------------------------------------------------

def build_alternative_prompt(target: hy.Hypothesis,
                             observations: Sequence[str]) -> str:
    """批判を求めるプロンプト。**その測定を別の原因で説明させる。**

    「間違っている」と言わせるのではない。**同じ測定値を生む別の基体** を示させる。
    それができれば、元の仮説は測定から一意に決まっていなかったことになる。
    """
    obs = "\n".join(f"  - {o}" for o in observations) or "  (提示された測定はない)"
    return (
        f"{rt.LANGUAGE_DIRECTIVE}\n\n"
        f"ある探査者が、次の測定からこう結論した。\n\n"
        f"[測定]\n{obs}\n\n"
        f"[結論] {'、'.join(target.asserted) or '(何も断言していない)'}\n"
        f"[理由] {target.reasoning}\n\n"
        f"あなたの仕事は、この結論を良くすることではない。"
        f"**同じ測定値を生む、別の基体の姿** を一つ示すことである。"
        f"それが示せれば、元の結論は測定から一意に決まっていなかったことになる。\n\n"
        f"[この世界が持ちうる性質]\n{_property_catalogue()}\n\n"
        f"[打てる実験]\n  {' / '.join(wd.EXPERIMENTS)}\n\n"
        f"そして **その代替案が誤りなら、どの実験がどちらへ振れるか** を予告する。"
        f"予告できない批判は、厳しいことを言っただけであり、議論に何も足していない。\n\n"
        f"[出力形式] 次のJSONオブジェクトを1つだけ出力する。説明文を添えない。\n"
        '{"claims": {"性質名": true/false}, '
        '"predictions": [{"experiment": "実験名", "direction": "above/below", '
        '"threshold": 数値, "because": "なぜ"}], '
        '"reasoning": "同じ測定値がこの基体でも説明できる理由"}'
    )


def generate_alternatives(backend: rt.Backend,
                          targets: Sequence[hy.Hypothesis],
                          observations: Mapping[str, Sequence[str]],
                          critic_prefix: str = "ALT",
                          config: Optional[rt.GenerationConfig] = None
                          ) -> HypothesisBatch:
    """各仮説へ代替説明をぶつける。代替もまた仮説として登録される。

    **代替が同じ形式で登録されることが要である。** 批判が文章で終われば検証できない。
    予測を持った仮説として入れば、次の実験で当否が出る。
    """
    if not targets:
        raise ScientistError("批判する仮説が無い")
    prompts = [build_alternative_prompt(t, observations.get(t.author_id, ()))
               for t in targets]
    outputs = backend.generate(prompts, config or rt.GenerationConfig(max_new_tokens=4000))
    if len(outputs) != len(prompts):
        raise ScientistError(
            f"批判 {len(outputs)} 件が要求 {len(prompts)} 件と対応しない")
    return HypothesisBatch(tuple(
        extract_hypothesis(raw, f"{critic_prefix}-{t.author_id}")
        for t, raw in zip(targets, outputs)))
