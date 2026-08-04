"""LLM実行層 — 常駐モデルと、正直に失敗する構造化抽出。

実測(2026-08-02, RTX 3060 / Qwen3.5-9B nf4):
  ロード 634秒・モデル 7.65GB・単体 3.99 tok/s
  4体同時バッチ 18.82 tok/s(4.7倍)・VRAMピーク 7.91GB

ここから導かれる設計:

  1. モデルは常駐させる。ロードは一度きりで償却される(10分の初回コストは、24時間走る
     議論にとって問題にならない)。
  2. エージェントは重みではなく人格プロンプトで分岐し、バッチで同時に思考する。
     VRAM は 7.65 -> 7.91GB しか増えず、スループットは 4.7倍になる。

そして最も重要な規律:

  9B は厳密なスキーマを常には守れない。解析に失敗したとき何をするかが、この系が劇場に
  なるか否かを分ける。答えは「記録しない。そして失敗を数える」である。壊れた出力を推測で
  補って主張に仕立てるのは、モデルが言っていないことを議論の記録に混ぜることであり、
  本システムが最も禁じる捏造そのものになる。

さらに実測で判明した二つの罠に対処する:
  - Qwen3.5 は thinking mode が既定で有効。200トークンをすべて思考に費やし結論に至らない。
  - 日本語で問うても英語で答える。

基体(GPU)と純粋ロジックを Backend protocol で分離する。審査員が GPU 無しでも
議論の骨格を検証できることは、提出物として本質的である。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Protocol, Sequence, Tuple



class RuntimeError_(RuntimeError):
    """実行層の前提が壊れたとき(黙って続行しない)。"""


# --- 生成条件 ----------------------------------------------------------------

@dataclass(frozen=True)
class GenerationConfig:
    """生成条件。thinking は既定で無効(実測: 有効だと結論に到達しない)。"""
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.9
    thinking: bool = False
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise RuntimeError_("max_new_tokens は 1 以上")
        if not 0.0 < self.temperature <= 2.0:
            raise RuntimeError_("temperature は 0 より大きく 2 以下")
        if not 0.0 < self.top_p <= 1.0:
            raise RuntimeError_("top_p は 0 より大きく 1 以下")


class Backend(Protocol):
    """モデル基体の最小契約。実装は差し替え可能(GPU 実機 / テスト用の代替)。"""

    def generate(self, prompts: Sequence[str],
                 config: GenerationConfig) -> Tuple[str, ...]:
        """プロンプト列に対する生成列を返す(順序は入力に対応)。"""
        ...


# --- 人格と出力形式のプロンプト構築 ------------------------------------------

# 出力契約。自由記述を許すと 9B は独白を返す(実測)。形式を先に固定する。
_CLAIM_SCHEMA_HINT = (
    '{"claim": "主張", "grounds": ["根拠1", "根拠2"], '
    '"falsifier": "これが観測されたらこの主張は誤り", '
    '"confidence": 0.0から1.0, "unresolved": ["未解決点"]}'
)

LANGUAGE_DIRECTIVE = (
    "回答はすべて日本語で書くこと。英語で書いてはならない。"
    "思考過程を出力してはならない。結論だけを指定の形式で出すこと。"
)


def build_claim_prompt(persona_brief: str, task: str,
                       observations: Sequence[str] = ()) -> str:
    """構造化主張を求めるプロンプト。反証条件を必須として明示する。

    反証条件を書かせることが要点である。書けない主張は表明にすぎず、議論の材料にならない
    """
    obs = "\n".join(f"  - {o}" for o in observations) or "  (提示された観察はない)"
    return (
        f"{persona_brief}\n\n"
        f"{LANGUAGE_DIRECTIVE}\n\n"
        f"[観察されている社会状態]\n{obs}\n\n"
        f"[今回の課題]\n{task}\n\n"
        f"[出力形式] 次のJSONオブジェクトを1つだけ出力する。説明文を添えない。\n"
        f"{_CLAIM_SCHEMA_HINT}\n"
        f"falsifier は必須である。「何が観測されたら自分の主張が誤りだと分かるか」を書く。"
        f"書けない主張は提出してはならない。"
    )


# --- 抽出: 壊れた出力を推測で補わない ----------------------------------------

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)
# CJK(漢字・ひらがな・カタカナ)が一定割合あれば日本語とみなす簡易判定。
_CJK = re.compile(r"[぀-ヿ一-鿿]")


def strip_thinking(text: str) -> Tuple[str, str]:
    """思考ブロックを本文から分離する。捨てずに返す(黙って消さない)。

    Returns: (本文, 思考部分)
    """
    thoughts = "\n".join(m.group(0) for m in _THINK_BLOCK.finditer(text))
    return _THINK_BLOCK.sub("", text).strip(), thoughts


def looks_japanese(text: str, min_ratio: float = 0.10) -> bool:
    """日本語で書かれているか(実測: 日本語で問うても英語で返ることがある)。"""
    if not text.strip():
        return False
    return len(_CJK.findall(text)) / max(1, len(text)) >= min_ratio


REJECT_TRUNCATED = "truncated"      # JSONは始まっているが閉じていない(生成枠切れ)
REJECT_NO_JSON = "json_not_found"
REJECT_BAD_JSON = "json_invalid"
REJECT_SCHEMA = "schema_violation"
REJECT_LANGUAGE = "not_japanese"
# 基体が要求そのものに失敗した(通信・認証・予算)。**モデルの出来の話ではない。**
# 実測 2026-08-04: 予算不足で空になった応答が番兵に置き換わり、抽出層がそれを
# 「日本語で書かれていない」と診断していた。基体の失敗がモデルの失敗として
# 記録されると、直すべき場所を見失う。
REJECT_REQUEST_FAILED = "request_failed"

# 基体が失敗を表す番兵(openai_backend.FAILED_MARKER と同一)。
# ここに置くのは、抽出層が基体モジュールへ依存せずに判別できるようにするため。
FAILED_MARKER = "__REQUEST_FAILED__"


def is_request_failure(raw: str) -> bool:
    """生出力が基体の失敗を表しているか。内容の良し悪しの前に判別する。"""
    return raw.strip() == FAILED_MARKER


# --- ここから下は領域層へ移した ----------------------------------------------
#
# 初版は主張(StructuredClaim)の抽出をここに置いていたため、基盤が領域モジュールへ
# 依存していた。**基盤が領域を知っていると、題材を変えるたびに基盤が壊れる。**
# 実際、制度シミュレーションから基体探査へ移す際に import が折れた。
#
# 生の出力から何を取り出すかは題材ごとに違う。取り出し方(思考ブロックの分離・
# 言語判定・要求失敗の判別)だけが共通である。共通部分だけをここに残す。


def extract_json_object(raw: str, require_japanese: bool = True
                        ) -> Tuple[Optional[dict], Optional[str], str]:
    """生出力から JSON オブジェクトを一つ取り出す。**推測で補わない。**

    Returns:
        (取り出せた辞書 または None, 棄却理由 または None, 思考ブロック)

    棄却理由を分けて返すのは、直すべき場所が違うからである。要求そのものが
    失敗したのか、言語を守れなかったのか、枠が足りず打ち切られたのか、
    JSON が壊れているのか —— 混ぜれば原因を見失う。
    """
    if is_request_failure(raw):
        return None, REJECT_REQUEST_FAILED, ""

    body, thoughts = strip_thinking(raw)

    m = _JSON_OBJ.search(body)
    if not m:
        # JSON が無いときだけ、本文全体で言語を判定する。
        if require_japanese and not looks_japanese(body):
            return None, REJECT_LANGUAGE, thoughts
        if "{" in body:
            return None, REJECT_TRUNCATED, thoughts
        return None, REJECT_NO_JSON, thoughts
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, REJECT_BAD_JSON, thoughts
    if not isinstance(data, dict):
        return None, REJECT_BAD_JSON, thoughts

    # 言語判定は **抽出した文字列値** に対して行う。実測 2026-08-05 の欠陥:
    # JSON 封筒ごと判定すると、ASCII のキーと数値が日本語の割合を薄め、
    # 短い日本語だけを含む正当な出力が not_japanese で棄却された。
    # キーは形式であって発話ではない —— 発話の言語だけを検査する。
    if require_japanese:
        spoken = " ".join(_string_values(data))
        if spoken and not looks_japanese(spoken):
            return None, REJECT_LANGUAGE, thoughts
    return data, None, thoughts


def _string_values(obj) -> List[str]:
    """JSON の中の、人が書いた文字列値だけを集める(キーと数値は形式である)。"""
    out: List[str] = []
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(_string_values(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_string_values(v))
    elif isinstance(obj, str):
        out.append(obj)
    return out
