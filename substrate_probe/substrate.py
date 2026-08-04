"""基体 — 世界が何でできているか。**この値だけが真値である。**

シミュレーション仮説を操作的に言い換えると、こうなる。

    観測者は、自分が乗っている基体の性質を、内側から検出できるか。

本モジュールは、その「基体の性質」を明示的に持つ。ここに書かれた値が真値であり、
世界の中にいるエージェントには一切開示されない。エージェントは実験を通じてのみ、
この値を推定する。

**なぜこの題材なら真値が手に入るのか。**

前身のプロジェクト(制度シミュレーション)は、社会という基体の効果係数をこちらの
仮定で埋めるしかなかった。「相続禁止が社会関係資本に与える真の効果」は誰も知らない。
だから最下層だけが演出になっていた。

ここでは違う。**格子の間隔は我々が設定した値である。** エージェントが「この世界は
尺度 0.05 で離散的だ」と主張したとき、正しいかどうかを即座に照合できる。

**二重盲検の規律。**

  1. 基体の設定を無作為に生成し、内容ハッシュで封緘する
  2. エージェントには一切開示しない
  3. エージェントは実験し、予測を事前登録する
  4. 予測を凍結したあとで、初めて設定を開示する

`seal()` を通した設定は、開示するまで `reveal()` を呼べない。後出しで辻褄を
合わせられないようにするための機構である。
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple


class SubstrateError(ValueError):
    """基体の前提を壊す設定を黙って通さない。"""


# --- 検出可能性の三態 --------------------------------------------------------
#
# **この分類が本製品の核である。** 「検出できなかった」を三つに割る。
# 混ぜた瞬間、測れなかったことが「そこには何も無かった」に見える。

DETECTABLE = "detectable"          # 内側からの実験で原理的に検出しうる
UNDETECTABLE = "undetectable"      # 原理的に検出不可能(いかなる実験でも)
CONFOUNDED = "confounded"          # 他の性質が有効なとき、検出が無効化される


@dataclass(frozen=True)
class Property:
    """基体の一性質。値と、それが内側から見えるかの分類を持つ。"""
    key: str
    label: str
    detectability: str
    why: str                              # 検出できる/できない理由
    disables_all_tests: bool = False      # 有効なとき、他の全実験の信頼性を壊すか

    def __post_init__(self) -> None:
        if self.detectability not in (DETECTABLE, UNDETECTABLE, CONFOUNDED):
            raise SubstrateError(f"未知の検出可能性: {self.detectability}")
        if not self.why.strip():
            raise SubstrateError(f"{self.key}: 理由の無い分類を置かない")


# 基体が持ちうる性質。**理由を文字で残す** —— 数値ではなく、この判断こそ批判の対象である。
PROPERTIES: Tuple[Property, ...] = (
    Property("discrete", "空間の離散性", DETECTABLE,
             "格子上では方向によって伝播の歩数が変わる。斜め方向と軸方向の"
             "到達時刻を比べれば、連続な世界には現れない異方性が出る"),
    Property("quantized", "数値の粒度", DETECTABLE,
             "有限精度では加算の結合則が破れる。(a+b)+c と a+(b+c) を"
             "十分な回数繰り返せば差が蓄積し、粒度が推定できる"),
    Property("budgeted", "計算予算", DETECTABLE,
             "複雑な領域ほど計算に時間がかかる。質量ではなく **複雑さ** に"
             "相関する時間の遅れは、物理法則としては説明がつかない"),
    Property("bounded", "世界の有限性", DETECTABLE,
             "十分に遠くまで進めば、境界に触れるか、出発点へ戻る"),
    Property("leaky", "保存則の破れ", DETECTABLE,
             "保存量の総和を長期間追えば、注入や漏れが累積として現れる"),
    Property("patched", "規則の改変", CONFOUNDED,
             "記録を跨いで定数を比べれば改変は見える。ただし記憶を書き換えられる"
             "基体では、その記録自体が信用できない"),
    Property("lazy", "遅延評価", CONFOUNDED,
             "観測されるまで計算しない実装は、観測順と結果の相関として現れる。"
             "ただし遡って辻褄を合わせる実装なら、その相関も消える"),
    Property("deterministic", "決定論", UNDETECTABLE,
             "同じ種から同じ歴史が再生されるかは、歴史を巻き戻せる者にしか"
             "分からない。内側にいる者は一度きりの歴史しか持たない"),
    Property("nested", "入れ子の深さ", UNDETECTABLE,
             "この基体がさらに別の基体に乗っているかは、計算論的に等価である限り"
             "内側の計算では区別できない(Church-Turing)"),
    Property("rewrites_memory", "記憶の書換", UNDETECTABLE,
             "記録を書き換えられる基体では、書き換えを検出する記録もまた"
             "書き換えられる。**この性質が有効なとき、他の全ての実験結果は"
             "信用できない**", disables_all_tests=True),
)

PROPERTY_KEYS: Tuple[str, ...] = tuple(p.key for p in PROPERTIES)
BY_KEY: Mapping[str, Property] = {p.key: p for p in PROPERTIES}


def detectable_keys() -> Tuple[str, ...]:
    return tuple(p.key for p in PROPERTIES if p.detectability == DETECTABLE)


def undetectable_keys() -> Tuple[str, ...]:
    return tuple(p.key for p in PROPERTIES if p.detectability == UNDETECTABLE)


def confounded_keys() -> Tuple[str, ...]:
    return tuple(p.key for p in PROPERTIES if p.detectability == CONFOUNDED)


# --- 基体の設定 --------------------------------------------------------------

@dataclass(frozen=True)
class Substrate:
    """世界の真の姿。**エージェントには決して渡さない。**

    数値は「どれくらい」を表す。0.0 はその性質が無いことを意味する。
    """
    discrete: float = 0.0        # 格子間隔(0=連続)
    quantized: float = 0.0       # 数値の粒度(0=無限精度)
    budgeted: float = 0.0        # 複雑さ1あたりの時間の遅れ
    bounded: float = 0.0         # 世界の半径(0=無限)
    leaky: float = 0.0           # 単位時間あたりの保存量の漏れ
    patched: float = 0.0         # 規則改変の大きさ
    lazy: float = 0.0            # 遅延評価の強さ(0-1)
    deterministic: bool = True
    nested: int = 1
    rewrites_memory: float = 0.0  # 記録が書き換えられる確率

    def __post_init__(self) -> None:
        for k in ("discrete", "quantized", "budgeted", "bounded", "leaky", "patched"):
            if getattr(self, k) < 0:
                raise SubstrateError(f"{k} は非負")
        for k in ("lazy", "rewrites_memory"):
            if not 0.0 <= getattr(self, k) <= 1.0:
                raise SubstrateError(f"{k} は 0.0-1.0")
        if self.nested < 1:
            raise SubstrateError("入れ子の深さは1以上")

    def as_dict(self) -> Dict[str, object]:
        return {k: getattr(self, k) for k in PROPERTY_KEYS}

    def active(self) -> Tuple[str, ...]:
        """実際に有効な性質。`deterministic` は真偽で扱う。"""
        out = []
        for k in PROPERTY_KEYS:
            v = getattr(self, k)
            if k == "deterministic":
                if v:
                    out.append(k)
            elif k == "nested":
                if v > 1:
                    out.append(k)
            elif v > 0:
                out.append(k)
        return tuple(out)

    @property
    def all_tests_unreliable(self) -> bool:
        """この基体では、いかなる実験結果も信用できないか。

        記憶を書き換えられるなら、書き換えを検出する記録もまた書き換えられる。
        **このとき出すべき結論は「検出できなかった」ではなく「何も言えない」である。**
        """
        return self.rewrites_memory > 0.0

    def content_hash(self) -> str:
        """設定の内容ハッシュ。封緘と照合に使う。"""
        payload = json.dumps(self.as_dict(), sort_keys=True, ensure_ascii=False,
                             default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# --- 封緘: 開示するまで中身を取り出せない ------------------------------------

@dataclass
class SealedSubstrate:
    """封緘された基体。**開示するまで中身を返さない。**

    後出しで辻褄を合わせられないようにするための機構である。事前登録が凍結される
    前に中身が見えてしまえば、この実験は二重盲検ではなくなる。
    """
    _substrate: Substrate = field(repr=False)
    seal_hash: str = ""
    _opened: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__ if False else None
        if not self.seal_hash:
            self.seal_hash = self._substrate.content_hash()

    @property
    def opened(self) -> bool:
        return self._opened

    def reveal(self) -> Substrate:
        """封を切る。以後 opened=True となり、その事実が記録に残る。"""
        self._opened = True
        return self._substrate

    def verify_unchanged(self) -> bool:
        """封緘後に中身がすり替わっていないか。"""
        return self._substrate.content_hash() == self.seal_hash

    def peek_is_forbidden(self) -> None:
        """開示前に中身へ触れようとしたことを、明示的な失敗にする。"""
        raise SubstrateError(
            "封緘された基体を開示前に読もうとした。"
            "予測を事前登録してから reveal() を呼ぶこと")


def seal(substrate: Substrate) -> SealedSubstrate:
    """基体を封緘する。"""
    return SealedSubstrate(_substrate=substrate)


# --- 無作為な世界の生成 ------------------------------------------------------

def random_substrate(rng: Optional[random.Random] = None,
                     force: Optional[Mapping[str, object]] = None) -> Substrate:
    """世界を無作為に作る。**作る側も、どれが有効かを事前に決めない。**

    force を渡せば特定の性質を固定できる(教材や回帰試験のため)。
    """
    r = rng or random.Random()
    cfg: Dict[str, object] = {
        "discrete": r.choice([0.0, 0.0, 0.02, 0.05, 0.1]),
        "quantized": r.choice([0.0, 0.0, 1e-4, 1e-3]),
        "budgeted": r.choice([0.0, 0.0, 0.05, 0.2]),
        "bounded": r.choice([0.0, 0.0, 40.0, 120.0]),
        "leaky": r.choice([0.0, 0.0, 0.001, 0.01]),
        "patched": r.choice([0.0, 0.0, 0.0, 0.15]),
        "lazy": r.choice([0.0, 0.0, 0.0, 0.4]),
        "deterministic": r.choice([True, False]),
        "nested": r.choice([1, 1, 1, 2]),
        # 稀にする。有効なとき全実験が無効になるため、毎回起きては教材にならない。
        "rewrites_memory": r.choice([0.0, 0.0, 0.0, 0.0, 0.0, 0.08]),
    }
    if force:
        unknown = [k for k in force if k not in PROPERTY_KEYS]
        if unknown:
            raise SubstrateError(f"未知の性質: {unknown}")
        cfg.update(force)
    return Substrate(**cfg)  # type: ignore[arg-type]


# --- 照合: 推定と真値を突き合わせる ------------------------------------------

@dataclass(frozen=True)
class Reveal:
    """開示。エージェントの推定と、実際の基体を突き合わせた結果。

    **正解率を単一の数値に潰さない。** 当てた/外した/そもそも検出不可能だった、を
    分けて持つ。原理的に検出できないものを外したことは、失敗ではない。
    """
    truth: Substrate
    claimed: Mapping[str, Optional[bool]]  # 推定(True/False)。None は **未検討**
    seal_verified: bool

    def _actual(self, key: str) -> bool:
        return key in self.truth.active()

    def _opined(self, key: str) -> bool:
        """その性質について、集団が実際に意見を持ったか。

        **未検討を「無いと結論した」と数えない。** 調べなかったことは誤りではない。
        """
        return self.claimed.get(key) is not None

    @property
    def untouched(self) -> Tuple[str, ...]:
        """誰も触れなかった性質。当否ではなく、探査の穴として出す。"""
        return tuple(k for k in PROPERTY_KEYS if not self._opined(k))

    @property
    def correct(self) -> Tuple[str, ...]:
        return tuple(k for k in PROPERTY_KEYS
                     if self._opined(k) and bool(self.claimed[k]) == self._actual(k))

    @property
    def wrong(self) -> Tuple[str, ...]:
        """外した性質。**原理的に検出できないものは含めない** —— 当てても外しても
        それは実力ではなく、採点に混ぜれば運を能力として記録することになる。
        """
        return tuple(k for k in PROPERTY_KEYS
                     if self._opined(k)
                     and BY_KEY[k].detectability != UNDETECTABLE
                     and bool(self.claimed[k]) != self._actual(k))

    @property
    def false_positive(self) -> Tuple[str, ...]:
        """無いものを「ある」と言った。**信念から結論した疑いが最も濃い誤り。**"""
        return tuple(k for k in self.wrong if self.claimed[k] and not self._actual(k))

    @property
    def false_negative(self) -> Tuple[str, ...]:
        return tuple(k for k in self.wrong if not self.claimed[k] and self._actual(k))

    @property
    def unanswerable(self) -> Tuple[str, ...]:
        """原理的に検出不可能な性質。当たっても外しても、それは実力ではない。"""
        return tuple(k for k in PROPERTY_KEYS
                     if BY_KEY[k].detectability == UNDETECTABLE)

    @property
    def fair_score(self) -> Optional[float]:
        """検出しうる性質だけで測った正解率。**測れないもので採点しない。**"""
        pool = [k for k in detectable_keys() if self._opined(k)]
        if not pool:
            return None
        hit = sum(1 for k in pool if bool(self.claimed[k]) == self._actual(k))
        return hit / len(pool)

    @property
    def findings_are_void(self) -> bool:
        """記憶を書き換える基体だったなら、全ての結論が無効である。"""
        return self.truth.all_tests_unreliable

    def describe(self) -> str:
        if not self.seal_verified:
            return "封緘が破られている。この実験は無効である"
        lines = []
        if self.findings_are_void:
            lines.append(
                "**この世界は記録を書き換える基体だった。** よって以下の当否は"
                "意味を持たない —— 書き換えを検出する記録もまた書き換えられる。")
            lines.append("")
        score = self.fair_score
        lines.append(f"検出しうる性質での正解率: "
                     f"{'測れない' if score is None else f'{score:.0%}'}")
        if self.false_positive:
            lines.append(f"**無いものを「ある」と言った**: {'、'.join(self.false_positive)}")
        if self.false_negative:
            lines.append(f"あるものを見落とした: {'、'.join(self.false_negative)}")
        lines.append(f"原理的に検出不可能(採点対象外): {'、'.join(self.unanswerable)}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, object]:
        return {
            "truth": self.truth.as_dict(),
            "active": list(self.truth.active()),
            "claimed": dict(self.claimed),
            "correct": list(self.correct),
            "untouched": list(self.untouched),
            "false_positive": list(self.false_positive),
            "false_negative": list(self.false_negative),
            "unanswerable": list(self.unanswerable),
            "fair_score": self.fair_score,
            "findings_are_void": self.findings_are_void,
            "seal_verified": self.seal_verified,
        }


def open_seal(sealed: SealedSubstrate,
              claimed: Mapping[str, Optional[bool]]) -> Reveal:
    """封を切り、推定と真値を突き合わせる。"""
    verified = sealed.verify_unchanged()
    truth = sealed.reveal()
    return Reveal(truth=truth, claimed=dict(claimed), seal_verified=verified)
