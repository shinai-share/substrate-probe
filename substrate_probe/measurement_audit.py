"""測定の妥当性審査 — その数値は行動を変えてよい根拠か(CONCEPT.md 3節)。

制度の効果を測るとき、三つの経路で「測ったつもり」が起きる。

    観測: 代理を測って対象を測っていない(「配布額100単位」で「生活の安定」を主張する)
    時間: 古い測定を現在の真実として使う(監視者自身が古さで嘘をつく)
    因果: 測定の入力に答えが混入している(制度規則の文言をそのまま効果指標にする)

三つ目が本システムにとって最も危険である。制度が直接設定した量をその制度の効果として
測るのは同語反復であり、何も検証していない。にもかかわらず数値は動くため、検証したように
見える。事前登録(preregistration)はこの審査を通らない指標を受け付けない。

**既定値は保守的である。** 何も与えなければ全軸で無効になる。妥当性は、対象を直接測り・
新鮮に測り・独立に測ったことで初めて獲得されるものであって、初期状態ではない。

出自について: 三軸の規律は本プロジェクトの外(masa様のAGI基盤 decorrelation_audit)で
定式化されたものである。本モジュールはそれを提出物が単体で動くよう再実装したものであり、
永続化層(否定記憶DB)は持たない。審査員の環境で追加物なしに動くことを優先した。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

# 漏れの経路(三軸)。
CH_OBSERVABLE = "observable"   # 代理を測り、対象で終端していない
CH_TIME = "time"               # 測定が古い、または時刻不明
CH_CAUSAL = "causal"           # 入力に答えが漏れている / 独立条件で確認していない

# 「今の真実」として行動に使ってよい測定の最大経過秒数。
DEFAULT_MAX_STALENESS_SECONDS = 3600.0


@dataclass(frozen=True)
class MeasurementClaim:
    """『この測定は何かを示している』という、審査される前の生の主張。"""
    claim: str                                   # 人間可読の主張
    observable: str                              # 実際に測った量(代理かもしれない)
    target: str                                  # その測定で主張したい対象
    proxy_delta: Optional[float] = None          # 観測量の変化
    target_delta: Optional[float] = None         # 対象を *直接* 測った変化(None=未測定)
    observed_at: Optional[str] = None            # ISO8601: 測定時刻(None=不明)
    evaluated_at: Optional[str] = None           # ISO8601: この測定で行動する時刻
    inputs: Tuple[str, ...] = ()                 # 測定を生んだ入力(制度規則の文言など)
    answer_tokens: Tuple[str, ...] = ()          # 独立に立証したい答え
    independent_condition: bool = False          # 摂動/盲検の下で再現したか


@dataclass(frozen=True)
class LeakReport:
    """三軸審査の結果。valid=True のときだけ、その測定は行動を変えてよい。"""
    valid: bool
    coupling_ok: bool
    fresh_ok: bool
    independence_ok: bool
    leak_channels: Tuple[str, ...]
    reason: str
    claim: str


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    """ISO8601 を datetime に。None/壊れた文字列は None(信用しない)。"""
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _check_coupling(c: MeasurementClaim) -> Tuple[bool, str]:
    """観測軸: 代理でなく対象そのもので終端しているか。"""
    if c.observable == c.target:
        if c.target_delta is None:
            return False, "対象を直接測っていない(値が無い)"
        return True, "対象そのものを直接測定"
    if c.target_delta is None:
        return False, f"代理'{c.observable}'で測り対象'{c.target}'を直接測っていない"
    if c.proxy_delta not in (None, 0) and c.target_delta == 0:
        return False, f"代理は動いた(Δ{c.proxy_delta})が対象は不変(基体と非結合)"
    return True, "代理と対象が整合"


def _check_freshness(c: MeasurementClaim, now: datetime,
                     max_staleness_seconds: float) -> Tuple[bool, str]:
    """時間軸: 『今の真実』として信用できるほど新鮮か。"""
    observed = _parse_iso(c.observed_at)
    if observed is None:
        return False, "測定時刻が不明/解釈不能"
    age = ((_parse_iso(c.evaluated_at) or now) - observed).total_seconds()
    if age > max_staleness_seconds:
        return False, (f"測定が古い(経過{age:.0f}s≈{age / 86400.0:.1f}d "
                       f"> 閾値{max_staleness_seconds:.0f}s)")
    return True, f"新鮮(経過{age:.0f}s)"


def _check_independence(c: MeasurementClaim) -> Tuple[bool, str]:
    """因果軸: 測定は主張対象から因果的に独立か(同語反復でないか)。"""
    leaked = sorted({t for t in c.answer_tokens
                     if t and any(t in inp for inp in c.inputs)})
    if leaked:
        return False, f"入力に答えが混入: {leaked}"
    if not c.independent_condition:
        return False, "摂動/盲検の下で再現していない(独立条件が未宣言)"
    return True, "独立条件で確認済み"


def audit(claim: MeasurementClaim, *,
          max_staleness_seconds: float = DEFAULT_MAX_STALENESS_SECONDS,
          now: Optional[datetime] = None) -> LeakReport:
    """測定主張を三軸で審査する。決して例外を投げない(想定外は棄権側へ倒す)。

    valid = 観測結合 AND 鮮度 AND 独立。一つでも漏れれば、その測定は行動を変えてよい
    根拠にならない。どの軸が漏れたかを leak_channels に列挙する —— 「無効」で終わらせず、
    直すべき場所を返す。
    """
    now = now or datetime.now()
    try:
        coupling_ok, c_reason = _check_coupling(claim)
        fresh_ok, f_reason = _check_freshness(claim, now, max_staleness_seconds)
        indep_ok, i_reason = _check_independence(claim)
    except Exception as e:  # 想定外でも「妥当」と言わない
        return LeakReport(
            valid=False, coupling_ok=False, fresh_ok=False, independence_ok=False,
            leak_channels=(CH_OBSERVABLE, CH_TIME, CH_CAUSAL),
            reason=f"審査中の内部エラー(安全側で無効): {e}", claim=claim.claim)

    channels: List[str] = []
    parts: List[str] = []
    for ok, ch, label, reason in (
            (coupling_ok, CH_OBSERVABLE, "観測漏れ", c_reason),
            (fresh_ok, CH_TIME, "時間漏れ", f_reason),
            (indep_ok, CH_CAUSAL, "因果漏れ", i_reason)):
        if not ok:
            channels.append(ch)
            parts.append(f"[{label}] {reason}")

    valid = not channels
    reason = (f"妥当: {c_reason}; {f_reason}; {i_reason}" if valid
              else " / ".join(parts))
    return LeakReport(valid=valid, coupling_ok=coupling_ok, fresh_ok=fresh_ok,
                      independence_ok=indep_ok, leak_channels=tuple(channels),
                      reason=reason, claim=claim.claim)
