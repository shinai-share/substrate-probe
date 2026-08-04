"""別重みの基体 — 批判者と分類器を提案者から切り離す。

なぜ「強い基体」ではなく「別の基体」なのか。

    出力 = f(重み, 入力)。証拠と探索領域(入力)は既に分けた。
    残る相関の源は **重みそのもの** である。

全エージェントを同じAPIモデルに置き換えても相関は減らない。むしろ強いモデルほど
「正解らしい答え」へ収束しやすく、疑似多様性は悪化しうる。効くのは強さではなく違いである。

ゆえに本モジュールは、提案者(ローカル Qwen3.5-9B 常駐)はそのままに、**批判者と分類器
だけ** を別の重みに移すために書かれている。批判者が提案者と違う重みであることは、
自己検証循環に対する構造的な対抗手段になる。強い判定者を1つ置くこととは別のことである。

規律:

  - 鍵が無ければ **例外を投げる**。黙ってローカルへ縮退しない。縮退した瞬間、
    「別重みで検証した」という報告が嘘になる。
  - `identity()` が実際に応答したモデル名を返す。報告は宣言ではなく応答に接地する。
  - 依存を増やさない(標準ライブラリの urllib のみ)。審査員が pip 追加なしで動かせる。
  - 失敗した要求は空文字で埋めず、その要求だけを失敗として残す
    (抽出層が棄却として数える)。

費用は呼び出し回数に比例する。提案者を移していないのは、周回あたりの呼び出しが
最も多いのが提案生成だからでもある。
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import env_config
from .llm_runtime import (FAILED_MARKER as _FAILED_MARKER,
                          GenerationConfig, RuntimeError_)

DEFAULT_MODEL = "gpt-5"

# 推論モデルの下限。**max_completion_tokens は推論と出力の合計である。**
#
# 実測 2026-08-04: 分類に 32 と 300 を与えたところ、どちらも本文が空で返った。
# 予算が内部推論に使い切られ、答えが1文字も出ない。2000 を与えると正しく答えた。
# このとき空応答は「分類不能」として記録されており、**飢えさせたモデルを
# 「分類できなかった」と報告する** ところだった。短い答えほど予算が要る、という
# 直感に反する性質なので、下限を機構で持つ。
MIN_COMPLETION_TOKENS = 2000
DEFAULT_ENDPOINT = "https://api.openai.com/v1/chat/completions"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"

# 失敗した1件を表す番兵。空文字を返すと抽出層が「JSONが無い」と誤診し、
# 通信の失敗がモデルの失敗として記録される(原因の取り違え)。
FAILED_MARKER = _FAILED_MARKER      # 単一の真実源は llm_runtime 側に置く

# 失敗の種別。**番兵だけでは原因が分からない。**
# 実測 2026-08-04: クレジット残高ゼロで全要求が失敗したが、返るのは番兵だけで理由が
# 消えており、原因の特定が手作業になった。バッチを守るために例外を飲むのは正しいが、
# 飲んだ理由まで捨てるのは正しくない。
ERROR_AUTH = "auth"                # 鍵が無効/権限が無い
ERROR_QUOTA = "quota"              # 残高・割当の枯渇
ERROR_RATE_LIMIT = "rate_limit"    # 一時的な絞り
ERROR_NETWORK = "network"          # 到達できない
ERROR_MALFORMED = "malformed"      # 応答の形が想定と違う

# 再試行しても結果が変わらない種別。ここに該当したら残りの要求を投げない
# (残高ゼロで4回叩くのは、費用にならない代わりに時間と診断を濁らせるだけである)。
PERMANENT_ERRORS: Tuple[str, ...] = (ERROR_AUTH, ERROR_QUOTA)


@dataclass(frozen=True)
class Failure:
    """1件の失敗。種別と、人間が読める理由を持つ(鍵は含めない)。"""
    kind: str
    detail: str

    @property
    def permanent(self) -> bool:
        return self.kind in PERMANENT_ERRORS


def classify_http_error(status: int, body: str) -> Failure:
    """HTTP の失敗を種別へ落とす。本文はそのまま載せない(長すぎるため要約する)。"""
    try:
        payload = json.loads(body).get("error", {})
        message = str(payload.get("message", ""))[:200]
        code = str(payload.get("code") or payload.get("type") or "")
    except (json.JSONDecodeError, AttributeError, TypeError):
        message, code = body[:200], ""

    if status in (401, 403):
        return Failure(ERROR_AUTH, f"HTTP {status}: {message or '認証に失敗'}")
    if status == 402 or "quota" in code or "credit" in code:
        return Failure(ERROR_QUOTA, f"HTTP {status}: {message or '残高・割当が尽きている'}")
    if status == 429:
        return Failure(ERROR_RATE_LIMIT, f"HTTP {status}: {message or '要求が絞られている'}")
    return Failure(ERROR_MALFORMED, f"HTTP {status}: {message or '想定外の応答'}")


def _reasoning_tokens(body: Dict[str, object]) -> Optional[int]:
    """応答が推論に使ったトークン数。取れなければ None(捏造しない)。"""
    try:
        usage = body.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        return int(details.get("reasoning_tokens"))
    except (AttributeError, TypeError, ValueError):
        return None


@dataclass
class OpenAIBackend:
    """GPT を Backend protocol として使う。批判者・分類器の役に割り当てる。

    llm_runtime.Backend を満たすので、AgentRuntime / label_mechanisms へそのまま渡せる。
    ローカル基体との差し替えが1行で済むことが、対照実験の前提である。
    """
    model: str = DEFAULT_MODEL
    endpoint: str = DEFAULT_ENDPOINT
    api_key_env: str = DEFAULT_API_KEY_ENV
    api_key: Optional[str] = None
    timeout: float = 120.0
    max_parallel: int = 4
    # 実際に応答したモデル名。宣言ではなく応答から埋める(捏造しない)。
    _responded_model: Optional[str] = field(default=None, repr=False)
    _calls: int = field(default=0, repr=False)
    _failures: List[Failure] = field(default_factory=list, repr=False)

    def _lookup_key(self) -> str:
        """明示指定 -> 環境変数 -> .env の順で探す。値はここから外へ出さない。

        .env を最後に見るのは、シェルや CI で明示的に渡された値を尊重するためである
        (env_config.load_env は既存の環境変数を上書きしない)。
        """
        if self.api_key:
            return self.api_key
        if os.environ.get(self.api_key_env):
            return os.environ[self.api_key_env]
        env_config.load_env()
        return os.environ.get(self.api_key_env, "")

    def resolve_key(self) -> str:
        """鍵を解決する。無ければ例外 —— 黙ってローカルへ縮退しない。"""
        key = self._lookup_key()
        if not key:
            raise RuntimeError_(
                f"APIキーが無い。{env_config.describe_key_presence(self.api_key_env)}。"
                f"鍵が無いまま別重みの検証を行ったことにはできない")
        return key

    def available(self) -> bool:
        """鍵があるか。実行前の判定に使う(例外を制御フローにしない)。"""
        return bool(self._lookup_key())

    def identity(self) -> str:
        """報告に載せる基体の名。応答が返るまで『未確認』と言う。"""
        return self._responded_model or f"{self.model}(未確認)"

    @property
    def call_count(self) -> int:
        """実際に発行した要求数。費用と再現性の記録に使う。"""
        return self._calls

    @property
    def failures(self) -> Tuple[Failure, ...]:
        """これまでの失敗。種別つきで返す(番兵だけでは原因が分からない)。"""
        return tuple(self._failures)

    @property
    def blocked_by(self) -> Optional[Failure]:
        """再試行しても変わらない失敗を掴んでいるか。掴んでいれば以後を送らない。"""
        return next((f for f in self._failures if f.permanent), None)

    def describe_failures(self) -> str:
        """失敗の要約。**原因を一目で言う** ことが目的である。"""
        if not self._failures:
            return "失敗なし"
        kinds: Dict[str, int] = {}
        for f in self._failures:
            kinds[f.kind] = kinds.get(f.kind, 0) + 1
        head = "、".join(f"{k}={v}件" for k, v in kinds.items())
        return f"{head} / 最初の理由: {self._failures[0].detail}"

    def preflight(self) -> Optional[Failure]:
        """最小の1件を投げ、遠い側が使えるかを先に確かめる。

        実測 2026-08-04 の教訓: 残高ゼロに気づく前にローカル 9B を 160秒かけて
        ロードしていた。**高い準備の前に、安い検証を置く。**

        Returns:
            使えるなら None、使えないなら理由。
        """
        before = len(self._failures)
        self.generate(["ok"], GenerationConfig(max_new_tokens=16))
        return self._failures[before] if len(self._failures) > before else None

    min_completion_tokens: int = MIN_COMPLETION_TOKENS

    def _payload(self, prompt: str, config: GenerationConfig) -> Dict[str, object]:
        # 呼び出し側が短い予算を指定しても下限を割らない。ローカル基体の感覚
        # (32トークンで機構名を返す)をそのまま持ち込むと、推論分で飢える。
        budget = max(config.max_new_tokens, self.min_completion_tokens)
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": budget,
        }

    def _fail(self, failure: Failure) -> str:
        """失敗を記録して番兵を返す。**理由を捨てない。**"""
        self._failures.append(failure)
        return FAILED_MARKER

    def _post(self, prompt: str, config: GenerationConfig) -> str:
        """1件の要求。失敗は番兵で返す(全体を巻き添えにしない)が、理由は残す。

        既に恒久的な失敗(認証・残高)を掴んでいるなら、この1件は投げない。
        残高ゼロで残りを叩いても結果は変わらず、診断を濁らせるだけである。
        """
        if self.blocked_by is not None:
            return self._fail(Failure(self.blocked_by.kind,
                                      f"先行要求が恒久的に失敗したため送信しない"
                                      f"({self.blocked_by.detail})"))
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(self._payload(prompt, config)).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.resolve_key()}"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as res:
                body = json.loads(res.read().decode("utf-8"))
        except urllib.error.HTTPError as e:      # URLError より先に捕まえる
            try:
                detail = e.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            return self._fail(classify_http_error(e.code, detail))
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            return self._fail(Failure(ERROR_NETWORK, f"到達できない: {e}"))
        except json.JSONDecodeError as e:
            return self._fail(Failure(ERROR_MALFORMED, f"応答がJSONでない: {e}"))

        self._responded_model = str(body.get("model") or self.model)
        try:
            content = str(body["choices"][0]["message"]["content"] or "")
        except (KeyError, IndexError, TypeError):
            return self._fail(Failure(ERROR_MALFORMED, "応答に本文が無い"))

        if not content.strip():
            # **空応答を「モデルの答え」として通さない。** 実測 2026-08-04 では、
            # 予算不足で空になった応答が「分類不能」として記録されていた。
            # 測れなかったことと、測って分からなかったことは違う。
            reasoning = _reasoning_tokens(body)
            hint = (f"予算が推論に使い切られた疑い(reasoning_tokens={reasoning})"
                    if reasoning else "理由不明")
            return self._fail(Failure(
                ERROR_MALFORMED,
                f"本文が空。max_completion_tokens を増やす。{hint}"))
        return content

    def generate(self, prompts: Sequence[str],
                 config: GenerationConfig) -> Tuple[str, ...]:
        """一括生成。APIに真のバッチは無いため並列化するが、順序は必ず保つ。

        順序を保つことが要件である。要求と応答の対応が崩れると、ある案への批判が
        別の案の批判として記録される —— 誰が何を言ったかが壊れる。
        """
        if not prompts:
            raise RuntimeError_("プロンプトが空")
        self.resolve_key()   # 鍵が無ければここで落とす(部分実行を避ける)
        self._calls += len(prompts)
        workers = max(1, min(self.max_parallel, len(prompts)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return tuple(pool.map(lambda p: self._post(p, config), prompts))


def describe_substrates(proposer: object, critic: object) -> str:
    """どの重みが何を担ったかを1行で残す。

    報告に基体名を書かないと、「別重みで検証した」という主張が検証できなくなる。
    自己申告ではなく、各基体が名乗った名前をそのまま並べる。
    """
    def name(b: object) -> str:
        fn = getattr(b, "identity", None)
        if callable(fn):
            return str(fn())
        return type(b).__name__
    return f"提案者={name(proposer)} / 批判者・分類器={name(critic)}"
