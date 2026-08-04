""".env からの鍵の読み込み — 依存を増やさず、環境変数を上書きしない。

外部パッケージ(python-dotenv)を使わないのは、提出物が pip 追加なしで動くべきだからである。
実装は数十行で足り、増やした依存は審査員の環境で必ず一度は壊れる。

規律:

  **既に設定されている環境変数を上書きしない。** シェルや CI で明示的に渡された値が、
  ファイルに書かれた古い値に静かに負けるのは、最も気づきにくい事故である。
  .env は「無ければ補う」ものであって「決める」ものではない。

  **値を決して出力しない。** 本モジュールが返すのは「読めたか」「どの鍵名が入っていたか」
  までであり、値そのものはログにも例外文にも載せない。

  **見つからないことを異常として扱わない。** .env はあってもなくてもよい。
  鍵が無いときに落ちるのは、鍵を必要とする側(openai_backend)の責務である。

なお .env は公開リポジトリへ絶対に入れない。`.gitignore` に登録し、雛形として
`.env.example` だけを配る(そちらには値を書かない)。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

# 探す場所。手前(パッケージ直下)を先に見て、無ければ repo 直下を見る。
_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_DIR = _PACKAGE_DIR.parent
SEARCH_PATHS: Tuple[Path, ...] = (_PACKAGE_DIR / ".env", _REPO_DIR / ".env")


def parse_env_text(text: str) -> Dict[str, str]:
    """.env の中身を辞書にする。壊れた行は黙って飛ばす(全体を落とさない)。

    対応する書き方:
        KEY=value / export KEY=value / KEY="value" / KEY='value'
        # から始まる行と空行は無視する。
    """
    out: Dict[str, str] = {}
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, _, value = s.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        if not key:
            continue
        v = value.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        out[key] = v
    return out


def read_env_file(path: Path) -> Dict[str, str]:
    """1ファイルを読む。読めなければ空(存在しないことは異常ではない)。"""
    try:
        return parse_env_text(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return {}


def load_env(paths: Optional[Iterable[Path]] = None,
             environ: Optional[Dict[str, str]] = None) -> Tuple[str, ...]:
    """.env を読み、**未設定の変数だけ** を環境へ補う。

    既に設定されている変数は上書きしない。シェルや CI で明示的に渡された値が、
    ファイルの古い値に静かに負けるのを防ぐ。

    Returns:
        実際に補った変数名(値は返さない)。既に設定済みだったものは含まれない。
    """
    env = os.environ if environ is None else environ
    filled: list[str] = []
    for path in (paths if paths is not None else SEARCH_PATHS):
        for key, value in read_env_file(Path(path)).items():
            if key in env and env[key]:
                continue          # 明示的に渡された値を尊重する
            env[key] = value
            filled.append(key)
    return tuple(dict.fromkeys(filled))


# 雛形。ここに値を書いてはならない —— .gitignore が `!.env.example` で **追跡対象** に
# しているため、値を書くとそのまま公開リポジトリへ入る。
EXAMPLE_PATH = _PACKAGE_DIR / ".env.example"


def example_holds_a_value(name: str) -> bool:
    """雛形に値が書き込まれていないか。

    実測 2026-08-04: 「.env.example を .env として複製し、値を埋める」という案内に対し、
    雛形そのものが編集された。雛形は追跡対象であり、そのまま commit すれば鍵が公開される。
    **案内文だけで守れる約束は、いずれ破れる。** 機構で検出する。
    """
    return bool(read_env_file(EXAMPLE_PATH).get(name))


def describe_key_presence(name: str,
                          environ: Optional[Dict[str, str]] = None) -> str:
    """鍵の有無を人間に伝える。**値は決して出さない。**

    .env まで含めて調べる。実測 2026-08-04 の欠陥: 本関数は os.environ しか見ず、
    基体が .env から鍵を解決できている状態でも「未設定」と報告していた。
    同じ状態を二つの口が食い違って報告するのは、本システムが最も嫌う嘘である。
    """
    env = os.environ if environ is None else environ
    warning = ""
    if example_holds_a_value(name):
        warning = (f" 【警告】{EXAMPLE_PATH} に値が書かれている。"
                   f"このファイルは追跡対象であり、commit すれば公開される。"
                   f"値は .env へ移し、雛形は空に戻すこと")

    if env.get(name):
        return f"{name}: 設定あり(環境変数)" + warning

    if environ is None:                 # 実環境なら .env も調べる(嘘をつかない)
        load_env()
        if os.environ.get(name):
            source = next((str(p) for p in SEARCH_PATHS
                           if Path(p).exists() and read_env_file(Path(p)).get(name)),
                          ".env")
            return f"{name}: 設定あり({source})" + warning

    found = [str(p) for p in SEARCH_PATHS if Path(p).exists()]
    where = "、".join(found) if found else "見つからない"
    return (f"{name}: 未設定(参照した .env: {where})。"
            f"{EXAMPLE_PATH} を **.env として複製してから** 値を埋める"
            f"(雛形自体に書かない)" + warning)
