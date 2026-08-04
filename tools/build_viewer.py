"""viewer/index.html を組み立てる — エンジンの定義を一箇所に保つ。

画面は自己完結した単一ファイルでなければならない(公開先が外部読込を許さない)。
しかしエンジンを index.html に手で貼れば、engine.js と二重定義になり、必ずずれる。

そこで index.html は **生成物** とする。app_template.html の <!--ENGINE--> に
engine.js の全文を注入して作る。テストは「index.html が engine.js の全文を
一字違わず含むこと」を照合し、片方だけ直した状態を通さない。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "viewer" / "app_template.html"
ENGINE = ROOT / "viewer" / "engine.js"
OUT = ROOT / "viewer" / "index.html"
MARK = "<!--ENGINE-->"

HEADER = (
    "<!-- 生成物である。直接編集しない。\n"
    "     直すのは viewer/app_template.html か viewer/engine.js であり、\n"
    "     python tools/build_viewer.py で再生成する。 -->\n"
)


def build() -> str:
    template = TEMPLATE.read_text(encoding="utf-8")
    engine = ENGINE.read_text(encoding="utf-8")
    if MARK not in template:
        raise SystemExit(f"テンプレートに {MARK} が無い")
    if "</script>" in engine.replace("<\\/script>", ""):
        raise SystemExit("engine.js が script 終了タグを含む(注入できない)")
    return HEADER + template.replace(MARK, engine)


def main() -> int:
    OUT.write_text(build(), encoding="utf-8")
    print(f"生成した: {OUT}({OUT.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
