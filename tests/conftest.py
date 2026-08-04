"""テストを周囲の環境から切り離す。

実測 2026-08-04 の欠陥: 開発機に実在する `.env` をテストが拾い、「鍵が無いときに
縮退せず落ちる」ことを検証しているはずのテストが、鍵がある経路を通っていた。
3件が失敗し、うち1件は実基体のロードまで進んだ。

**周囲の状態でテストの結論が変わるなら、それはテストではない。** 提出物は審査員の
環境で同じ結果を出さねばならず、開発機に鍵があるかどうかで挙動が変わってはならない。

既定では .env を一切見ない。鍵の読み込みそのものを検証するテストは、自分で
`SEARCH_PATHS` を tmp_path へ差し替えて明示的に opt-in する。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from substrate_probe import env_config as ec
from substrate_probe import openai_backend as ob


@pytest.fixture(autouse=True)
def isolate_ambient_secrets(monkeypatch):
    """開発機の .env と環境変数を、既定で見えなくする。

    自分で SEARCH_PATHS を差し替えるテストは、この fixture のあとに monkeypatch を
    重ねるため、そちらが勝つ(opt-in できる)。
    """
    monkeypatch.setattr(ec, "SEARCH_PATHS", ())
    monkeypatch.delenv(ob.DEFAULT_API_KEY_ENV, raising=False)
