"""実験機サーバ — 本物の言語モデルが探査者として振る舞い、画面へ流れる。

    python -m substrate_probe.serve

ブラウザで http://localhost:8420 を開き、開始を押すと:

    世界を封緘 -> 実験を非対称に配る -> 測定 -> **言語モデルが仮説を書く**
      -> その生の推論が画面へ流れる -> 予測を凍結 -> 照合 -> 開封

公開版アーティファクトの実行環境は外部への通信を許さないため、言語モデルを呼べない。
それが「規則で動く探査者」という妥協の正体だった。本サーバはその制約の外にある ——
**探査者は本物の言語モデルであり、何を考えたかがそのまま観測対象になる。**

設計の規律:

  一. **開封イベントより前に、真値をネットワークへ載せない。**
      SSE で流れる各イベントは、reveal を除き基体の中身を含まない。画面ではなく
      イベント列の水準で二重盲検を守る(開発者ツールで覗いても真値は無い)。

  二. **縮退しない。** 鍵が無ければ「LLMモードは使えない」と言い、理由を返す。
      規則モードで動かして「言語モデルが考えた」ように見せることはしない。

  三. 探査者は全員同一の重みである。それは欠陥ではなく観測対象である ——
      同じ重みが違う証拠を見て、違う結論に至るのか。画面にもそう明記する。
"""
from __future__ import annotations

import json
import queue
import random
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from . import hypothesis as hy
from . import investigation as inv
from . import llm_runtime as rt
from . import llm_scientist as sci
from . import substrate as sb
from . import world as wd

ROOT = Path(__file__).resolve().parent.parent
VIEWER = ROOT / "viewer" / "index.html"

DEFAULT_PORT = 8420

# 探査者の人格。**結論は与えない** —— 何を重く見るかだけを与える。
PERSONAS: Mapping[str, str] = {
    "A0": "あなたは測定の異常をまず疑う探査者である。装置の癖と世界の性質を区別したがる。",
    "A1": "あなたは保存則を重んじる探査者である。総量が合わないことに敏感である。",
    "A2": "あなたは世界の果てを気にする探査者である。どこまで行けるかを知りたがる。",
    "A3": "あなたは時間の流れ方に注目する探査者である。速さの違いに理由を求める。",
}
AGENT_IDS: Tuple[str, ...] = tuple(PERSONAS)


class ServerError(RuntimeError):
    """実験機の前提を壊す要求を黙って通さない。"""


@dataclass
class Run:
    """一回の探査走行。イベント列が唯一の出力である。"""
    run_id: str
    events: "queue.Queue[Optional[dict]]" = field(default_factory=queue.Queue)
    done: bool = False

    def emit(self, kind: str, **payload) -> None:
        self.events.put({"type": kind, "at": time.time(), **payload})

    def close(self) -> None:
        self.done = True
        self.events.put(None)


def _make_backend():
    """探査者の基体を用意する。**使えなければ理由を返し、縮退しない。**"""
    from .openai_backend import OpenAIBackend
    backend = OpenAIBackend()
    if not backend.available():
        return None, f"{backend.api_key_env} が未設定"
    failure = backend.preflight()
    if failure is not None:
        return None, f"{failure.kind}: {failure.detail}"
    return backend, backend.identity()


def _investigate(run: Run, backend, substrate_name: str,
                 worlds: int, seed: int, trials: int = 4000) -> None:
    """探査を走らせ、段階をイベントとして流す。

    **reveal より前のイベントは基体の中身を含まない。** 画面ではなくイベント列の
    水準で二重盲検を守る —— 開発者ツールで覗いても真値は無い。
    """
    rng = random.Random(seed)
    try:
        for wix in range(worlds):
            wid = f"W-{wix + 1:02d}"
            truth = sb.random_substrate(rng)
            sealed = sb.seal(truth)
            run.emit("seal", world=wid, seal=sealed.seal_hash, index=wix, total=worlds)

            assignment = inv.assign_experiments(
                AGENT_IDS, overlap=0.0, salt=f"{run.run_id}:{wix}")
            run.emit("deal", world=wid,
                     deal={a: list(v) for a, v in assignment.items()})

            # 測定。数値だけを流す(どの性質を探るかは流さない)。
            world = wd.World(substrate=truth, seed=rng.randrange(10 ** 6))
            observations: Dict[str, List[str]] = {}
            measured: Dict[str, Optional[float]] = {e: None for e in wd.EXPERIMENTS}
            for agent_id in AGENT_IDS:
                observations[agent_id] = []
                for exp in assignment[agent_id]:
                    m = world.run(exp, trials)
                    measured[exp] = m.value
                    observations[agent_id].append(
                        f"実験「{exp}」を{m.trials}回: 測定値 {m.value:.5f}")
                    run.emit("probe", world=wid, agent=agent_id,
                             experiment=exp, value=round(m.value, 5))

            # 言語モデルが考える。一人ずつ —— 生の推論を隠さない。
            hypotheses: List[hy.Hypothesis] = []
            for agent_id in AGENT_IDS:
                run.emit("thinking", world=wid, agent=agent_id)
                prompt = sci.build_hypothesis_prompt(
                    PERSONAS[agent_id], observations[agent_id],
                    assignment[agent_id])
                before_failures = len(getattr(backend, "failures", ()))
                # 仮説は10性質の検討を要する。推論モデルは思考で予算を食うため
                # 広めに取る(実測: 4000 では空応答が出うる)。
                raw = backend.generate([prompt],
                                       rt.GenerationConfig(max_new_tokens=8000))[0]
                ex = sci.extract_hypothesis(raw, agent_id)
                if ex.ok:
                    h = ex.hypothesis
                    hypotheses.append(h)
                    run.emit("hypothesis", world=wid, agent=agent_id,
                             claims={k: bool(v) for k, v in h.claims.items()},
                             overreach=list(h.overreach),
                             reasoning=h.reasoning,
                             predictions=[{"experiment": p.experiment,
                                           "direction": p.direction,
                                           "threshold": p.threshold,
                                           "because": p.because}
                                          for p in h.predictions])
                else:
                    # 基体側の失敗理由を必ず添える。**理由の無い棄却イベントは、
                    # 次に何を直すべきかを observers から奪う** —— 実測 2026-08-05:
                    # 全棄却が起きたとき、画面には「取り出せなかった」しか無く、
                    # 診断が手作業になった。
                    backend_failures = getattr(backend, "failures", ())[before_failures:]
                    run.emit("rejected", world=wid, agent=agent_id,
                             reason=ex.reject_reason or "unknown",
                             detail=ex.detail,
                             backend_detail="; ".join(
                                 f"{f.kind}: {f.detail}" for f in backend_failures),
                             raw_head=ex.raw[:200])

            if len(hypotheses) < 2:
                run.emit("aborted", world=wid,
                         reason="仮説が2件未満では収束を測れない")
                continue

            frozen = [hy.freeze(h) for h in hypotheses]
            run.emit("freeze", world=wid, count=len(frozen))

            judgments = [hy.judge(f, measured) for f in frozen]
            for j in judgments:
                run.emit("judgment", world=wid, **j.to_dict())

            consensus = hy.consensus(hypotheses)
            run.emit("consensus", world=wid,
                     consensus={k: v for k, v in consensus.items()})

            # ここで初めて封を切る。真値が初めてイベントに載る。
            reveal = sb.open_seal(sealed, consensus)
            run.emit("reveal", world=wid, **reveal.to_dict())

        run.emit("finished", worlds=worlds, substrate=substrate_name)
    except Exception as e:  # 走行中の失敗を画面へ届ける(黙って止まらない)
        run.emit("error", detail=f"{type(e).__name__}: {e}")
    finally:
        run.close()


class _Handler(BaseHTTPRequestHandler):
    runs: Dict[str, Run] = {}
    lock = threading.Lock()

    def log_message(self, *args) -> None:  # 標準の雑多なログを黙らせる
        pass

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            body = VIEWER.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/health":
            backend, detail = _make_backend()
            if backend is None:
                # 使えない理由を返す。**縮退して「使える」と言わない。**
                self._json(200, {"llm": False, "reason": detail})
            else:
                self._json(200, {"llm": True, "substrate": detail,
                                 "note": "全探査者は同一の重みである。それは欠陥では"
                                         "なく観測対象である"})
            return
        if self.path.startswith("/api/events/"):
            run_id = self.path.rsplit("/", 1)[-1]
            with self.lock:
                run = self.runs.get(run_id)
            if run is None:
                self._json(404, {"error": "走行が無い"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            while True:
                try:
                    ev = run.events.get(timeout=120)
                except queue.Empty:
                    break
                if ev is None:
                    break
                data = json.dumps(ev, ensure_ascii=False)
                try:
                    self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionAbortedError, OSError):
                    break
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path.startswith("/api/start"):
            backend, detail = _make_backend()
            if backend is None:
                self._json(503, {"error": "LLMモードを開始できない", "reason": detail})
                return
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            worlds = max(1, min(8, int(q.get("worlds", ["3"])[0])))
            run_id = f"r{int(time.time() * 1000):x}"
            run = Run(run_id=run_id)
            with self.lock:
                self.runs[run_id] = run
                # 古い走行を掃除する(イベントを吐き終えたものだけ)
                for k in [k for k, r in self.runs.items() if r.done and k != run_id]:
                    del self.runs[k]
            seed = int(time.time()) & 0xFFFF
            threading.Thread(
                target=_investigate,
                args=(run, backend, detail, worlds, seed),
                daemon=True).start()
            self._json(200, {"run_id": run_id, "worlds": worlds,
                             "substrate": detail})
            return
        self._json(404, {"error": "not found"})


def serve(port: int = DEFAULT_PORT) -> None:
    if not VIEWER.exists():
        raise ServerError(
            "viewer/index.html が無い。python tools/build_viewer.py で生成すること")
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    backend, detail = _make_backend()
    mode = f"LLMモード({detail})" if backend else f"LLM不可({detail})"
    print(f"実験機サーバ: http://localhost:{port}  [{mode}]")
    print("ブラウザで開き、「探査を開始する」を押す。Ctrl+C で停止。")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    serve()
