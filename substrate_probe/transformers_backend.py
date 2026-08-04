"""実基体バックエンド — 常駐する Qwen3.5 を transformers で駆動する。

Ollama には依存しない。実測(2026-08-02)で llama-server.exe が欠損しており生成不能だったが、
より本質的な理由は、提出物が審査員の環境で再現できねばならないことである。壊れやすい中間層を
直すのではなく、不要にする。

実測値(RTX 3060 12GB / Qwen3.5-9B nf4):
  初回ロード 634秒 -> 量子化済みを保存して再利用(7.2GB)
  モデル 7.65GB / 4体バッチ時ピーク 7.91GB(空き 11.8GB に対し余裕あり)
  単体 3.99 tok/s / 4体同時 18.82 tok/s(4.7倍)

thinking mode は既定で無効にする(実測: 有効だと生成枠を思考に使い切り結論に至らない)。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

from .llm_runtime import Backend, GenerationConfig, RuntimeError_

# 量子化済みの保存先(初回ロードの 634秒を二度払わないため)。
DEFAULT_QUANTIZED_PATH = Path(r"E:\masa-agi-hybrid\quantized\Qwen3.5-9B-nf4")
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-9B"


@dataclass
class TransformersBackend:
    """常駐モデル。プロセスが生きている限りロードは一度きりで償却される。

    model_path が存在すればそちらを使う(量子化済み)。無ければ model_id から読み、
    4bit 量子化する。どちらも失敗したら黙って縮退せず例外にする。
    """
    model_path: Optional[Path] = None
    model_id: str = DEFAULT_MODEL_ID
    device: str = "cuda:0"
    # 必要とする空きVRAM(GB)。実測 2026-08-02: 9B nf4 はモデル 7.65GB + バッチで 7.91GB。
    # デスクトップ(Chrome等)が数GBを保持するため、空きは時間帯で変動する。
    required_free_gb: float = 8.5
    allow_spillover: bool = False
    _model: Any = field(default=None, repr=False)
    _tokenizer: Any = field(default=None, repr=False)

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def free_vram_gb(self) -> Optional[float]:
        """実際に確保できる空きVRAM(GB)。測れなければ None(捏造しない)。"""
        try:
            import torch
            if not torch.cuda.is_available():
                return None
            return torch.cuda.mem_get_info()[0] / 1e9
        except Exception:
            return None

    def check_headroom(self) -> Tuple[bool, str]:
        """ロード前に空きVRAMを実測する。

        実測 2026-08-02 の教訓: デスクトップ(Chrome/iCloud等)が約6GBを保持している状態で
        7.65GB のモデルを載せると、Windows WDDM がシステムRAMへ退避し **48倍** 遅くなった。
        黙って劣化するのではなく、載せる前に足りないと言う(fail-closed)。
        """
        free = self.free_vram_gb()
        if free is None:
            return False, "空きVRAMを測れない(CUDAが無い、または取得失敗)"
        if free < self.required_free_gb:
            return False, (f"空きVRAM {free:.2f}GB < 必要 {self.required_free_gb:.2f}GB。"
                           f"デスクトップアプリがVRAMを保持している可能性がある。"
                           f"退避が起きると数十倍遅くなる")
        return True, f"空きVRAM {free:.2f}GB(必要 {self.required_free_gb:.2f}GB)"

    def load(self) -> "TransformersBackend":
        """モデルを常駐させる。すでにロード済みなら何もしない(冪等)。

        空きVRAMが足りなければ例外にする。allow_spillover=True のときだけ、
        遅くなることを承知で続行する(黙って遅くなることは許さない)。
        """
        if self.loaded:
            return self
        ok, detail = self.check_headroom()
        if not ok and not self.allow_spillover:
            raise RuntimeError_(
                f"VRAM不足のためロードを中止した: {detail}。"
                f"承知のうえで続けるなら allow_spillover=True を指定する")
        try:
            import torch
            from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                      BitsAndBytesConfig)
        except ImportError as e:
            raise RuntimeError_(f"実基体の依存が無い: {e}") from e

        src = self.model_path or DEFAULT_QUANTIZED_PATH
        use_saved = Path(src).exists()
        target = str(src) if use_saved else self.model_id

        tok = AutoTokenizer.from_pretrained(target)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"   # 生成では左詰めが正しい(右詰めは出力を壊す)

        kwargs = {"device_map": self.device}
        if not use_saved:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True)
        self._model = AutoModelForCausalLM.from_pretrained(target, **kwargs)
        self._tokenizer = tok
        return self

    def _apply_template(self, prompt: str, thinking: bool) -> str:
        """chat template を適用する。thinking の無効化はモデルによって引数が異なるため、
        受け付けない実装では黙って落とさず既定に戻す(嘘の設定をしない)。
        """
        msgs = [{"role": "user", "content": prompt}]
        try:
            return self._tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False,
                enable_thinking=thinking)
        except TypeError:
            return self._tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False)

    def generate(self, prompts: Sequence[str],
                 config: GenerationConfig) -> Tuple[str, ...]:
        """一括生成。バッチであることが並列思考の実体(実測 4.7倍)。"""
        if not prompts:
            raise RuntimeError_("プロンプトが空")
        if not self.loaded:
            self.load()
        import torch

        if config.seed is not None:
            torch.manual_seed(config.seed)

        texts = [self._apply_template(p, config.thinking) for p in prompts]
        enc = self._tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self._model.generate(
                **enc,
                max_new_tokens=config.max_new_tokens,
                do_sample=True,
                temperature=config.temperature,
                top_p=config.top_p,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        cut = enc["input_ids"].shape[-1]
        return tuple(
            self._tokenizer.decode(row[cut:], skip_special_tokens=True)
            for row in out
        )

    def vram_gb(self) -> Optional[float]:
        """現在の VRAM 使用量(GB)。測れなければ None(捏造しない)。"""
        try:
            import torch
            if not torch.cuda.is_available():
                return None
            return torch.cuda.memory_allocated() / 1e9
        except Exception:
            return None
