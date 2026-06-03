"""系统提示增强包（System Prompt Addendum Loader）

通用原则：
  LLM 行为合规很大程度依赖 system prompt 的强度。
  本模块负责加载 `src/middleware/prompts/system_addendum.md`，
  并在 agent 启动时把它拼接到 system 消息末尾。
  与具体 skill / 具体业务无关，所有 LLM 决策都受其约束。
"""
from __future__ import annotations

import threading
from functools import lru_cache
from pathlib import Path


_ADDENDUM_PATH = Path(__file__).parent / "system_addendum.md"
_lock = threading.Lock()


@lru_cache(maxsize=1)
def load_system_addendum(path: str | None = None) -> str:
    """读取 system_addendum.md 内容并缓存。

    Args:
        path: 可选的自定义路径；默认读 src/middleware/prompts/system_addendum.md

    Returns:
        完整的 addendum 文本（含 markdown 标记）。如果文件不存在返回空字符串。
    """
    p = Path(path) if path else _ADDENDUM_PATH
    try:
        with _lock:
            if not p.exists():
                return ""
            return p.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def has_addendum() -> bool:
    """检查 addendum 文件是否存在。"""
    return _ADDENDUM_PATH.exists()


__all__ = [
    "load_system_addendum",
    "has_addendum",
]
