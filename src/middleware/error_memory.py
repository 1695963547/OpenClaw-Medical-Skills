"""错误记忆与策略阶梯（Error Memory & Strategy Escalator）—— 解决 P6

通用原则：
  同一个错误反复出现是 agent 失控的最常见征兆。
  本模块提供：

  1. ErrorMemory       跨 turn / 跨 step 跟踪错误指纹；
                        同一指纹累计到阈值后输出"反向教育提示"给 LLM。
  2. StrategyEscalator 失败次数递进时，从"重试" → "换工具/API" → "切本地知识库" → "终止"。

  这两个组件与具体 skill / 具体 API 无关，
  适用于任何 LLM Agent 控制流。

设计要点：
  - 按 session_id 隔离（与 src.tools._executor 同样的线程局部单例模式）
  - 错误指纹算法对任何错误字符串都鲁棒（去 ID、去数字、去引号内容）
  - 提示生成使用模板化语言，避免硬编码业务术语
"""
from __future__ import annotations

import hashlib
import re
import threading
from collections import defaultdict
from typing import Optional


# ────────────────────────────────────────────────────────────────
# 错误指纹（去噪音）
# ────────────────────────────────────────────────────────────────
_NORMALIZE_NUMBER_RE = re.compile(r"\b\d+\b")
_NORMALIZE_STRING_RE = re.compile(r"'[^']{0,40}'|\"[^\"]{0,40}\"")
_NORMALIZE_PATH_RE = re.compile(r"[A-Za-z]:\\[^\s)]+|(?:\.{0,2}/)+[\w./]+")
_NORMALIZE_URL_RE = re.compile(r"https?://[^\s)]+")


def _fingerprint(error: str) -> str:
    """把错误字符串归一化为短指纹（12 字符 hex）。

    目的：相同的"根因"在多次出现时会被识别为同一指纹，
    即使报错内容中的具体数字、ID、路径、URL 变了。
    """
    if not error:
        return "empty"
    norm = error
    # 先去掉 URL/路径（避免 URL 里的 query string 干扰）
    norm = _NORMALIZE_URL_RE.sub("<URL>", norm)
    norm = _NORMALIZE_PATH_RE.sub("<PATH>", norm)
    # 再去掉引号内容（'xxx'、"xxx"）
    norm = _NORMALIZE_STRING_RE.sub("'<X>'", norm)
    # 最后去掉裸数字
    norm = _NORMALIZE_NUMBER_RE.sub("N", norm)
    # 折叠空白
    norm = re.sub(r"\s+", " ", norm).strip()
    return hashlib.md5(norm.encode("utf-8")).hexdigest()[:12]


def _classify_root_cause(error: str) -> str:
    """从错误字符串粗略归类根因（用于反向教育提示的标签）。

    标签是稳定的、面向 LLM 的"概念"，不绑定具体业务。
    """
    e = error.lower()
    if "modulenotfounderror" in e or "importerror" in e or "no module named" in e:
        return "missing_module"
    if "cannot query field" in e or "did you mean" in e:
        return "schema_drift"
    if "404" in e or "not found" in e or "unknown tool" in e:
        return "resource_not_found"
    if "timeout" in e or "timed out" in e or "timeouterror" in e:
        return "network_timeout"
    if "401" in e or "unauthorized" in e or "api key" in e:
        return "auth_failed"
    if "429" in e or "rate limit" in e:
        return "rate_limited"
    if "winerror" in e or "system cannot find" in e or "no such file" in e:
        return "platform_mismatch"
    if "field required" in e or "missing" in e and "argument" in e:
        return "param_missing"
    if "typeerror" in e or "has no attribute" in e or "attributeerror" in e:
        return "type_mismatch"
    if "valueerror" in e or "keyerror" in e:
        return "data_shape"
    return "generic"


# ────────────────────────────────────────────────────────────────
# 错误记忆
# ────────────────────────────────────────────────────────────────
class ErrorMemory:
    """跨 step 跟踪错误指纹。

    用法：
        mem = get_error_memory(session_id)
        count = mem.record(error_string, tool_name="execute_code")
        if mem.should_escalate(error_string):
            hint = mem.build_learning_hint()
            inject_into_llm(hint)
    """

    DEFAULT_THRESHOLD = 2

    def __init__(self, threshold: int = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self._counts: dict[str, int] = defaultdict(int)
        self._samples: dict[str, tuple[str, str]] = {}     # fp -> (cause, sample)
        self._tools: dict[str, str] = {}                   # fp -> tool_name
        self._lock = threading.Lock()

    def record(
        self,
        error: str,
        tool_name: str = "",
    ) -> int:
        """记录一次错误，返回累计次数。"""
        if not error:
            return 0
        fp = _fingerprint(error)
        cause = _classify_root_cause(error)
        with self._lock:
            self._counts[fp] += 1
            if fp not in self._samples:
                # 截断长错误到 280 字符
                self._samples[fp] = (cause, error[:280])
            if tool_name:
                self._tools[fp] = tool_name
            return self._counts[fp]

    def should_escalate(self, error: str) -> bool:
        """判断此错误是否已达到升级阈值。"""
        if not error:
            return False
        return self._counts.get(_fingerprint(error), 0) >= self.threshold

    def get_count(self, error: str) -> int:
        if not error:
            return 0
        return self._counts.get(_fingerprint(error), 0)

    def build_learning_hint(self, error: Optional[str] = None) -> str:
        """构造一段给 LLM 看的"反向教育"提示。

        - 如果传入 error：只针对此错误给出提示
        - 如果不传：给出当前出现次数最多的错误
        """
        if error:
            fp = _fingerprint(error)
        else:
            if not self._counts:
                return ""
            fp = max(self._counts.items(), key=lambda kv: kv[1])[0]
        cause, sample = self._samples.get(fp, ("generic", ""))
        count = self._counts[fp]
        tool_hint = ""
        if fp in self._tools:
            tool_hint = f"（出现在 `{self._tools[fp]}` 工具中）"
        return (
            f"[ErrorMemory] 同一类错误已出现 {count} 次{tool_hint}，"
            f"根因归类: **{cause}**。\n"
            f"  最近一次样例:\n    {sample}\n\n"
            f"⚠️ 请换一个**完全不同的方法**（不要再重试同样的代码/参数）。"
            f"建议策略：\n"
            f"  1. 如果是 {cause}：换工具/换 API/换数据源，或切到本地知识库\n"
            f"  2. 如确无替代方案：直接基于已有信息给用户回复并说明限制\n"
            f"  3. 不要在同一路径上重试 ≥2 次"
        )

    def top_error(self) -> Optional[tuple[str, int, str]]:
        """返回 (指纹, 次数, 根因标签) 最多的错误，若无错误则 None。"""
        if not self._counts:
            return None
        fp, count = max(self._counts.items(), key=lambda kv: kv[1])
        cause, _ = self._samples.get(fp, ("generic", ""))
        return fp, count, cause

    def reset(self) -> None:
        """清空记忆（调试/单测用）。"""
        with self._lock:
            self._counts.clear()
            self._samples.clear()
            self._tools.clear()


# ────────────────────────────────────────────────────────────────
# 策略阶梯
# ────────────────────────────────────────────────────────────────
class StrategyEscalator:
    """失败次数递进时，从轻到重的兜底策略。

    阶段 1（attempts=1）: 注入学习提示 + 允许重试
    阶段 2（attempts=2）: 建议换工具/换 API
    阶段 3（attempts=3）: 切到本地知识库 / 模拟数据
    阶段 4+         : 强制终止，要求 LLM 总结
    """

    def __init__(self):
        self.attempts = 0

    def next_attempt(self) -> str:
        self.attempts += 1
        return {
            1: "retry_with_hint",
            2: "switch_tool",
            3: "use_local_kb",
        }.get(self.attempts, "terminate")

    def should_force_terminate(self) -> bool:
        return self.attempts >= 4


# ────────────────────────────────────────────────────────────────
# Session 隔离的单例存储
# ────────────────────────────────────────────────────────────────
_session_local = threading.local()
_memories: dict[str, ErrorMemory] = {}
_escalators: dict[str, StrategyEscalator] = {}
_memories_lock = threading.Lock()


def set_memory_session_id(session_id: str) -> None:
    """由 main.py / agent.py 在切换 session 时调用。"""
    _session_local.session_id = session_id


def get_memory_session_id() -> str:
    return getattr(_session_local, "session_id", "")


def get_error_memory(session_id: Optional[str] = None) -> ErrorMemory:
    """获取当前 session 的 ErrorMemory（懒创建）。"""
    sid = session_id or get_memory_session_id() or "_default_"
    with _memories_lock:
        if sid not in _memories:
            _memories[sid] = ErrorMemory()
        return _memories[sid]


def get_escalator(session_id: Optional[str] = None) -> StrategyEscalator:
    sid = session_id or get_memory_session_id() or "_default_"
    with _memories_lock:
        if sid not in _escalators:
            _escalators[sid] = StrategyEscalator()
        return _escalators[sid]


def cleanup_memory(session_id: str) -> None:
    """清理指定 session 的记忆（benchmark 任务完成后调用，避免内存泄漏）。"""
    with _memories_lock:
        _memories.pop(session_id, None)
        _escalators.pop(session_id, None)


__all__ = [
    "ErrorMemory",
    "StrategyEscalator",
    "set_memory_session_id",
    "get_memory_session_id",
    "get_error_memory",
    "get_escalator",
    "cleanup_memory",
    "_fingerprint",
    "_classify_root_cause",
]
