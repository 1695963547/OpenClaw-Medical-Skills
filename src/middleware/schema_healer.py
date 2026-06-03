"""GraphQL Schema 自愈层（Schema Healer）—— 解决 P2

通用原则：
  GraphQL 服务端升级时，字段会重命名 / 移除 / 改参数类型。
  服务端在 400 错误中通常会附带 "Did you mean 'X'?" 提示。
  本模块：

  1. 解析 400 错误体中的字段重命名建议
  2. 维护"已知坏字段"黑名单 → 避免下次重蹈覆辙
  3. 提供查询重写功能（保守策略：仅做字段名替换，不动整体结构）
  4. 生成"反向教育"提示，让 LLM 下一轮决策时知道正确字段名

  全部基于纯字符串处理（不引入 graphql-core 等重依赖），
  与具体 API 无关 —— 任何 GraphQL 服务（OpenTargets、Hasura、Apollo、GitHub、Linear 等）都受益。
"""
from __future__ import annotations

import re
import threading
from typing import Optional


# ────────────────────────────────────────────────────────────────
# 错误体解析
# ────────────────────────────────────────────────────────────────
# 常见 GraphQL 错误模式（兼容多家实现）
_FIELD_NOT_FOUND_RE = re.compile(
    r"Cannot query field\s+['\"](?P<old>[A-Za-z_]\w*)['\"]\s+on type\s+['\"](?P<type>\w+)['\"]"
    r"(?:\s*\.\s*Did you mean\s+(?P<suggestions>[^?]+))?",
    re.IGNORECASE,
)
_UNKNOWN_ARG_RE = re.compile(
    r"Unknown argument\s+['\"](?P<arg>\w+)['\"]\s+on field\s+['\"](?P<field>\w+)['\"]"
    r"(?:\s+of type\s+['\"][^'\"]+['\"])?"
    r"(?:\s*\.\s*Did you mean\s+(?P<suggestions>[^?]+))?",
    re.IGNORECASE,
)
# Field 名字可能含点（如 Pagination.index），所以 [A-Za-z_][\w.]* 允许
_REQUIRED_FIELD_RE = re.compile(
    r"Field\s+['\"](?P<field>[A-Za-z_][\w.]*)['\"][^.]*?of required type\s+['\"](?P<type>[^'\"]+)['\"]\s+was not provided",
    re.IGNORECASE,
)


def _parse_suggestion_list(s: str) -> list[str]:
    """从 'Did you mean' 字符串中抽取候选字段名列表。

    形如: "Did you mean 'associatedDrugs', 'knownDrugsList', or 'targetInteractions'?"
    """
    return [m.group(1) for m in re.finditer(r"'([A-Za-z_]\w*)'", s)]


# ────────────────────────────────────────────────────────────────
# 已知坏字段名（黑名单）
#  启动时/使用中持续累积。
#  维护两组：
#    (old_field, parent_type) → list of new_field candidates
#    old_field → new_field（粗粒度，只看字段名不看 parent）
# ────────────────────────────────────────────────────────────────
class _RenameMap:
    """线程安全的重命名缓存。"""

    def __init__(self):
        # (old_field, parent_type) -> [new_field, ...]
        self._by_type: dict[tuple[str, str], list[str]] = {}
        # old_field -> [new_field, ...]（跨 type 合并）
        self._by_field: dict[str, list[str]] = {}
        self._lock = threading.Lock()

    def add(self, old: str, new: str, parent_type: str = "") -> None:
        with self._lock:
            if parent_type:
                key = (old, parent_type)
                lst = self._by_type.setdefault(key, [])
                if new not in lst:
                    lst.append(new)
            lst2 = self._by_field.setdefault(old, [])
            if new not in lst2:
                lst2.append(new)

    def suggest(self, old: str, parent_type: str = "") -> list[str]:
        """返回 old 字段的所有候选新名（优先 type-specific，再 fallback to global）。"""
        with self._lock:
            if parent_type and (old, parent_type) in self._by_type:
                return list(self._by_type[(old, parent_type)])
            return list(self._by_field.get(old, []))

    def known_broken_fields(self) -> set[tuple[str, str]]:
        with self._lock:
            return set(self._by_type.keys())

    def all_renames(self) -> dict[str, list[str]]:
        with self._lock:
            return {k: list(v) for k, v in self._by_field.items()}

    def clear(self) -> None:
        with self._lock:
            self._by_type.clear()
            self._by_field.clear()


# ────────────────────────────────────────────────────────────────
# 主类：GraphQLHealer
# ────────────────────────────────────────────────────────────────
class GraphQLHealer:
    """线程安全的 GraphQL 错误解析 + 重写器。

    用法：
        healer = GraphQLHealer.shared()
        # 1) 解析错误体，记录重命名建议
        hints = healer.learn_from_error(error_text, parent_type="Target")
        # 2) 构造给 LLM 的反向教育提示
        if hints:
            msg = healer.build_learning_hint(hints)
        # 3) 主动重写查询（保守策略：仅做字段名替换）
        new_query = healer.rewrite_query(old_query, parent_type="Target")
        # 4) 检查 LLM 想用的字段是否在黑名单中
        if healer.is_known_broken("knownDrugs", "Target"):
            ...
    """

    _shared: Optional["GraphQLHealer"] = None
    _shared_lock = threading.Lock()

    def __init__(self):
        self._renames = _RenameMap()

    @classmethod
    def shared(cls) -> "GraphQLHealer":
        """进程级单例（多线程安全）。"""
        if cls._shared is None:
            with cls._shared_lock:
                if cls._shared is None:
                    cls._shared = cls()
        return cls._shared

    # ── 1) 错误解析 ──
    def learn_from_error(self, error_body: str, parent_type: str = "") -> list[dict]:
        """从 GraphQL 错误体中抽取字段重命名建议并记录到缓存。

        Returns:
            list of dict: 每条建议形如
            {"old": "knownDrugs", "new": "associatedDrugs", "type": "Target", "kind": "field_rename" | "arg_rename" | "missing_required" | "unknown_field"}
        """
        if not error_body:
            return []
        hints: list[dict] = []

        # 1.1 解析 "Cannot query field 'X' on type 'Y'"
        for m in _FIELD_NOT_FOUND_RE.finditer(error_body):
            old = m.group("old")
            ptype = m.group("type") or parent_type
            sug_str = m.group("suggestions") or ""
            for new in _parse_suggestion_list(sug_str):
                if new != old:
                    self._renames.add(old, new, ptype)
                    hints.append({
                        "kind": "field_rename",
                        "old": old,
                        "new": new,
                        "type": ptype,
                    })
            # 即使没建议也记录 "这个字段不存在"
            if not _parse_suggestion_list(sug_str):
                hints.append({
                    "kind": "unknown_field",
                    "old": old,
                    "new": None,
                    "type": ptype,
                })

        # 1.2 解析 "Unknown argument 'X' on field 'Y'"
        for m in _UNKNOWN_ARG_RE.finditer(error_body):
            arg = m.group("arg")
            field = m.group("field")
            sug_str = m.group("suggestions") or ""
            for new_arg in _parse_suggestion_list(sug_str):
                if new_arg != arg:
                    # 参数重命名：原参数 → 新参数
                    self._renames.add(arg, new_arg, f"arg:{field}")
                    hints.append({
                        "kind": "arg_rename",
                        "old": arg,
                        "new": new_arg,
                        "type": f"arg:{field}",
                    })
            if not _parse_suggestion_list(sug_str):
                hints.append({
                    "kind": "unknown_arg",
                    "old": arg,
                    "new": None,
                    "type": f"arg:{field}",
                })

        # 1.3 解析 "Field 'X' of required type 'Y' was not provided"
        for m in _REQUIRED_FIELD_RE.finditer(error_body):
            hints.append({
                "kind": "missing_required",
                "old": m.group("field"),
                "new": None,
                "type": m.group("type"),
            })

        return hints

    # ── 2) 反向教育提示 ──
    def build_learning_hint(self, hints: list[dict] | None = None) -> str:
        """构造给 LLM 看的"反向教育"消息。

        提示 LLM：
        - 哪些字段已经确认不可用（不要重试）
        - 用什么新字段代替
        - 必要时读 SKILL.md 看正确 schema
        """
        if hints is None:
            hints = []
        # 累计所有已知重命名
        renames = self._renames.all_renames()
        if not hints and not renames:
            return ""

        lines: list[str] = ["[SchemaHealer] GraphQL Schema 错误自愈报告："]
        if hints:
            lines.append("")
            lines.append("本次错误的新发现：")
            for h in hints[:8]:
                if h["kind"] == "field_rename":
                    lines.append(
                        f"  • 字段重命名: type=`{h['type']}` 上 `'{h['old']}'` "
                        f"→ 改用 `'{h['new']}'`"
                    )
                elif h["kind"] == "unknown_field":
                    lines.append(
                        f"  • 字段不存在: type=`{h['type']}` 上无 `{h['old']}` 字段"
                    )
                elif h["kind"] == "arg_rename":
                    lines.append(
                        f"  • 参数重命名: `{h['type'].replace('arg:', '')}({h['old']})` "
                        f"→ 改用 `({h['new']})`"
                    )
                elif h["kind"] == "unknown_arg":
                    lines.append(
                        f"  • 参数不存在: `{h['type'].replace('arg:', '')}` 无 `{h['old']}` 参数"
                    )
                elif h["kind"] == "missing_required":
                    lines.append(
                        f"  • 缺少必填参数: `{h['old']}` (类型 `{h['type']}`)"
                    )
        if renames:
            lines.append("")
            lines.append("已积累的重命名规则（不要违反）：")
            for old, news in list(renames.items())[:12]:
                lines.append(f"  • `{old}` → {news}")

        lines.append("")
        lines.append("⚠️ 下次请用上面的新字段名。")
        lines.append("如不确定 schema，先调用 read_file(skill_id, 'SKILL.md') 查阅文档。")
        return "\n".join(lines)

    # ── 3) 查询重写（保守） ──
    def rewrite_query(
        self,
        query: str,
        parent_type: str = "",
        max_replacements: int = 5,
    ) -> tuple[str, list[dict]]:
        """根据已学习的重命名规则，保守地重写 GraphQL 查询。

        Returns:
            (rewritten_query, applied_renames)

        策略：
          - 只做"裸字段名"替换（用 \\b 边界）
          - 不会动 alias、fragment、变量
          - 一次最多替换 max_replacements 个字段（防止误改）
          - 返回 applied_renames 让调用方知晓
        """
        if not query:
            return query, []
        renames = self._renames.all_renames()
        if not renames:
            return query, []

        applied: list[dict] = []
        new_query = query
        for old, news in renames.items():
            if not news:
                continue
            if len(applied) >= max_replacements:
                break
            # 用 \b 边界避免替换到子串（如 knownDrugsList 不会变）
            # 用 re.sub 而非 str.replace，规避部分替换
            new_field = news[0]
            pattern = r"\b" + re.escape(old) + r"\b"
            if re.search(pattern, new_query):
                # 排除 alias 场景（如 myKnownDrugs: knownDrugs 中的 myKnownDrugs）
                # alias 通常是 "<name>:"，我们只替换无 ":" 前缀的
                # 简单起见：先检查替换后是否会破坏 alias
                # 用 negative lookbehind 排除 ":<space>?knownDrugs" 情况
                safe_pattern = r"(?<![\w:.])(?<![\w:])" + re.escape(old) + r"\b"
                count_before = len(re.findall(safe_pattern, new_query))
                if count_before > 0:
                    new_query = re.sub(safe_pattern, new_field, new_query)
                    applied.append({
                        "old": old,
                        "new": new_field,
                        "count": count_before,
                    })
        return new_query, applied

    # ── 4) 主动检查 ──
    def is_known_broken(self, field: str, parent_type: str = "") -> bool:
        """检查某字段是否已知是坏的（避免 LLM 重复使用）。"""
        if not field:
            return False
        if parent_type and (field, parent_type) in self._renames.known_broken_fields():
            return True
        return field in self._renames.all_renames()

    def suggest_replacement(self, field: str, parent_type: str = "") -> Optional[str]:
        """推荐一个替代字段名（没有则返回 None）。"""
        cands = self._renames.suggest(field, parent_type)
        return cands[0] if cands else None

    def snapshot(self) -> dict:
        """导出当前所有重命名规则（用于诊断 / 持久化）。"""
        return {
            "by_type": {f"{k[0]}|{k[1]}": v for k, v in self._renames._by_type.items()},
            "by_field": self._renames.all_renames(),
        }

    def reset(self) -> None:
        """清空所有积累（调试用）。"""
        self._renames.clear()


# ────────────────────────────────────────────────────────────────
# 便利函数：直接用 healer = GraphQLHealer.shared()
# ────────────────────────────────────────────────────────────────
def parse_graphql_error(error_body: str, parent_type: str = "") -> list[dict]:
    """便利函数：直接解析 + 记录。"""
    return GraphQLHealer.shared().learn_from_error(error_body, parent_type)


def get_rewrite_hint() -> str:
    """便利函数：获取当前所有重命名规则的提示。"""
    return GraphQLHealer.shared().build_learning_hint()


__all__ = [
    "GraphQLHealer",
    "parse_graphql_error",
    "get_rewrite_hint",
]
