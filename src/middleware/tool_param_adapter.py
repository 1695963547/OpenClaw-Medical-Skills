"""工具参数归一化（Tool Parameter Adapter）—— 解决 P1

通用原则：
  LangChain `@tool` 装饰器会从函数签名生成 Pydantic schema，
  当 LLM 调用工具时使用 schema 中未声明的字段名，验证立刻失败。
  本模块提供两种互补的修复策略：

策略 A（推荐）：在工具函数内显式列出常见别名作为可选参数，
             借助"所有参数都有默认值"使 Pydantic schema 永远合法，
             函数体内部再做归一化。

策略 B（兜底）：对纯 Python 工具函数使用 `normalize_tool_kwargs` 装饰器，
             在函数被调用前/失败后尝试用别名表重写 kwargs 并重试。

新增任何工具/字段时，只需要在 COMMON_ALIASES 中加一行，
所有应用此中间件的工具都自动受益。
"""
import re
from functools import wraps
from typing import Any, Callable

# ────────────────────────────────────────────────────────────────
# 通用别名表（与具体 skill / 具体工具解耦）
# 任何工具只要在 COMMON_ALIASES 登记了别名，调用时就会被自动归一化。
# ────────────────────────────────────────────────────────────────
COMMON_ALIASES: dict[str, str] = {
    # 命名风格：单复数 / 简写
    "task_id": "step_number",
    "step_id": "step_number",
    "step": "step_number",
    "step_idx": "step_number",
    "step_idx_0": "step_number",      # 0-indexed
    "step_idx1": "step_number",        # 1-indexed
    "stepindex": "step_number",
    "task_index": "step_number",      # 兼容旧调用
    "task_index_0": "step_number",
    "taskindex": "step_number",
    "index": "step_number",
    "idx": "step_number",
    "n": "step_number",
    "no": "step_number",
    "num": "step_number",
    "number": "step_number",
    "i": "step_number",
    "k": "step_number",
    # camelCase → snake_case
    "stepNumber": "step_number",
    "taskId": "step_number",
    "stepId": "step_number",
    "taskIndex": "step_number",
    "resultSummary": "result_summary",
    "relatedSkill": "related_skill",
    "stepType": "step_type",
    "skillId": "skill_id",
    "filePath": "file_path",
    "maxResults": "max_results",
    "pageSize": "page_size",
    # 医学/生物 API 常见别名
    "indication": "disease_query",
    "disease": "disease_query",
    "condition": "disease_query",
    "target_name": "target_chembl_id",
    "gene": "target_chembl_id",
    "target": "target_chembl_id",
    "drug_name": "drug_chembl_id",
    "compound": "drug_chembl_id",
    # execute_code 常见别名
    "codex": "code",
    "src": "code",
    "source": "code",
    "script": "code",
    "lang": "language",
    "prog": "language",
    "skill": "related_skill",
    "skill_name": "related_skill",
    # read_file 常见别名
    "path": "file_path",
    "filepath": "file_path",
    "filename": "file_path",
    "file": "file_path",
    "skill": "skill_id",
    "name": "skill_id",
    "max_lines": "limit",
    "maxlines": "limit",
    "num_lines": "limit",
    "start_line": "offset",
    "startline": "offset",
    # retrieve_skills 常见别名
    "q": "query",
    "search": "query",
    "keyword": "query",
    "keywords": "query",
    "text": "query",
}


# ────────────────────────────────────────────────────────────────
# 工具函数：扫描文本中出现的潜在字段名
# ────────────────────────────────────────────────────────────────
_FIELD_TOKEN_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")


def extract_unknown_fields(error_message: str) -> set[str]:
    """从 Pydantic ValidationError / LangChain ToolMessage 错误体中
    抽取"未知字段"或"缺失字段"名。

    支持的格式示例：
      - "step_number: Field required"
      - "update_task_status() got an unexpected keyword argument 'task_id'"
      - "validation error for find_drugs_by_indicationArguments"
    """
    fields: set[str] = set()
    # 1) missing 字段
    for m in re.finditer(r"'?(\w+)'?\s*[:\s]\s*Field required", error_message):
        fields.add(m.group(1))
    # 2) unexpected keyword
    for m in re.finditer(r"unexpected keyword argument\s+['\"](\w+)['\"]", error_message):
        fields.add(m.group(1))
    # 3) Pydantic v2 风格
    for m in re.finditer(r"type=missing[^\]]*?input_value=\{([^}]+)\}", error_message):
        for name in _FIELD_TOKEN_RE.findall(m.group(1)):
            fields.add(name)
    return fields


def apply_aliases(kwargs: dict[str, Any]) -> dict[str, Any]:
    """把 kwargs 中出现在 COMMON_ALIASES 的键替换为标准字段名。
    仅当标准字段尚未被显式赋值时才覆盖（避免覆盖 LLM 主动给出的精确值）。
    """
    renamed = dict(kwargs)
    for old, new in list(COMMON_ALIASES.items()):
        if old in renamed and new not in renamed:
            renamed[new] = renamed.pop(old)
    return renamed


# ────────────────────────────────────────────────────────────────
# 策略 A：健壮签名生成器（推荐）
# 让任何 tool 函数自动接受常见别名。
# 用法：见 tools.py 中 update_task_status 的实际实现（直接列出所有别名为可选参数）。
# ────────────────────────────────────────────────────────────────
def robust_signature_aliases(field: str) -> list[str]:
    """给定一个标准字段名，返回所有可能的别名（来自 COMMON_ALIASES）。
    工具作者可据此为自己的函数批量声明可选别名参数。
    """
    return [alias for alias, target in COMMON_ALIASES.items() if target == field]


# ────────────────────────────────────────────────────────────────
# 策略 B：装饰器（兜底）
# 对无法改签名的工具，使用此装饰器做运行时归一化。
# ────────────────────────────────────────────────────────────────
def normalize_tool_kwargs(func: Callable) -> Callable:
    """装饰器：在工具函数被调用前自动归一化 kwargs。

    注意：LangChain `@tool` 装饰过的函数在 ToolNode 调用时已通过 Pydantic 校验，
    所以本装饰器主要针对 *校验失败后重试* 的场景。
    对签名已用"所有参数可选 + 别名映射"策略的工具（策略 A），本装饰器是冗余的，
    保留作为兜底以防未来工具漏改。
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (TypeError, KeyError) as e:
            err = str(e)
            unknown = extract_unknown_fields(err)
            if not unknown:
                raise
            # 尝试移除未知字段（如果它们其实是别名）
            renamed = apply_aliases(kwargs)
            if renamed != kwargs:
                # 移除原始未知字段（已经被 apply_aliases 复制到新键）
                for k in unknown:
                    renamed.pop(k, None)
                return func(*args, **renamed)
            raise

    return wrapper


__all__ = [
    "COMMON_ALIASES",
    "extract_unknown_fields",
    "apply_aliases",
    "robust_signature_aliases",
    "normalize_tool_kwargs",
]
