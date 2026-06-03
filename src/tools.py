"""
LLM 工具函数模块 — 定义 Agent 可调用的 4 个工具
==================================================

本模块定义了 LangGraph Agent 在决策时可使用的全部工具函数：

┌─────────────────────┬────────────────────────────────────────────┐
│ 工具名               │ 功能                                       │
├─────────────────────┼────────────────────────────────────────────┤
│ retrieve_skills     │ 语义检索 869 个生物医学技能（ChromaDB）       │
│ read_file           │ 读取技能目录下的 SKILL.md 或参考文件          │
│ execute_code        │ 在隔离 venv 中执行 Python/Bash/R/JS 代码      │
│ update_task_status  │ 更新多步骤计划的子任务状态                    │
└─────────────────────┴────────────────────────────────────────────┘

此外还提供依赖自动安装的 3 层防御机制：
  第 1 层：LLM 在代码中写 `# pip install xxx` → CodeExecutor 自动预安装
  第 2 层：post_tools 检测到 ModuleNotFoundError → auto_install_skill_deps() 批量安装
  第 3 层：从代码 import 语句解析包名 → auto_install_from_code() 兜底安装

架构关系：
  agent.py (MedicalSkillAgent)
      ↓ init_retriever() 注入依赖
  tools.py (本模块)
      ↓ 调用
  code_executor.py (代码执行器) + skill_retriever.py (语义检索)
"""
import re
import threading
from pathlib import Path
from langchain_core.tools import tool
from src.code_executor import CodeExecutor
from src.skill_retriever import SkillRetriever
from src.skill_context import SkillContext
from src.skill_stats import SkillStatsTracker
from src.middleware.tool_param_adapter import (
    normalize_tool_kwargs as _normalize_tool_kwargs,
    apply_aliases as _apply_tool_aliases,
)

# ── 技能目录根路径（所有 869 个技能的顶层目录）──
SKILLS_ROOT = Path("./skills")

# ── 代码执行器实例（启用虚拟环境隔离）──
# 每个 session 拥有独立的 venv，跨轮复用，程序退出时统一清理
_executor = CodeExecutor(use_venv=True)

# ── 模块级运行时依赖（由 agent.py 的 init_retriever() 注入）──
# 工具函数本身是无状态的，通过这些全局变量访问检索器、上下文组装器和统计追踪器
_retriever: SkillRetriever | None = None   # 技能语义检索器
_ctx: SkillContext | None = None            # 技能上下文组装器（加载 SKILL.md 摘要）
_stats: SkillStatsTracker | None = None     # 技能使用统计追踪器
_top_k: int = 8                             # 检索返回的最大技能数（默认 8）

_session_local = threading.local()            # 线程局部存储：session_id（支持并发 benchmark）


def set_session_id(session_id: str):
    """由 main.py 调用，设置当前会话 ID（用于 venv 隔离）。线程安全。"""
    _session_local.session_id = session_id


def get_session_id() -> str:
    """获取当前线程的 session_id。线程安全。"""
    return getattr(_session_local, "session_id", "")


def cleanup_session_venv(session_id: str):
    """清理指定 session 的 venv 和 conda env（benchmark 任务完成后调用）

    Args:
        session_id: 要清理的会话 ID（与 thread_id 一致）
    """
    _executor.cleanup_session(session_id)


def get_workspace_dir() -> Path:
    """返回 workspace 目录路径（供 benchmark 清理输出文件用）"""
    return _executor.workspace


def auto_install_package(pkg_name: str) -> bool:
    """post_tools_node 检测到 ModuleNotFoundError 后自动补装单个包

    在隔离 venv 中安装缺失的包，LLM 无需手动写 pip install 代码。
    安装成功后只需重试原来的 import 即可。

    注意：优先使用 auto_install_skill_deps() 批量安装整个技能的 requirements.txt，
    本函数仅作为兜底（包名不在 requirements.txt 中的情况）。
    """
    sid = get_session_id()
    if not sid:
        return False
    return _executor.install_package(sid, pkg_name)


def auto_install_skill_deps(skill_id: str) -> dict:
    """批量安装技能目录下 requirements.txt 的所有依赖

    第2层防御（兜底）：当第1层主动安装未覆盖（LLM 未填 related_skill）时，
    post_tools_node 检测到 ModuleNotFoundError 后调用此方法，
    一次性安装整个技能的所有依赖包。

    Args:
        skill_id: 技能 ID（目录名），如 "scrna-qc"

    Returns:
        {"installed": [...], "failed": [...], "skipped": [...]}
    """
    sid = get_session_id()
    if not sid or not skill_id:
        return {"installed": [], "failed": [], "skipped": []}
    return _executor.install_skill_dependencies(sid, skill_id)


def parse_imports_from_code(code: str) -> list[str]:
    """从代码中解析所有 import 语句，提取包名（兜底方案）

    当 selected_skill 为空且出现 ModuleNotFoundError 时，
    从代码中提取所有顶层 import 的包名，批量安装。

    Args:
        code: 完整的 Python 代码

    Returns:
        包名列表（去重，含别名映射）
    """
    import re as _re
    packages = []
    # 匹配 import xxx 或 import xxx as yyy
    for m in _re.finditer(r'^\s*import\s+(\w+)', code, _re.MULTILINE):
        pkg = m.group(1)
        if pkg not in packages:
            packages.append(pkg)
    # 匹配 from xxx import yyy
    for m in _re.finditer(r'^\s*from\s+(\w+)', code, _re.MULTILINE):
        pkg = m.group(1)
        if pkg not in packages:
            packages.append(pkg)
    return packages


def auto_install_from_code(code: str) -> dict:
    """从代码中提取 import 并批量安装（兜底方案）

    Args:
        code: 完整的 Python 代码

    Returns:
        {"installed": [...], "failed": [...]}
    """
    sid = get_session_id()
    if not sid:
        return {"installed": [], "failed": []}
    packages = parse_imports_from_code(code)
    if not packages:
        return {"installed": [], "failed": []}
    return _executor.install_batch_packages(sid, packages)


def init_retriever(retriever: SkillRetriever, ctx: SkillContext, stats: SkillStatsTracker, top_k: int = 8):
    """由 MedicalSkillAgent.__init__() 调用，注入运行时依赖。

    工具函数是模块级的（非类方法），无法通过构造函数注入依赖，
    因此使用这个初始化函数在 Agent 启动时将检索器、上下文组装器、
    统计追踪器注入到模块全局变量中。
    """
    global _retriever, _ctx, _stats, _top_k
    _retriever = retriever
    _ctx = ctx
    _stats = stats
    _top_k = top_k


# ══════════════════════════════════════════════════════════
#  工具 1/4：retrieve_skills — LLM 主动检索技能
# ══════════════════════════════════════════════════════════

@tool
def retrieve_skills(
    query: str = "",
    # ── 常见别名（兜底）──
    q: str = "",
    search: str = "",
    keyword: str = "",
    keywords: str = "",
    text: str = "",
) -> str:
    """当用户问题涉及生物医学专业技能、分析方法、实验设计时，调用此工具检索相关知识库。

    使用场景：用户问到基因分析、序列比对、CRISPR、放射学、生物信息学工具等专业问题时，
    先调用此工具获取匹配的技能摘要，再根据需要调用 read_file 获取完整文档。

    不需要调用的场景：简单打招呼、闲聊、问你是谁、感谢等日常对话。

    Args:
        query: 用于检索的查询文本，建议用用户问题中的关键词组合。

    Returns:
        匹配的技能摘要（技能名称 + 描述 + 文件清单），不含 SKILL.md 全文。
        如果需要某个技能的完整文档，请调用 read_file(skill_id, "SKILL.md")。
        如果没有匹配到相关技能，会返回提示信息。
    """
    # ── 别名归一化（防止 LLM 用 q/search/keyword 替代 query）──
    if not query:
        for alias in (q, search, keyword, keywords, text):
            if alias:
                query = alias
                break
    if not query:
        return "[错误] 缺少必填参数: query（或其别名 q/search/keyword/keywords/text）。请提供检索关键词。"

    if _retriever is None or _ctx is None or _stats is None:
        return "[错误] 检索引擎未初始化。"

    # 先做语义召回，再把每个技能的摘要上下文拼给 LLM。
    skills = _retriever.retrieve(query, k=_top_k)

    # 通用 skill ID 兜底：查询文本中包含完整技能 ID 时，强制加入结果
    skill_ids_in_result = {s["id"] for s in skills}
    query_lc = query.lower()
    for skill_id in _retriever.registry_by_id:
        if skill_id in skill_ids_in_result:
            continue
        if skill_id in query_lc:
            forced = _retriever.get(skill_id)
            if forced:
                skills.insert(0, {
                    "id": forced["id"],
                    "description": forced.get("description", ""),
                    "path": forced.get("path", ""),
                })
            break

    if not skills:
        return "未检索到与您问题匹配的技能。如果这是专业领域问题，请尝试用更具体的关键词重新提问；如果是日常对话，请直接回答。"

    skill_ids = [skill["id"] for skill in skills]
    _stats.record_retrieval(skill_ids)

    # ── 完整注入：返回 SKILL.md 全文 + 文件清单
    # 确保Agent始终获得完整代码示例和API指导，避免凭记忆写代码
    ctx_parts = []
    for skill in skills:
        skeleton = _ctx.load_skeleton(skill["id"])
        ctx_parts.append(f"## 技能: {skill['id']}\n{skeleton}")

    context = "\n\n".join(ctx_parts)
    return (f"已检索到 {len(skills)} 个相关技能：{', '.join(skill_ids)}\n\n"
            f"{'=' * 60}\n\n{context}\n\n"
            f"{'=' * 60}\n\n"
            f"⚠️ **重要提示**：\n"
            f"1. 上述技能已为你预加载完整文档（含 API 地址、代码示例），请严格遵循\n"
            f"2. **禁止凭记忆直接编写 API URL 或端点路径**——必须从上方 SKILL.md 文档获取\n"
            f"3. 封装脚本（如 tu.tools.xxx）必须优先使用，不要自己从零实现\n"
            f"4. 代码示例中的参数（如 limit、pageSize）不要随意修改")


# ══════════════════════════════════════════════════════════
#  辅助函数：列出技能目录下的可读文件
# ══════════════════════════════════════════════════════════

def _list_available_files(skill_id: str) -> list[str]:
    """扫描技能目录，返回所有可读文件的相对路径列表（含 SKILL.md）。"""
    skill_dir = SKILLS_ROOT / skill_id
    if not skill_dir.is_dir():
        return []
    available = []
    # SKILL.md 始终列在首位
    if (skill_dir / "SKILL.md").exists():
        available.append("SKILL.md")
    for p in sorted(skill_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.name == "SKILL.md":
            continue
        rel = p.relative_to(skill_dir).as_posix()
        available.append(rel)
    return available


# ══════════════════════════════════════════════════════════
#  工具 2/4：read_file — 深入阅读技能文档
# ══════════════════════════════════════════════════════════

@tool
def read_file(
    file_path: str = "",
    skill_id: str = "",
    limit: int = 300,
    offset: int = 1,
    # ── 常见别名（兜底）──
    path: str = "",
    filepath: str = "",
    filename: str = "",
    file: str = "",
    skill: str = "",
    name: str = "",
    skillId: str = "",
    max_lines: int = -1,
    start_line: int = -1,
) -> str:
    """读取指定技能目录下的参考文件。

    使用场景：技能文档的文件清单中列出了可用的 examples/、references/、scripts/ 文件，
    你觉得某个文件对当前任务有帮助时，调用此工具获取其完整内容。

    Args:
        skill_id: 技能 ID（目录名），如 "bio-blast-searches"
        file_path: 文件在技能目录内的相对路径，如 "examples/basic_blast.py" 或 "references/basic_inference.md"
        limit: 读取的行数上限，默认 300。避免大文件撑爆上下文。
        offset: 从哪一行开始读取，默认 1（第一行）。

    Returns:
        文件内容（限制行数）。
        如果文件不存在或路径越权，返回错误信息。
    """
    # ── 别名归一化 ──
    if not file_path:
        for alias in (path, filepath, filename, file):
            if alias:
                file_path = alias
                break
    if not file_path:
        return "[错误] 缺少必填参数: file_path（或其别名 path/filepath/filename）。"
    if not skill_id:
        for alias in (skill, name, skillId):
            if alias:
                skill_id = alias
                break
    if not skill_id:
        return "[错误] 缺少必填参数: skill_id（或其别名 skill/name）。"
    if max_lines is not None and max_lines >= 0:
        limit = max_lines
    if start_line is not None and start_line >= 0:
        offset = start_line
    full_path = (SKILLS_ROOT / skill_id / file_path).resolve()
    # 防止通过 ../../ 之类的路径穿越访问技能目录之外的文件。
    if not str(full_path).startswith(str(SKILLS_ROOT.resolve())):
        return "[错误] 路径越权，拒绝访问。"
    if not full_path.exists():
        # 文件不存在时直接返回该技能下真实可读的文件列表，帮助模型纠正路径。
        available = _list_available_files(skill_id)
        if available:
            file_list = "\n".join(f"  - {f}" for f in available)
            return (f"[错误] 文件不存在: {file_path}。\n"
                    f"该技能目录下可用的文件：\n{file_list}\n"
                    f"请从上述列表中选择文件路径。")
        else:
            return (f"[错误] 文件不存在: {file_path}。\n"
                    f"该技能目录下没有额外的可读文件（仅有 SKILL.md）。"
                    f"SKILL.md 中引用的外部应用文件路径（如 src/lib/..., docs/..., @/lib/...）"
                    f"不属于本技能目录，无法通过 read_file 读取。")
    if full_path.is_dir():
        # 路径是目录时，返回该目录下的文件列表，而非抛出 Permission denied。
        available = []
        for p in sorted(full_path.rglob("*")):
            if p.is_file() and p.name != "SKILL.md":
                rel = p.relative_to(full_path).as_posix()
                available.append(rel)
        if available:
            file_list = "\n".join(f"  - {f}" for f in available)
            return (f"[提示] '{file_path}' 是一个目录，不能直接读取。该目录下的文件：\n"
                    f"{file_list}\n"
                    f"请选择具体文件路径后再调用 read_file。")
        else:
            return f"[提示] '{file_path}' 是一个空目录，没有可读文件。"
    try:
        content = full_path.read_text(encoding="utf-8")
        lines = content.split("\n")
        start_idx = max(0, offset - 1)
        end_idx = start_idx + limit
        
        result_lines = lines[start_idx:end_idx]
        result = "\n".join(result_lines)
        
        if end_idx < len(lines):
            result += f"\n\n... (截断，文件剩余 {len(lines) - end_idx} 行未读取，请使用 offset={end_idx + 1} 继续读取)"
        elif start_idx > 0:
            result = f"... (文件前 {start_idx} 行已略过)\n\n" + result

        # ── 当读取的是 SKILL.md 时，附加依赖安装指引 ──
        # 方案 C：让 LLM 自己决定依赖并在代码中写入 pip install，
        # 由执行器的 _extract_pip_installs 机制自动预安装。
        is_skill_md = (file_path == "SKILL.md" or file_path.endswith("/SKILL.md"))
        is_last_page = (end_idx >= len(lines))
        if is_skill_md and is_last_page:
            result += (

                "\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "📋 **依赖安装提醒**：\n"
                "如果你需要基于此技能执行代码，请务必在 execute_code 的代码中\n"
                "显式写入所需的 pip install 命令。系统会自动提取并预安装这些包。\n\n"
                "写法示例（在代码开头用注释行标明）：\n"
                "```python\n"
                "# pip install pydeseq2 pandas numpy scipy scikit-learn\n"
                "import pydeseq2\n"
                "import pandas as pd\n"
                "...\n"
                "```\n\n"
                "关键规则：\n"
                "1. 在代码第一行用 `# pip install 包1 包2 ...` 列出所有需要安装的包\n"
                "2. 系统会在执行代码前自动预安装这些包到隔离的虚拟环境\n"
                "3. 请根据上方 SKILL.md 中的依赖/安装/前置条件章节来确定需要安装哪些包\n"
                "4. 仅列出非标准库的第三方包（如 numpy, scanpy, pydeseq2），无需列出标准库（如 os, json, sys）\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )

        return result
    except Exception as e:
        return f"[错误] 读取失败: {e}"

# ══════════════════════════════════════════════════════════
#  工具 3/4：update_task_status — 多步骤任务状态追踪
# ══════════════════════════════════════════════════════════

@tool
def update_task_status(
    status: str = "pending",
    step_number: int = -1,
    result_summary: str = "",
    task_index: int = -1,
    # ── 常见别名（让 LLM 即使用别名也能通过 Pydantic 校验）──
    task_id: int = -1,
    step_id: int = -1,
    step: int = -1,
    step_index: int = -1,
    step_idx: int = -1,
    step_idx_0: int = -1,
    stepNumber: int = -1,
    taskId: int = -1,
    index: int = -1,
    idx: int = -1,
    n: int = -1,
    num: int = -1,
    number: int = -1,
    resultSummary: str = "",
) -> str:
    """更新多步骤任务中特定步骤的状态。

    Args:
        status: 新状态，只能是 "done", "in_progress", "failed"
        step_number: 步骤编号（从1开始，对应执行计划中的步骤序号。如第1步填1，第2步填2）
        result_summary: 该步骤执行结果的简短总结
        task_index: 0-indexed 步骤编号（兼容旧调用，与 step_number 二选一）
        task_id / step_id / step / step_index / step_idx / index / idx / n / num / number:
            上述都是 step_number 的常见别名，LLM 误用时函数内部会自动归一化。

    Returns:
        状态更新结果和下一步建议。
    """
    # ── 别名归一化：把 LLM 用错名的字段映射回 step_number ──
    # 优先级：显式 step_number > task_index（+1）> 其他别名
    aliases_provided = {
        "task_index": task_index,
        "task_id": task_id,
        "step_id": step_id,
        "step": step,
        "step_index": step_index,
        "step_idx": step_idx,
        "step_idx_0": step_idx_0,
        "stepNumber": stepNumber,
        "taskId": taskId,
        "index": index,
        "idx": idx,
        "n": n,
        "num": num,
        "number": number,
    }
    if step_number < 1:
        for name, val in aliases_provided.items():
            if val is not None and val >= 0:
                if name == "step_idx_0" or name == "step_index" or name == "task_index":
                    # 0-indexed → 转 1-indexed
                    step_number = val + 1
                else:
                    step_number = val
                break
    if step_number < 1:
        step_number = 1

    # resultSummary 兜底
    if not result_summary and resultSummary:
        result_summary = resultSummary

    # 规范化 status（防止 LLM 写 "completed"/"finish" 等近义词）
    status_norm = (status or "").strip().lower()
    if status_norm in ("completed", "complete", "finish", "finished", "ok", "success", "successful", "succeeded", "done!"):
        status = "done"
    elif status_norm in ("in-progress", "inprogress", "running", "processing", "working", "start", "started"):
        status = "in_progress"
    elif status_norm in ("fail", "failure", "error", "errored", "broken", "crashed"):
        status = "failed"
    elif status_norm in ("pending", "todo", "wait", "waiting", "queued", "queue"):
        status = "pending"

    return f"步骤 {step_number} 已标记为 {status}。\n总结: {result_summary}\n请继续执行下一个 pending 的步骤。"

# ══════════════════════════════════════════════════════════
#  工具 4/4：execute_code — 在隔离环境中执行代码
# ══════════════════════════════════════════════════════════

@tool
def execute_code(
    code: str = "",
    language: str = "python",
    related_skill: str = "",
    step_type: str = "analysis",
    # ── 常见别名（兜底）──
    script: str = "",
    src: str = "",
    source: str = "",
    codex: str = "",
    lang: str = "",
    prog: str = "",
    skill: str = "",
    skill_name: str = "",
    relatedSkill: str = "",
    stepType: str = "",
) -> str:
    """在受控工作目录中执行 Python / Bash / R / JavaScript 代码。

    执行流程：
    1. CodeExecutor 提取代码中的 `# pip install xxx` 注释行 → 自动预安装依赖
    2. 如果填写了 related_skill → 额外读取该技能的 requirements.txt 批量安装
    3. 在隔离的虚拟环境（venv）中执行代码
    4. 构建结构化输出：[包安装状态] + [执行结果] + [错误修正建议]

    安全控制（均在 CodeExecutor 内部实现）：
    - AST 静态分析：拦截 os.system()、subprocess 等危险调用
    - 文件访问限制：仅允许 ./workspace 目录
    - 超时保护：防止死循环

    Args:
        code: 完整的可运行代码，必须含所有 import、数据定义、输出打印。
        language: "python"、"bash"、"r" 或 "javascript"。默认 "python"。
        related_skill: 当前使用的技能 ID（如 "scrna-qc"），用于自动安装依赖。
        step_type: 步骤性质标记（preparation/analysis/final），帮助系统判断是否继续执行。

    Returns:
        结构化执行结果：[包安装状态] + [执行成功/失败] + stdout/stderr + 修正建议。
    """
    # ── 别名归一化 ──
    if not code:
        for alias in (script, src, source, codex):
            if alias:
                code = alias
                break
    if not code:
        return "[执行失败] 缺少必填参数: code（或其别名 script/src/source）。请在调用时提供要执行的完整代码。"
    if not language or language == "python":
        for alias in (lang, prog):
            if alias:
                language = alias
                break
    if not related_skill:
        for alias in (skill, skill_name, relatedSkill):
            if alias:
                related_skill = alias
                break
    if not step_type or step_type == "analysis":
        if stepType:
            step_type = stepType
    # 真正的安全控制与执行细节都在 CodeExecutor 内部完成。
    # 使用流式执行，实时将代码输出记录到日志
    result = _executor.execute_streaming(language, code, skill_id=related_skill,
                                          session_id=get_session_id())

    # ── 构建结构化输出（供 LLM 和 post_tools_node 解析）──
    # 输出格式：[包安装状态] + [执行成功/失败] + stdout/stderr + 修正建议
    output_parts = []

    # ── 部分 1：pip install 结果追踪 ──
    # 展示依赖安装的方式和结果（venv/conda/系统安装/安装失败）
    pip_result = result.get("pip_result")
    if pip_result:
        if pip_result.get("method") == "venv":
            output_parts.append(f"[包安装-虚拟环境] {pip_result.get('status', '')}")
        elif pip_result.get("method") == "conda_env":
            conda_pkgs = pip_result.get("conda_packages", [])
            pip_pkgs = pip_result.get("pip_packages", [])
            parts = []
            if conda_pkgs:
                parts.append(f"conda 安装: {', '.join(conda_pkgs)}")
            if pip_pkgs:
                parts.append(f"pip 安装: {', '.join(pip_pkgs)}")
            output_parts.append(
                f"[包安装-conda环境] {pip_result.get('status', '')} ({'; '.join(parts) if parts else '无包'})"
            )
        elif pip_result.get("method") == "venv_failed":
            output_parts.append(f"[包安装-虚拟环境失败] {pip_result.get('status', '')}，降级到系统Python")
        elif pip_result.get("method") == "system":
            installed = pip_result.get("installed", [])
            failed = pip_result.get("failed", [])
            if installed:
                output_parts.append(f"[包安装成功] {', '.join(installed)}")
            if failed:
                failed_names = [f["package"] for f in failed]
                output_parts.append(f"[包安装失败] {', '.join(failed_names)}")

    # ── 部分 2：代码执行结果 ──
    if result["success"]:
        # ── 检测脚本内部错误（returncode=0 但 stdout/stderr 中有错误标记）──
        # 某些脚本用 try/except 捕获异常后仅 print 错误而不 sys.exit(1)，
        # 导致 returncode=0 但实际执行失败。通过正则扫描输出中的错误模式来修正。
        combined_output = result.get("stdout", "") + result.get("stderr", "")
        internal_error_patterns = [
            r"Error during (training|prediction|evaluation|execution|fitting|loading|processing)",
            r"AttributeError:",
            r"ValueError:",
            r"TypeError:",
            r"RuntimeError:",
            r"unsupported format string",
            r"only \d+-dimensional arrays can be converted",
            r"has no attribute",
            r"cannot be multiplied",  # 矩阵维度错误
            r"Traceback \(most recent call last\)",  # 通用 Python 异常追踪
        ]
        has_internal_error = any(
            re.search(p, combined_output) for p in internal_error_patterns
        )

        if has_internal_error:
            # 脚本内部报错但 returncode=0，按失败处理
            output_parts.append("[执行失败-脚本内部错误]")
            if result.get("stdout"):
                output_parts.append(result["stdout"])
            if result.get("stderr"):
                output_parts.append("\nSTDERR:")
                output_parts.append(result["stderr"])
            output_parts.append("\n请分析错误原因并修正代码后重试。如果多次尝试相同报错，请停止重试并直接回复用户失败原因。")
        else:
            stdout_text = result.get("stdout", "")
            stderr_text = result.get("stderr", "")
            # ── 从 stderr 中提取 API 响应追踪信息 ──
            # code_executor 注入的 [API] 和 [API-Body] 行记录了所有 HTTP 请求
            # 提取出来展示给 LLM，让 LLM 看到实际 API 返回了什么
            api_trace_lines = []
            for _line in stderr_text.splitlines():
                if _line.startswith("[API]") or _line.startswith("[API-Body]"):
                    api_trace_lines.append(_line)

            # ── 成功但输出为空/无意义 → 强警告，防止 LLM 疯狂重试 ──
            if not stdout_text or len(stdout_text.strip()) < 10:
                output_parts.append("[执行成功-输出为空]")
                if stdout_text.strip():
                    output_parts.append(stdout_text)
                else:
                    output_parts.append("（代码执行完成但未产生任何输出）")
                # 展示 API 追踪信息，帮助诊断空输出原因
                if api_trace_lines:
                    output_parts.append("\n\n📡 API 调用追踪（代码调用了 API 但未 print 结果）：")
                    for _tl in api_trace_lines[:10]:  # 最多展示 10 行
                        output_parts.append(f"  {_tl}")
                output_parts.append(
                    "\n\n⚠️ 重要：代码执行成功但没有任何有效输出，通常说明 API 返回了空数据或代码逻辑有问题。"
                    "\n请不要继续重试相同或相似的代码！建议："
                    "\n  1. 检查代码逻辑和 API 端点是否正确"
                    "\n  2. 换用模拟数据或本地知识库完成分析"
                    "\n  3. 直接基于已有知识给出结论性回复"
                )
            else:
                output_parts.append("[执行成功]")
                output_parts.append(stdout_text)
                # 即使有输出，也附加 API 追踪信息（帮助用户/日志定位问题）
                if api_trace_lines:
                    output_parts.append("\n\n📡 API 调用追踪：")
                    for _tl in api_trace_lines[:5]:  # 成功时展示前 5 行
                        output_parts.append(f"  {_tl}")

            # API 调用警告（代码可能捕获了 API 异常但未崩溃）
            api_error = result.get("api_error")
            if api_error:
                output_parts.append(f"\n{api_error}")
    else:
        # API 调用细粒度追踪
        api_error = result.get("api_error")
        error_msg = result.get("error", "")

        # ── 超时特殊处理：明确告知 LLM 换策略 ──
        if "超时" in error_msg or "timeout" in error_msg.lower():
            output_parts.append(f"[执行失败-超时]")
            output_parts.append(f"代码执行超时 ({error_msg})，通常意味着外部 API 无响应或网络不可达。")
            output_parts.append(f"\n⚠️ 重要：请不要重试相同的 API 调用！建议：")
            output_parts.append(f"  1. 使用模拟数据/本地知识库完成分析")
            output_parts.append(f"  2. 调用其他可用的 API 作为替代数据源")
            output_parts.append(f"  3. 直接基于已有知识给出结论性回复")
        elif api_error:
            output_parts.append(f"[执行失败-外部API错误]\n{api_error}")
            output_parts.append(f"详细错误:\n{result.get('stderr') or error_msg}")
            output_parts.append("\n请分析原因并修正代码后重试。如果多次尝试相同报错，请停止重试并直接回复用户失败原因。")
        else:
            output_parts.append("[执行失败]")
            error = result.get("stderr") or error_msg
            output_parts.append(error)
            output_parts.append("\n请分析原因并修正代码后重试。如果多次尝试相同报错，请停止重试并直接回复用户失败原因。")

    return "\n".join(output_parts)
