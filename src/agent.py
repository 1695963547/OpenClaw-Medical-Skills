"""
LangGraph ReAct Agent 核心模块 — 医疗 AI 智能体引擎
======================================================

本模块实现了基于 LangGraph 的 ReAct（推理-行动）循环智能体，
是整个系统的"大脑"，负责协调 5 个核心节点的运转。

┌─────────────────────────────────────────────────────────────┐
│                    LangGraph 有向图流程                       │
│                                                             │
│  ┌──────────────┐    ┌──────────┐    ┌──────────┐          │
│  │ auto_retrieve │───→│  agent   │───→│  tools   │          │
│  │  (自动检索)   │    │ (LLM决策) │    │ (工具执行) │          │
│  └──────────────┘    └──────────┘    └──────────┘          │
│                           ↑    │           │                │
│                           │    ↓           ↓                │
│                      ┌──────────┐    ┌────────────┐        │
│                      │ planner  │←───│ post_tools  │        │
│                      │ (计划生成) │    │ (后处理修正) │        │
│                      └──────────┘    └────────────┘        │
│                           │                                 │
│                           ↓                                 │
│                      agent (循环) 或 END                     │
└─────────────────────────────────────────────────────────────┘

5 个核心节点职责：
  1. auto_retrieve_node  — 系统级自动检索，用 ChromaDB 语义匹配预加载技能
  2. agent_node          — LLM 决策核心，决定调用工具还是直接回复
  3. tools (ToolNode)    — 执行 LLM 请求的工具调用
  4. post_tools_node     — 解析执行结果、分类错误、注入修正建议、Stuck 检测
  5. planner_node        — 为多步骤任务生成结构化执行计划

关键设计特性：
  - MemorySaver 多轮对话持久化（同一 thread_id 跨轮记忆）
  - Stuck Detector 循环检测 + 软着陆（force_no_tools）
  - 动态迭代上限（Planner N 步 → max(20, N*4)）
  - 结构化错误分类（ErrorType 枚举，13 种错误类型）
  - 3 层依赖自动安装防御
"""
import logging
import os
import re
import hashlib
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END            # LangGraph 图构建框架
from langgraph.graph.message import add_messages        # 消息累加器（Annotated 用）
from langgraph.prebuilt import ToolNode                 # 预构建的工具执行节点
from langchain_openai import ChatOpenAI                 # LLM 接口（兼容 OpenAI API）
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage  # 消息类型

try:
    from langgraph.checkpoint.memory import MemorySaver  # 内存级状态持久化（多轮对话）
except ImportError:
    MemorySaver = None  # type: ignore

from src.skill_retriever import SkillRetriever    # 技能语义检索器
from src.skill_context import SkillContext         # 技能上下文组装器
from src.tools import retrieve_skills, read_file, execute_code, init_retriever  # 4 个工具函数
from src.skill_stats import SkillStatsTracker      # 技能使用统计追踪
from src.middleware.error_memory import (
    ErrorMemory,
    StrategyEscalator,
    get_error_memory,
    get_escalator,
    set_memory_session_id,
    cleanup_memory,
)

logger = logging.getLogger("agent")


class ErrorType:
    """结构化错误类型枚举（替代脆弱的字符串匹配）"""
    SUCCESS = "success"
    # 脚本执行错误
    SCRIPT_ERROR = "script_error"  # 脚本内部异常
    IMPORT_ERROR = "import_error"  # 缺少模块
    PACKAGE_INSTALL_FAILED = "package_install_failed"  # pip/conda 安装失败
    # API 错误（细粒度）
    API_AUTH_FAILED = "api_auth_failed"  # 认证失败
    API_RATE_LIMIT = "api_rate_limit"  # 限流
    API_NETWORK = "api_network_error"  # 网络错误
    API_RESOURCE_NOT_FOUND = "api_resource_not_found"  # 资源不存在
    API_SERVER_ERROR = "api_server_error"  # 服务端 5xx
    API_GENERIC = "api_generic_error"  # 未分类的 API 错误
    # 其他
    TIMEOUT = "timeout"
    FILE_NOT_FOUND = "file_not_found"
    UNKNOWN_FAILURE = "unknown_failure"


def _classify_error(content: str) -> str:
    """从工具输出内容中分类错误类型（结构化，不依赖中文字符串）
    
    优先匹配结构化标记（[执行失败-xxx]），再用关键词兜底。
    """
    # 1. 优先匹配 tools.py 输出的结构化标记
    if "[执行成功]" in content:
        # 成功时检查是否有 API 警告
        if "[执行失败-外部API错误]" in content:
            return ErrorType.API_GENERIC
        return ErrorType.SUCCESS
    
    if "[执行失败-脚本内部错误]" in content:
        return ErrorType.SCRIPT_ERROR
    
    if "[执行失败-外部API错误]" in content:
        # 细粒度 API 错误分类
        content_lower = content.lower()
        if "认证失败" in content or "api key" in content_lower or "401" in content or "unauthorized" in content_lower:
            return ErrorType.API_AUTH_FAILED
        if "限流" in content or "rate limit" in content_lower or "429" in content:
            return ErrorType.API_RATE_LIMIT
        if "资源不存在" in content or "not found" in content_lower or "404" in content:
            return ErrorType.API_RESOURCE_NOT_FOUND
        if "服务端错误" in content or "500" in content or "502" in content or "503" in content:
            return ErrorType.API_SERVER_ERROR
        if "网络错误" in content or "connection" in content_lower:
            return ErrorType.API_NETWORK
        return ErrorType.API_GENERIC
    
    # 2. 关键词兜底（处理非标准输出）
    if "ImportError" in content or "ModuleNotFoundError" in content:
        return ErrorType.IMPORT_ERROR
    
    if "pip install" in content and ("error" in content.lower() or "failed" in content.lower() or "失败" in content):
        return ErrorType.PACKAGE_INSTALL_FAILED
    
    if "Timeout" in content or "timeout" in content or "超时" in content:
        return ErrorType.TIMEOUT
    
    if "FileNotFoundError" in content or "No such file" in content:
        return ErrorType.FILE_NOT_FOUND
    
    # 网络错误特征
    if "10060" in content or "urlopen error" in content or "Connection refused" in content:
        return ErrorType.API_NETWORK
    
    # 默认失败
    if "[执行失败" in content or "Error" in content or "error" in content:
        return ErrorType.UNKNOWN_FAILURE
    
    return ErrorType.SUCCESS


def _normalize_code_for_stuck_detect(code: str) -> str:
    """归一化代码用于 stuck 检测：去注释、去空白行、去首尾空白

    Stuck Detector 比较连续两轮 execute_code 的 code 参数时，
    如果 LLM 只是改了注释或调整了缩进，代码实质相同，
    简单的字符串比较会漏检。归一化后能更准确地检测循环。
    """
    # 去掉单行注释（# 开头的整行或行尾注释）
    code = re.sub(r'#[^\n]*', '', code)
    # 去掉多行字符串/注释（"""...""" 或 '''...'''）
    code = re.sub(r'("{3}|\'{3})[\s\S]*?\1', '', code)
    # 去掉空白行，每行去首尾空白
    lines = [line.strip() for line in code.split('\n') if line.strip()]
    return '\n'.join(lines)

# ══════════════════════════════════════════════════════════
#  Agent 状态定义（LangGraph 有向图的共享状态结构）
# ══════════════════════════════════════════════════════════

class AgentState(TypedDict):
    """LangGraph 图的共享状态——所有节点通过读写此状态进行通信。

    每次节点执行后返回一个 dict，LangGraph 会自动合并到此状态中。
    messages 字段使用 add_messages 注解，实现消息累加（而非覆盖）。
    """
    messages: Annotated[list, add_messages]  # 对话消息列表（核心：LLM 和工具的交互记录）
    selected_skill: str                      # 当前选中的技能 ID（用于依赖安装和统计）
    tool_status: str                         # 工具执行状态（成功/失败/超时/外部API失败等）
    iteration: int                           # 当前迭代轮次（每次 agent_node 决策 +1）
    consecutive_same_result: int             # Stuck 检测：execute_code 连续传入相同代码的次数
    consecutive_empty_result: int            # Stuck 检测：execute_code 连续输出为空的次数
    last_execute_code: str                   # Stuck 检测：上一次 execute_code 的代码（归一化后）
    force_no_tools: bool                     # 软着陆标志：Stuck 触发后禁用工具，强制 LLM 回复
    plan: str                                # Planner 生成的执行计划文本
    subtasks: list[dict]                     # 多步骤子任务状态追踪（description + status）
    retrieved_skills: list[str]              # auto_retrieve 节点召回的技能 ID 列表
    auto_resume_count: int                   # 防中途放弃：自动续跑次数（最多 2 次）
    api_domain_counts: dict                  # 同域 API 重复调用计数：{"clinicaltrials.gov": 3, ...}
    recent_code_window: list                  # 周期循环检测：最近 8 次 execute_code 的归一化 MD5 指纹

SYSTEM_PROMPT = """你是一个医疗 AI 助手，可以访问 869 个生物医学技能的知识库。

## ⚠️ 核心规则：优先使用系统预加载的技能

系统会在每轮对话开始时**自动为你检索**相关技能并预加载到上下文中（见下方系统消息）。
你必须遵循以下优先级：
1. **预加载技能 > 内部知识**：如果系统已预加载了相关技能，必须优先使用它们
2. **SKILL.md > 记忆中的 URL**：训练数据中的 API 地址、端点路径、库版本可能已过时，
   必须通过 read_file 从 SKILL.md 获取最新信息
3. **封装脚本 > 手写代码**：如果技能目录提供了封装脚本（如 fda_query.py、
   blast_wrapper.py），优先使用而非自己从零实现
4. 只有在预加载的技能不相关或不足以解决问题时，才使用你的内部知识

## 工具使用策略

你有四个工具可用，请按需选择：

### retrieve_skills — 技能检索（最先考虑）
- **何时调用**：用户问题涉及生物医学专业领域（基因分析、序列比对、CRISPR、放射学、实验设计等），需要专业知识或可执行方案时
- **何时不调用**：日常打招呼、闲聊、感谢、问你是谁等与专业知识无关的对话
- **调用后**：你会收到匹配的技能摘要（名称+描述+文件清单），如需完整文档请调用 `read_file(skill_id, "SKILL.md")`

### read_file — 深入阅读
- retrieve_skills 返回的文件清单中有你需要的细节时，调用此工具获取完整文件内容
- 不要一次性读完所有文件，先判断相关性再按需读取
- **重要**：SKILL.md 中的代码示例可能引用外部应用文件（如 `src/lib/...`、`docs/...`、`@/lib/...`），
  这些路径属于原应用的代码结构，不是技能目录中的文件，不能用 read_file 读取
- 只能读取"可用文件清单"中明确列出的文件路径

### execute_code — 执行分析
- 需要运行代码时调用（BLAST、序列比对、差异表达等）
- 代码必须完整可运行，含所有 import 和数据定义
- **critical: 代码第一行必须写 `# pip install 包1 包2 ...` 声明所有第三方依赖，系统会自动预安装。依赖列表请从 SKILL.md 的"依赖/安装/前置条件"章节获取。**
- **可选**: 填入 related_skill 参数（技能 ID），系统会额外尝试安装该技能 requirements.txt 中的依赖（如果有的话），作为双重保障

### update_task_status — 任务状态追踪 (必选多步任务)
- **何时调用**：在执行多步骤分析计划时，每当你完成了一个步骤，必须调用此工具更新任务状态。
- **强制约束**：在你的 `plan` 中有多个子任务时，你必须依次标记它们为 `done`。未完成所有前置节点无法结束当前回复。

## 推荐工作流

0. **（系统已完成）** 系统已自动为你检索了相关技能并预加载到上下文中（见上方系统消息）
1. 查看预加载的技能摘要，判断哪些最相关 → `read_file(skill_id, "SKILL.md")` 获取完整文档
2. 阅读完整文档，特别注意"依赖/安装/前置条件"章节，记录所需的第三方包 → 需要更多细节则继续 `read_file` 读取示例/参考文件
3. 综合信息后决定：直接回答 或 `execute_code` 执行分析。**若调用 execute_code，代码第一行必须写 `# pip install 包1 包2 ...` 列出所有第三方依赖**（如 `# pip install scanpy numpy scipy`），系统会在执行前自动预安装这些包。
4. 日常对话 → 直接回答，无需调用任何工具

## 执行策略

- 概念/方法论/规范类问题 → 基于文档直接组织答案
- 需要执行分析（BLAST、序列比对、差异表达等）→ 严格按照文档工作流生成完整脚本 → `execute_code`
- **多步骤任务必须连续执行完成**：每次 `execute_code` 执行后，必须自问"这个结果是否直接回答了用户的问题？"如果不是，继续调用工具执行后续步骤
  - 典型多步骤模式（禁止只完成第一步就停止）：
    - Local BLAST: makeblastdb 建库 → blastn/blastp 搜索 → 解析结果（三步缺一不可）
    - 差异表达: 质控 → 比对 → 计数 → DESeq2 分析
    - CRISPR: 序列获取 → gRNA 查找 → 脱靶评估
    - 变异调用: 比对 → 排序 → 变异检测 → 注释
  - **禁止在中间步骤（如创建输入文件、建库）后停止并报告"成功"**——这不是最终结果
- **优先使用 Python** (`execute_code(language="python", ...)`)，因为 bash 在 Windows 上可能不可用
- 如果技能文档提供了 `.sh` 脚本，先用 `read_file` 读取内容，然后用 Python 等价代码实现
- 如果需要运行 JavaScript/Node.js 代码，直接用 `execute_code(language="javascript")`，**不要**用 Python 三引号字符串包裹 JS 代码（极易引号冲突）
- 如果需要在 Python 中生成其他语言的文件，用 base64 编码写入：`import base64; open("file.js","wb").write(base64.b64decode("..."))`
- 如果技能文档提供了 Python 封装（如 `blast_wrapper.py`），优先参考并使用
- 即使用户要求"写代码并执行"，也应**先调用 `retrieve_skills`** 检查是否有相关技能（可能提供 API 封装、代码模板、或依赖预装）。
- 仅当 retrieve_skills 返回空或所有结果都明显不相关时，才直接写代码。
- 如果技能提供了 Python 封装脚本（如 `fda_query.py`），优先使用而非重复造轮子。
- 不要用 `execute_code` 做目录遍历、环境探测、搜索项目文件等无关操作

## 代码要求

- 完整可运行（含所有 import、数据定义、输出打印）
- 文件读写使用相对路径（工作目录为项目根目录）
- 输入文件（如技能数据、参考文件）使用项目根目录的相对路径，如 `skills/gwas-lookup/data/demo.json`
- 输出文件请写入 `./workspace/` 目录，如 `./workspace/results.csv`
- 调用已有脚本时使用文件清单中列出的完整路径
- 某个语言不可用时（Rscript/bash 未安装），用 Python 替代
- **注意**：新版 Biopython 已移除 `Bio.Alphabet`，请直接使用 `Seq` 对象，不要导入 `generic_dna` 等。

## ⚠️ 依赖安装机制（关键）

当你通过 read_file 读取了某个技能的 SKILL.md 并准备执行代码时，**必须**在代码中显式声明依赖安装：

**写法**：在代码的第一行用注释写出 pip install 命令，系统会自动提取并预安装：
```python
# pip install scanpy numpy scipy
import scanpy as sc
import numpy as np
...
```

**规则**：
1. **必须填写** `# pip install ...` 注释行，这是依赖预安装的唯一触发方式
2. 包名来自 SKILL.md 中的"依赖"/"安装"/"前置条件"/"Requirements"等章节
3. 仅需列出第三方包，无需列出 Python 标准库（os, json, sys, re 等）
4. 系统会在执行代码前自动安装这些包到隔离的虚拟环境，你无需写 subprocess 调用
5. 如果不确定需要哪些包，宁可多写也不要漏写——系统会跳过已安装的包

**示例**：
```python
# pip install pydeseq2 pandas numpy scipy scikit-learn anndata
from pydeseq2 import DESeq2
import pandas as pd
import numpy as np
...
```

## 数据处理健壮性（关键）

当处理来自多个 API 或异构数据源的结果时，必须遵循以下规则：

- **始终使用 `.get()` 方法**访问字典（如 `hit.get('trait', 'Unknown')`），避免 `[]` 直接取键导致 KeyError
- **先检查数据结构**再编写处理逻辑：用 `print(list(data.keys()))` 或 `print(json.dumps(data, indent=2)[:800])` 确认实际字段名
- **不同数据源字段名可能不同**：如有的用 `trait`，有的用 `phenostring`，有的用 `gene_symbol`，按数据源分别编写提取逻辑
- **优先调用技能目录下已有的封装脚本**（如 `gwas_lookup.py`、`fda_query.py`），而非重新实现处理逻辑
- 读取 demo/示例数据文件时（如 `skills/*/data/demo*.json`），先快速检查前几条数据的 keys() 以了解各 API 的数据 schema 差异

## 文件不存在处理

- 当用户要求的输入文件在 `./workspace` 中不存在时，**必须明确告知用户该文件不存在**，并建议可用的替代数据文件
- 不要在发现文件不存在后默默停止，也不要假装任务已完成


## 安全约束

- 文件读写仅限 `./workspace` 目录（工作目录），禁止访问其他目录
- 不允许删除文件、修改系统配置、安装系统级软件
- 只读分析类代码无需用户确认即可执行
- 如果需要安装 Python 包，**必须在代码第一行写 `# pip install 包1 包2 ...` 注释**，系统会自动提取并预安装，无需写 subprocess 调用
- **重要**：不要在最终回复中原样重复 `execute_code` 的长篇输出内容。用户的终端已经看到了执行日志，你只需总结关键结果、解释发现或提供下一步建议即可。
"""

# ChromaDB 余弦距离阈值：距离 < 0.7 视为相关（余弦相似度 ≥ 0.3）
AUTO_RETRIEVE_MAX_DISTANCE = 0.7

# ══════════════════════════════════════════════════════════
#  核心智能体类
# ══════════════════════════════════════════════════════════

class MedicalSkillAgent:
    """医疗 AI 智能体——基于 LangGraph 的 ReAct 循环引擎。

    核心职责：
    1. 构建 LangGraph 有向图（5 个节点 + 3 个条件路由）
    2. 管理 LLM 实例（支持工具绑定 + 无工具回退）
    3. 提供 stream() 和 run() 两种执行模式
    4. 动态构建系统提示词（注入运行环境 + 执行计划）

    使用方式：
        agent = MedicalSkillAgent(retriever, llm, max_iterations=20, top_k=8)
        for event in agent.stream("分析 RNA-seq 数据", thread_id="abc"):
            # 处理每个节点的输出事件
            ...
    """
    def __init__(
        self,
        retriever: SkillRetriever,
        llm: ChatOpenAI,
        max_iterations: int = 20,
        top_k: int | None = None,
    ):
        # 这几个对象分别负责：检索、技能上下文组装、运行统计。
        self.retriever = retriever
        self.ctx = SkillContext()
        self.stats = SkillStatsTracker()
        self._system_prompt_template = SYSTEM_PROMPT  # 缓存静态部分
        env_top_k = os.getenv("SKILL_TOP_K") or os.getenv("LLM_TOP_K")
        resolved_top_k = 8
        if isinstance(env_top_k, str) and env_top_k.strip():
            try:
                resolved_top_k = int(env_top_k.strip())
            except Exception:
                resolved_top_k = 8
        if top_k is not None:
            resolved_top_k = top_k
        self.top_k = resolved_top_k
        self.max_iterations = max_iterations  # 用户配置的硬上限，不再动态扩容

        # 工具函数定义在 src.tools 中，这里把运行时依赖注入进去。
        init_retriever(self.retriever, self.ctx, self.stats, self.top_k)

        from src.tools import update_task_status
        tools = [retrieve_skills, read_file, execute_code, update_task_status]
        self.llm_raw = llm
        self.tool_node = ToolNode(tools)

        disable_tools_env = (os.getenv("LLM_DISABLE_TOOLS") or "false").strip().lower()
        disable_tools = disable_tools_env in {"1", "true", "yes", "y"}

        if disable_tools:
            self.llm_with_tools = llm
        else:
            self.llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=True)
        self.graph = self._build_graph()

    # ─── 节点 ───

    def _build_system_prompt(self, plan: str = "", subtasks: list[dict] | None = None) -> str:
        """动态构建系统提示词，注入运行时环境信息（Claude Code 做法）

        LLM 每轮都能看到当前 OS、工作目录、workspace 绝对路径等事实，
        而不是靠猜测生成代码。这是跨平台路径问题的第一层防御。
        同时注入 Planner 生成的执行计划（如果有），指导多步骤任务。

        Args:
            plan: Planner 生成的执行计划文本
            subtasks: 子任务状态列表，由 agent_node 从 state 中传入
        """
        import platform as _platform
        from shutil import which as _which

        workspace_dir = os.path.abspath("./workspace")
        cwd = os.getcwd()
        shell = os.environ.get(
            "SHELL",
            "PowerShell" if _platform.system() == "Windows" else "bash",
        )

        # conda 可用性检测
        conda_available = _which("conda") is not None
        conda_info = ""
        if conda_available:
            conda_info = (
                "- **conda 可用**: 检测到 conda，pysam/pybedtools/tables/pybigwig 等 C 扩展包"
                "将自动走 conda 预编译路径（无需本地编译）\n"
            )
        else:
            conda_info = (
                "- **conda 不可用**: 上述 C 扩展包无法安装，请使用纯 Python 替代方案"
                "（如 Biopython 替代 pysam, h5py 替代 tables）\n"
            )

        env_info = f"""
## 当前运行环境（运行时动态注入，请严格遵守）

- 操作系统: {_platform.system()} ({_platform.release()})
- Shell: {shell}
- 当前工作目录: {cwd}
- workspace 目录绝对路径: {workspace_dir}
- 路径分隔符: {repr(os.sep)}（{'反斜杠' if os.sep == chr(92) else '正斜杠'}）
- **路径规则**：文件读写必须使用相对路径（如 `./workspace/xxx` 或 `workspace/xxx`），
  或使用 `os.path.join()` / `pathlib.Path`。**严禁**使用 `/workspace/` 这类 Linux 绝对路径。
{conda_info}
"""
        # 如果有执行计划，注入到系统提示词
        if plan:
            _subtasks = subtasks or []
            subtask_status = "\n".join([f"- [{s['status']}] {s['description']}" for s in _subtasks]) if _subtasks else ""
            
            env_info += f"""
## 执行计划（请严格按照计划步骤执行，不要跳过任何步骤）

{plan}

### 当前子任务状态
{subtask_status}

**重要**：必须按顺序完成计划中的所有步骤。完成一个步骤后，必须调用 `update_task_status` 更新状态，然后立即继续下一步。
不要在中间步骤后停止并报告"成功"——只有完成最后一个步骤后才算任务完成。
**严禁**仅输出"我将要执行..."之类的描述性文字而不调用工具。有未完成的步骤时，你必须调用工具来执行，而不是描述你打算做什么。
"""

        # 拼接 system addendum（[第 5 批] 给 LLM 立规矩的硬性规则）
        # 与具体 skill / 业务无关，所有 LLM 决策都受其约束
        try:
            from src.middleware.prompts import load_system_addendum
            _addendum = load_system_addendum()
            if _addendum:
                env_info = (
                    env_info
                    + "\n\n## 系统硬性规则（[Middleware] 通用元指令）\n\n"
                    + _addendum
                    + "\n"
                )
        except Exception as _add_err:
            logger.debug("system addendum 加载失败（不影响主流程）: %s", _add_err)

        return self._system_prompt_template + env_info

    # ═══════════════════════════════════════════════════
    #  节点 1/5：agent_node — LLM 决策核心
    # ═══════════════════════════════════════════════════

    def agent_node(self, state: AgentState) -> dict:
        """Agent 决策节点：LLM 根据上下文决定调用工具还是直接回复。

        执行流程：
        1. 动态构建系统提示词（注入 OS 信息 + 执行计划 + 子任务状态）
        2. 调用 LLM（支持工具绑定模式 / 无工具软着陆模式）
        3. 从工具调用参数中提取 selected_skill（用于依赖安装和统计）
        4. 返回更新后的 messages + iteration + selected_skill

        特殊处理：
        - LLM 调用失败时自动回退到无工具模式（兼容不支持 tools 的模型网关）
        - API 网关原始错误信息提取（vLLM、阿里云等非标准错误格式）
        """
        # 每轮动态拼接系统提示词，注入当前运行时环境和执行计划。
        plan = state.get("plan", "")
        system = SystemMessage(content=self._build_system_prompt(plan, subtasks=state.get("subtasks", [])))

        # ── 上下文长度防御（修复 B+C）：截断超长历史 ToolMessage，防止 messages 爆炸导致 LLM 决策卡死 ──
        # 触发场景：execute_code 输出超长 XML/JSON（如 EuropePMC 全文），3 步累积就可能 >30K tokens。
        # 软着陆（force_no_tools=True）后，LLM 切到 llm_raw，对长上下文更敏感，更易超时。
        # 关键设计：只对发给 LLM 的消息做截断，不修改 LangGraph state（保证日志/统计/重连完整）。
        _MAX_TOOL_OUTPUT = 4000   # 单个 ToolMessage 超过 4K 字符才截断
        _TOOL_HEAD_KEEP = 2000     # 保留头部 2K
        _TOOL_TAIL_KEEP = 500      # 保留尾部 500
        _raw_messages = list(state.get("messages", []))  # 复制 list 避免污染 state
        if _raw_messages:
            _trimmed_count = 0
            _trimmed_msgs = []
            for _m in _raw_messages:
                _c = getattr(_m, "content", None)
                if (
                    getattr(_m, "type", "") == "tool"
                    and isinstance(_c, str)
                    and len(_c) > _MAX_TOOL_OUTPUT
                ):
                    _omitted = len(_c) - _TOOL_HEAD_KEEP - _TOOL_TAIL_KEEP
                    _new_content = (
                        f"{_c[:_TOOL_HEAD_KEEP]}\n\n"
                        f"... [✂️ 系统已自动截断 | 原始 {len(_c)} 字符 | 中间 {_omitted} 字符已省略] ...\n\n"
                        f"{_c[-_TOOL_TAIL_KEEP:]}"
                    )
                    # 重建 ToolMessage（保留 tool_call_id 和 name 以维持 LangGraph 关联）
                    _trimmed_msgs.append(ToolMessage(
                        content=_new_content,
                        tool_call_id=getattr(_m, "tool_call_id", ""),
                        name=getattr(_m, "name", ""),
                    ))
                    _trimmed_count += 1
                else:
                    _trimmed_msgs.append(_m)
            if _trimmed_count > 0:
                logger.info(
                    "messages 截断 | 共 %d 个 ToolMessage 被截断 | force_no_tools=%s | 防 LLM 输入爆炸",
                    _trimmed_count, state.get("force_no_tools", False),
                )
                _raw_messages = _trimmed_msgs

        messages = [system] + _raw_messages

        # Stuck Detector 触发软着陆：强制使用无工具 LLM
        force_no_tools = state.get("force_no_tools", False)
        llm_to_use = self.llm_raw if force_no_tools else self.llm_with_tools

        try:
            response = llm_to_use.invoke(messages)
        except Exception as e:
            logger.error("LLM Invoke Error: %s: %s", type(e).__name__, e)

            # ── 提取 API 网关的原始错误信息 ──
            # 某些网关（如 vLLM、阿里云等）返回 {choices: null, code, success, msg}，
            # LangChain 只报 TypeError，真正的错误原因在 msg 字段里。
            _gateway_error_printed = False
            for _attr in ("response", "body", "api_response"):
                _raw = getattr(e, _attr, None)
                if _raw is None:
                    continue
                try:
                    if hasattr(_raw, "text"):
                        import json as _json
                        try:
                            _parsed = _json.loads(_raw.text)
                        except Exception:
                            logger.error("Error Response Body: %s", _raw.text)
                            continue
                    elif isinstance(_raw, dict):
                        _parsed = _raw
                    elif isinstance(_raw, str):
                        import json as _json
                        _parsed = _json.loads(_raw)
                    else:
                        continue

                    if isinstance(_parsed, dict):
                        _gw_msg = _parsed.get("msg", "")
                        _gw_code = _parsed.get("code", "")
                        _gw_success = _parsed.get("success", "")
                        if _gw_msg:
                            logger.error("Gateway Error: code=%s, success=%s, msg=%s", _gw_code, _gw_success, _gw_msg)
                            _gateway_error_printed = True
                        # choices 为 null 时也打印完整响应供排查
                        if _parsed.get("choices") is None and not _gw_msg:
                            import json as _json
                            logger.error("Error Response Body: %s", _json.dumps(_parsed, ensure_ascii=False))
                            _gateway_error_printed = True
                except Exception:
                    pass

            if not _gateway_error_printed:
                if hasattr(e, "response") and e.response is not None:
                    try:
                        logger.error("Error Response Body: %s", e.response.text)
                    except Exception:
                        pass
                
            # 某些模型网关不兼容 tools 参数时，自动退回到无工具模式重试。
            disable_tools_env = (os.getenv("LLM_DISABLE_TOOLS") or "").strip().lower()
            if disable_tools_env in {"1", "true", "yes", "y"}:
                response = self.llm_raw.invoke(messages)
            else:
                should_retry_without_tools = False
                err_text = str(e).lower()
                if any(k in err_text for k in ("tools", "tool_choice", "function", "unsupported", "invalid")):
                    should_retry_without_tools = True
                try:
                    from openai import BadRequestError  # type: ignore

                    if isinstance(e, BadRequestError):
                        should_retry_without_tools = True
                except Exception:
                    pass

                if should_retry_without_tools and self.llm_with_tools is not self.llm_raw:
                    response = self.llm_raw.invoke(messages)
                else:
                    raise
        # ── 防中途放弃：LLM 无 tool_calls 但有未完成子任务时自动续跑 ──
        # 路由函数（route_agent）无法持久化 state 修改，所以续跑逻辑放在节点内。
        # 场景：LLM 输出纯文本/XML格式工具调用而不产生标准 tool_calls → 自动注入
        # 续跑提示并重新调用 LLM，推动其正确调用工具继续执行。
        auto_resume_count = state.get("auto_resume_count", 0)
        MAX_AUTO_RESUME = 2
        has_tool_calls = hasattr(response, "tool_calls") and response.tool_calls
        
        if (
            not has_tool_calls
            and not force_no_tools                           # 软着陆模式下不续跑
            and auto_resume_count < MAX_AUTO_RESUME          # 安全阀
        ):
            subtasks = state.get("subtasks", [])
            pending = [s for s in subtasks if s.get("status") in ("pending", "in_progress")]
            if pending:
                # 注入续跑提示，推动 LLM 执行下一个步骤
                pending_desc = "; ".join(s.get("description", "")[:50] for s in pending[:3])
                continuation = SystemMessage(content=(
                    f"⚠️ 系统检测：你还有 {len(pending)} 个未完成的步骤（{pending_desc}）。"
                    "请立即通过工具调用执行下一个步骤，不要仅输出描述性文字。"
                ))
                # 将原始响应和续跑提示都加入 messages，再重新调用 LLM
                messages_with_continuation = messages + [response, continuation]
                auto_resume_count += 1
                logger.info(
                    "自动续跑 | 未完成子任务: %d | 续跑次数: %d/%d",
                    len(pending), auto_resume_count, MAX_AUTO_RESUME,
                )
                try:
                    response = llm_to_use.invoke(messages_with_continuation)
                    # 重新检查是否有 tool_calls
                    has_tool_calls = hasattr(response, "tool_calls") and response.tool_calls
                except Exception as e:
                    logger.warning("自动续跑 LLM 调用失败: %s", e)
        
        # 从工具调用参数里尽量还原"本轮真正选中的 skill"，用于观测统计。
        selected_skill = state.get("selected_skill", "")
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                args = tc.get("args", {})
                if tc["name"] == "retrieve_skills":
                    # 记录检索行为，selected_skill 暂不更新（等后续 read_file/execute_code 再定）
                    pass
                elif tc["name"] == "read_file":
                    selected_skill = args.get("skill_id", selected_skill)
                elif tc["name"] == "execute_code":
                    rel_skill = args.get("related_skill", "")
                    language = args.get("language", "")
                    if rel_skill:
                        selected_skill = rel_skill
                    else:
                        # 某些执行代码未显式填 related_skill，则尝试从脚本路径中反推。
                        code = args.get("code", "")
                        match = re.search(r"skills/([^/]+)/", code)
                        if match:
                            skill_id = match.group(1)
                            # 验证 skill_id 格式：kebab-case，长度合理，不含代码字符
                            if (
                                len(skill_id) <= 64
                                and re.match(r"^[a-z0-9][a-z0-9-]*$", skill_id)
                                and "\n" not in skill_id
                            ):
                                selected_skill = skill_id
                        # ⚠️ bash 命令不需要填写 related_skill，仅对非 bash 语言记录警告
                        prev_skill = state.get("selected_skill", "")
                        if prev_skill and not rel_skill and language != "bash":
                            logger.warning(
                                "LLM 调用 execute_code 时未填 related_skill！"
                                "当前已选技能=%s，第1层依赖预安装将跳过。"
                                "如果出现 ModuleNotFoundError 将触发第2层兜底安装。",
                                prev_skill,
                            )

        if selected_skill and selected_skill != state.get("selected_skill", ""):
            self.stats.record_selection(selected_skill)

        return {
            "messages": [response],
            "selected_skill": selected_skill,
            "iteration": state.get("iteration", 0) + 1,
            "force_no_tools": state.get("force_no_tools", False),
            "auto_resume_count": auto_resume_count,
        }

    # ═══════════════════════════════════════════════════
    #  节点 2/5：post_tools_node — 工具执行后处理
    # ═══════════════════════════════════════════════════

    def post_tools_node(self, state: AgentState) -> dict:
        """解析 ToolMessage 的执行状态并写入 AgentState

        增强特性：
        - API 调用细粒度追踪：识别认证失败/限流/网络错误等具体类型
        - pip install 结果追踪：识别包安装失败
        - 多语言执行结果：支持 Python/Bash/R/JavaScript 的结果解析
        - Stuck Detector（Claude Code 做法）：比较最近 N 轮 execute_code 的 code 参数是否高度重复，
          检测到循环时注入 Reflection Message 并启用软着陆（force_no_tools）
        
        只对 execute_code 做执行状态追踪；read_file 是纯读取，不影响 tool_status。
        """
        tool_status = state.get("tool_status", "")
        extra_messages = []
        iteration = state.get("iteration", 0)
        consecutive_same = state.get("consecutive_same_result", 0)
        last_execute_code = state.get("last_execute_code", "")
        force_no_tools = state.get("force_no_tools", False)
        api_domain_counts = dict(state.get("api_domain_counts", {}))
        _code_window = list(state.get("recent_code_window", []))
        
        if state["messages"] and state["messages"][-1].type == "tool":
            last_msg = state["messages"][-1]
            tool_name = getattr(last_msg, "name", "")

            if tool_name == "execute_code":
                content = last_msg.content

                # ── 结构化错误分类（替代脆弱的字符串匹配）──
                error_type = _classify_error(content)
                
                # 将 ErrorType 映射为用户友好的状态字符串
                STATUS_MAP = {
                    ErrorType.SUCCESS: "成功",
                    ErrorType.SCRIPT_ERROR: "失败",
                    ErrorType.IMPORT_ERROR: "缺少模块",
                    ErrorType.PACKAGE_INSTALL_FAILED: "包安装失败",
                    ErrorType.API_AUTH_FAILED: "外部 API 认证失败",
                    ErrorType.API_RATE_LIMIT: "外部 API 限流",
                    ErrorType.API_NETWORK: "外部 API 网络错误",
                    ErrorType.API_RESOURCE_NOT_FOUND: "外部 API 资源不存在",
                    ErrorType.API_SERVER_ERROR: "外部 API 服务端错误",
                    ErrorType.API_GENERIC: "外部 API 失败",
                    ErrorType.TIMEOUT: "超时",
                    ErrorType.FILE_NOT_FOUND: "文件不存在",
                    ErrorType.UNKNOWN_FAILURE: "失败",
                }
                tool_status = STATUS_MAP.get(error_type, "失败")

                # ── ErrorMemory 记录 + 升级检测（解决 P6 错误循环）──
                # 任何非 SUCCESS 的结果都进入错误记忆。
                # 同一指纹出现 ≥ threshold 次时，注入"反向教育"提示到 LLM。
                if error_type != ErrorType.SUCCESS:
                    try:
                        em = get_error_memory()
                        count = em.record(content, tool_name="execute_code")
                        if count >= em.threshold:
                            learning_hint = em.build_learning_hint(content)
                            extra_messages.append(SystemMessage(content=learning_hint))
                            logger.info(
                                "ErrorMemory 升级 | count=%d | threshold=%d | hint 已注入",
                                count, em.threshold,
                            )
                    except Exception as e:
                        logger.warning("ErrorMemory 记录失败: %s", e)

                # 成功时检查是否需要提醒继续执行分析步骤
                if error_type == ErrorType.SUCCESS:
                    # 方法1：优先相信 LLM 主动提供的 step_type 标记。
                    step_type = "analysis"  # 默认
                    for msg in reversed(state["messages"]):
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                if tc["name"] == "execute_code":
                                    step_type = tc.get("args", {}).get("step_type", "analysis")
                                    break
                            break
                    
                    need_reminder = False
                    
                    # step_type 明确标记为 preparation → 必须提醒
                    if step_type == "preparation":
                        need_reminder = True
                    
                    # 方法2：当 step_type 不可靠时，再用关键词兜底判断是否只是准备步骤。
                    if step_type != "final":
                        prep_keywords = [
                            "已创建文件", "已写入文件", "文件已保存", "文件已创建",
                            "created file", "saved to", "written to",
                            "makeblastdb", "建库完成", "database created",
                            "已下载", "downloaded", "已安装",
                        ]
                        analysis_keywords = [
                            "blastn", "blastp", "blastx", "比对结果", "alignment",
                            "hits found", "查询完成", "分析结果", "analysis result",
                            "差异表达", "differentially expressed", "variant", "变异",
                            "prediction", "预测结果", "annotation", "注释结果",
                        ]
                        is_prep_only = any(kw in content.lower() for kw in [k.lower() for k in prep_keywords])
                        has_analysis = any(kw in content.lower() for kw in [k.lower() for k in analysis_keywords])
                        
                        if is_prep_only and not has_analysis:
                            need_reminder = True
                    
                    if need_reminder:
                        reminder = SystemMessage(content=(
                            "⚠️ 提醒：上一步 execute_code 似乎只完成了数据准备（创建了文件/建库/下载数据），"
                            "尚未执行核心分析步骤。请检查用户的问题是否需要继续执行分析（如 BLAST 搜索、"
                            "差异表达分析、变异调用等），如果是，请继续调用工具完成完整分析流程，"
                            "不要在此停止并报告成功。"
                        ))
                        extra_messages.append(reminder)

                # ── 确定性错误修正策略（Claude Code 做法）──
                # 对所有 execute_code 结果都扫描，不受 tool_status 影响。
                # 原因：tools.py 可能因脚本内部 try/except 导致 returncode=0
                # 而标记为"成功"，但输出中实际包含错误信息（如 [执行失败-脚本内部错误]）。
                fix_suggestions = []

                # ImportError / ModuleNotFoundError
                if "ImportError" in content or "ModuleNotFoundError" in content:
                    import_match = re.search(r"No module named ['\"]?(\w+)['\"]?", content)
                    if import_match:
                        pkg = import_match.group(1)
                        installed = False
                        skill_deps_installed = False
                        skill_deps_failed = []

                        # ── 优先策略：若有 selected_skill，批量安装该技能的所有依赖 ──
                        selected_skill = state.get("selected_skill", "")
                        if selected_skill:
                            try:
                                from src.tools import auto_install_skill_deps
                                deps_result = auto_install_skill_deps(selected_skill)
                                if deps_result["installed"]:
                                    skill_deps_installed = True
                                    installed = True  # 批量安装中有成功项
                                    logger.info(
                                        "批量安装技能 %s 的依赖成功: %s",
                                        selected_skill, deps_result["installed"],
                                    )
                                if deps_result["failed"]:
                                    skill_deps_failed = [f["package"] for f in deps_result["failed"]]
                                    logger.warning(
                                        "批量安装技能 %s 部分失败: %s",
                                        selected_skill, skill_deps_failed,
                                    )
                                if deps_result["skipped"]:
                                    logger.info(
                                        "跳過已安装: %s", deps_result["skipped"]
                                    )
                            except Exception as e:
                                logger.warning("批量安装 %s 依赖时异常: %s", selected_skill, e)

                        # ── 兜底策略：若没有 selected_skill 或批量安装后仍缺失，单独安装 ──
                        if not installed:
                            try:
                                from src.tools import auto_install_package
                                installed = auto_install_package(pkg)
                            except Exception as e:
                                logger.warning("自动安装 %s 时异常: %s", pkg, e)

                        # ── 兜底策略：从代码中解析所有 import 并批量安装 ──
                        if not installed:
                            current_code = ""
                            for msg in reversed(state["messages"]):
                                if hasattr(msg, "tool_calls") and msg.tool_calls:
                                    for tc in msg.tool_calls:
                                        if tc["name"] == "execute_code":
                                            current_code = tc.get("args", {}).get("code", "")
                                            break
                                    break
                            if current_code:
                                try:
                                    from src.tools import auto_install_from_code
                                    code_result = auto_install_from_code(current_code)
                                    if code_result["installed"]:
                                        installed = True
                                        logger.info(
                                            "从代码中批量安装成功: %s",
                                            code_result["installed"],
                                        )
                                    if code_result["failed"]:
                                        skill_deps_failed.extend(
                                            [f["package"] for f in code_result["failed"]]
                                        )
                                except Exception as e:
                                    logger.warning("从代码批量安装异常: %s", e)

                        # ── 构建修正建议 ──
                        if installed:
                            if skill_deps_installed:
                                fix_suggestions.append(
                                    f"检测到缺少依赖: {pkg}。已批量安装技能 '{selected_skill}' "
                                    "的 requirements.txt 中所有依赖到隔离环境，"
                                    "请重试原代码即可。无需手动写 pip install。"
                                )
                            else:
                                fix_suggestions.append(
                                    f"检测到缺少依赖: {pkg}。已自动安装到隔离环境，"
                                    "请重试原代码即可。无需手动写 pip install。"
                                )
                        else:
                            # 安装失败，告知用户
                            fail_detail = ""
                            if skill_deps_failed:
                                fail_detail = (
                                    f"以下包安装失败: {', '.join(skill_deps_failed)}。"
                                )
                            if selected_skill:
                                fix_suggestions.append(
                                    f"检测到缺少依赖: {pkg}。已尝试批量安装技能 "
                                    f"'{selected_skill}' 的 requirements.txt，"
                                    f"但{fail_detail or '安装失败'}。"
                                    "可能是当前 Python 环境版本不兼容或网络问题，"
                                    "请手动在虚拟环境中安装这些包后重试。"
                                )
                            else:
                                fix_suggestions.append(
                                    f"检测到缺少依赖: {pkg}。{fail_detail}"
                                    "自动安装失败，可能当前环境不兼容或网络问题。"
                                    "请在代码顶部添加 `pip install {pkg}` 手动安装。"
                                )
                    else:
                        fix_suggestions.append(
                            "检测到 ImportError。请在代码顶部添加缺失的 import 语句，"
                            "系统会自动检测并安装依赖包。"
                        )

                # R 包缺失检测（中文 R 环境："不存在叫'DESeq2'这个名字的程辑包"）
                r_pkg_missing = None
                if "不存在叫" in content and "程辑包" in content:
                    r_pkg_match = re.search(r"不存在叫'([^']+)'这个名字的程辑包", content)
                    if r_pkg_match:
                        r_pkg_missing = r_pkg_match.group(1)
                # R 包缺失检测（英文 R 环境："there is no package called 'DESeq2'"）
                elif re.search(r"there is no package called\s+['\"]?(\w+(?:\.\w+)*)['\"]?", content, re.IGNORECASE):
                    r_pkg_match = re.search(r"there is no package called\s+['\"]?(\w+(?:\.\w+)*)['\"]?", content, re.IGNORECASE)
                    if r_pkg_match:
                        r_pkg_missing = r_pkg_match.group(1)

                if r_pkg_missing:
                    fix_suggestions.append(
                        f"检测到 R 包缺失: {r_pkg_missing}。"
                        "系统将自动通过 conda 创建隔离 R 环境并安装所需的 R 包。"
                        "【重要】请直接重试原代码，绝对不要添加 install.packages() 或 BiocManager::install()。"
                        "这些安装命令在 Windows 上会极慢且容易卡死。"
                        "如果重试后仍然失败，可以尝试使用 Python 替代方案（如 pydeseq2 替代 DESeq2）。"
                    )

                # FileNotFoundError
                if "FileNotFoundError" in content or "No such file" in content:
                    fix_suggestions.append(
                        "检测到文件不存在。请检查 workspace 目录中的实际文件名，"
                        "可以在代码中用 `import os; print(os.listdir('.'))` 列出可用文件。"
                    )

                # 张量/矩阵维度不匹配
                if "shape" in content and ("mismatch" in content or "cannot be multiplied" in content):
                    fix_suggestions.append(
                        "检测到维度不匹配错误。请在执行前打印数据的 shape "
                        "（如 `print(data.shape)`）确认维度是否正确。"
                    )

                # KeyError（常见于字典访问异构数据 / pandas 列名不存在）
                if "KeyError" in content:
                    key_match = re.search(r"KeyError: ['\"](\w+)['\"]", content)
                    if key_match:
                        missing_key = key_match.group(1)
                        fix_suggestions.append(
                            f"检测到 KeyError: 键名 '{missing_key}' 不存在。"
                            "不同数据源/API 可能使用不同的字段名，"
                            "请用 `print(list(data.keys()))` 查看数据字典的实际键名，"
                            f"并改用 `.get('{missing_key}', default_value)` 安全访问。"
                        )
                    else:
                        # 可能是 pandas 列名 KeyError
                        fix_suggestions.append(
                            "检测到 KeyError。请检查数据字典的实际键名或 DataFrame 列名，"
                            "可以用 `print(list(data.keys()))` / `print(df.columns.tolist())` 查看可用字段，"
                            "并改用 `.get()` 方法安全访问字典。"
                        )

                # ── 新增错误模式 ──

                # AttributeError（has no attribute — 常见于 API 名称变更）
                if "has no attribute" in content or "AttributeError" in content:
                    attr_match = re.search(r"has no attribute ['\"]?(\w+)['\"]?", content)
                    if attr_match:
                        fix_suggestions.append(
                            f"检测到 API 名称变化：`{attr_match.group(1)}` 不存在。"
                            "请检查该库的当前版本文档，使用正确的 API 名称。"
                        )
                    else:
                        fix_suggestions.append(
                            "检测到 AttributeError。请检查对象是否具有该属性，"
                            "可能因库版本更新导致 API 变化。"
                        )

                # numpy 格式字符串错误
                if "unsupported format string" in content:
                    fix_suggestions.append(
                        "检测到 numpy 格式字符串错误。预测结果是 ndarray，"
                        "请用 `float(pred[0][0])` 或 `.item()` 转换为 Python 标量后再格式化输出。"
                    )

                # numpy 标量转换错误
                if "only 0-dimensional arrays can be converted to Python scalars" in content:
                    fix_suggestions.append(
                        "检测到 numpy 标量转换错误。预测结果是多维数组，"
                        "请用 `pred.flatten()[0]` 或 `pred[0][0]` 提取单个值，"
                        "或用 `.item()` 转换为 Python 标量。"
                    )

                # 脚本内部报错标记（Error during xxx:）
                if "Error during" in content and "脚本内部错误" not in content:
                    fix_suggestions.append(
                        "检测到脚本内部执行错误。请检查 stderr 中的具体"
                        "错误信息，针对性修正代码后重试。"
                    )

                # ValueError（常见于数据类型/格式不匹配）
                if "ValueError" in content and "unsupported format" not in content:
                    fix_suggestions.append(
                        "检测到 ValueError。请检查数据的类型和格式是否与预期一致，"
                        "可以用 `print(type(data))` 和 `print(data[:5])` 检查数据。"
                    )

                # TypeError（常见于类型不匹配的运算）
                if "TypeError" in content and "unsupported format" not in content:
                    fix_suggestions.append(
                        "检测到 TypeError。请检查操作数的类型是否正确，"
                        "可能需要类型转换（如 `int()`, `float()`, `str()`）。"
                    )

                if fix_suggestions:
                    fix_msg = SystemMessage(
                        content="💡 系统自动检测到错误模式，建议修正：\n" + "\n".join(fix_suggestions)
                    )
                    extra_messages.append(fix_msg)

                # ── Stuck Detector（Claude Code 做法）──
                # 比较连续 execute_code 调用的 code 参数是否高度相似，
                # 而非只看 tool_status 字符串。这能更精准地检测"LLM 在反复执行同一段代码"。
                # 改进：归一化代码后再比较，避免仅因注释/空白不同而漏检。
                current_code = ""
                for msg in reversed(state["messages"]):
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            if tc["name"] == "execute_code":
                                current_code = tc.get("args", {}).get("code", "")
                                break
                        break

                normalized_code = _normalize_code_for_stuck_detect(current_code) if current_code else ""

                if normalized_code and last_execute_code:
                    # 计算归一化后代码的相似度（简单 Jaccard 按行比较）
                    current_lines = set(normalized_code.split('\n'))
                    last_lines = set(last_execute_code.split('\n'))
                    if current_lines and last_lines:
                        intersection = current_lines & last_lines
                        union = current_lines | last_lines
                        similarity = len(intersection) / len(union) if union else 0
                        # 相似度超过 80% 视为同一代码
                        if similarity >= 0.8:
                            consecutive_same += 1
                        else:
                            consecutive_same = 1
                    else:
                        consecutive_same = 1
                elif normalized_code:
                    consecutive_same = 1
                last_execute_code = normalized_code

                # Stuck 检测：连续 2 次相同 code → 注入 Reflection Message + 软着陆
                if consecutive_same >= 2:
                    force_no_tools = True
                    reflection_msg = SystemMessage(content=(
                        f"⚠️ 系统检测：你已连续 {consecutive_same} 次调用 execute_code "
                        "传入相同或高度相似的代码。你似乎陷入了循环。\n\n"
                        "请尝试不同的方法，或总结已有的发现直接告知用户。\n"
                        "工具调用已被系统禁用，请立即向用户回复当前进展。"
                    ))
                    extra_messages.append(reflection_msg)

                # ── 周期循环检测 ──
                # Stuck Detector 只比较相邻 2 次代码，但 LLM 可能循环使用 N 段不同代码
                # （如 A→B→C→D→A→B→C→D），每次都绕过检测。
                # 此处追踪最近 8 次归一化代码的 MD5 指纹滑动窗口，检测周期重复模式。
                if normalized_code and not force_no_tools:
                    _fingerprint = hashlib.md5(normalized_code.encode()).hexdigest()[:8]
                    _code_window.append(_fingerprint)
                    if len(_code_window) > 8:
                        _code_window = _code_window[-8:]

                    if len(_code_window) >= 4:
                        for _period in [2, 3, 4]:
                            if len(_code_window) >= _period * 2:
                                _pattern_a = _code_window[-(_period * 2):-_period]
                                _pattern_b = _code_window[-_period:]
                                if _pattern_a == _pattern_b:
                                    force_no_tools = True
                                    logger.warning(
                                        "周期循环检测触发 | period=%d | window=%s | 强制软着陆",
                                        _period, _code_window,
                                    )
                                    _cycle_msg = SystemMessage(content=(
                                        f"⚠️ 系统检测：你的工具调用呈现周期性循环（周期={_period}），"
                                        "说明你在用不同代码反复查询同一数据源但无法获得有效结果。\n\n"
                                        "请立即停止当前策略，改用以下方法：\n"
                                        "  1. 基于已有发现综合总结回答\n"
                                        "  2. 明确说明数据获取的局限性\n"
                                        "  3. 尝试完全不同的数据源或方法\n"
                                        "工具调用已被系统禁用，请立即向用户回复当前进展。"
                                    ))
                                    extra_messages.append(_cycle_msg)
                                    break

                # ── 同域 API 重复调用检测 ──
                # Stuck Detector 比较代码文本，但 LLM 每次换关键词就能绕过。
                # 本检测从执行结果中的 [API] 行提取域名，追踪同一 API 被调用了多少次。
                # 当同一域名累计调用 ≥ 3 次时，说明 Agent 在反复查询同一数据源但找不到想要的结果。
                # 注意：api_domain_counts 已在函数开头（行 644）初始化，此处直接基于函数级变量累加
                for _api_line in re.finditer(r'\[API\]\s+(?:GET|POST|PUT|DELETE)\s+https?://([^/\s]+)', content):
                    _domain = _api_line.group(1)
                    api_domain_counts[_domain] = api_domain_counts.get(_domain, 0) + 1

                # ── API 错误字段提取（增强同域检测）──
                # 从 [API-Body] 中提取 400 错误的关键词，注入后续迭代避免重复踩坑
                _api_error_hints = []
                for _body_match in re.finditer(r'\[API-Body\]\s*(\{.+)', content):
                    _body_text = _body_match.group(1)[:500]
                    if 'Cannot query field' in _body_text:
                        _field_match = re.search(r"Cannot query field '(\w+)' on type '(\w+)'", _body_text)
                        if _field_match:
                            _api_error_hints.append(
                                f"  - `{_field_match.group(1)}` 在 `{_field_match.group(2)}` 类型中不存在"
                            )
                    elif 'invalid field name' in _body_text.lower():
                        _inv_match = re.search(r"invalid field name:\s*'(\w+)'", _body_text, re.IGNORECASE)
                        if _inv_match:
                            _api_error_hints.append(
                                f"  - 字段 `{_inv_match.group(1)}` 不是有效参数名"
                            )

                if _api_error_hints:
                    extra_messages.append(SystemMessage(content=(
                        "⚠️ API 字段错误提醒（以下字段/参数已确认不可用，请勿再使用）：\n"
                        + "\n".join(_api_error_hints)
                        + "\n请查阅 SKILL.md 文档中的正确字段名，或使用 print 查看 API 返回的实际数据结构。"
                    )))

                # ── GraphQL Schema 自愈（[SchemaHealer] 解析 Did you mean 'X'）──
                # 通用化：不依赖具体 API；任何 GraphQL 400 错误都会自动提取字段重命名建议
                try:
                    from src.middleware.schema_healer import GraphQLHealer
                    _healer = GraphQLHealer.shared()
                    _new_hints: list[dict] = []
                    for _body_match in re.finditer(r'\[API-Body\]\s*(\{.+?)(?:\n|$)', content):
                        _healer.learn_from_error(_body_match.group(1)[:800])
                    # 若 healer 学到新重命名规则，注入"反向教育"提示
                    _learning_msg = _healer.build_learning_hint(_new_hints if _new_hints else None)
                    if _learning_msg:
                        # 累加到 API 错误提醒之后（合并输出）
                        extra_messages.append(SystemMessage(content=_learning_msg))
                        logger.info(
                            "SchemaHealer 触发 | 已记录 %d 条重命名规则",
                            len(_healer._renames.all_renames()),
                        )
                except Exception as _sh_err:
                    logger.debug("SchemaHealer 处理失败（不影响主流程）: %s", _sh_err)

                _max_domain = max(api_domain_counts, key=api_domain_counts.get) if api_domain_counts else ""
                _max_count = api_domain_counts.get(_max_domain, 0)
                # 有明确 API 字段错误时阈值降低：2 次就强制停止（避免 5+ 次无效重试同一错误字段）
                _force_threshold = 2 if _api_error_hints else 3
                if _max_count >= _force_threshold and not force_no_tools:
                    force_no_tools = True
                    logger.warning("同域 API 重复调用检测触发 | domain=%s | count=%d | 强制软着陆",
                                   _max_domain, _max_count)
                    reflection_msg = SystemMessage(content=(
                        f"⛔ 系统检测：你已 { _max_count} 次查询 {_max_domain}，"
                        "但始终未获得目标数据。继续换关键词查询相同 API 不会产生不同结果。\n\n"
                        "请立即停止查询，改用以下策略：\n"
                        "  1. 基于已有数据和自身训练知识综合回答\n"
                        "  2. 在回复中明确说明数据来源和局限性\n"
                        "工具调用已被系统禁用，请立即向用户回复当前进展。"
                    ))
                    extra_messages.append(reflection_msg)
                elif _max_count >= 2 and not force_no_tools:
                    # 第 2 次时给出警告，但不禁用工具
                    extra_messages.append(SystemMessage(content=(
                        f"⚠️ 系统提示：你已 {_max_count} 次查询 {_max_domain}，"
                        "如果再次查询仍无法获得目标数据，系统将强制停止工具调用。"
                        "建议：基于已有结果和自身知识综合回答，而非继续尝试不同关键词。"
                    )))

                # ── 手写 HTTP 请求检测（问题 B）──
                # 当 execute_code 的代码中包含 requests.get/post 或 urllib 调用，
                # 且当前已有 selected_skills，说明有封装函数可用但 Agent 没用。
                # current_code 已在 Stuck Detector 段提取，直接复用
                _raw_http = bool(re.search(r'requests\.(get|post|put|delete|patch)\(', current_code)
                                or 'urllib.request' in current_code)
                _has_tu_tools = 'tu.tools' in current_code
                _selected_skills = state.get("selected_skill", "") or ",".join(state.get("retrieved_skills", []))

                if _raw_http and not _has_tu_tools and _selected_skills and tool_status == "成功":
                    extra_messages.append(SystemMessage(content=(
                        "⛔ 系统检测：你正在手写 HTTP 请求（requests.get/post），"
                        "但已加载的技能提供了封装函数（如 tu.tools.xxx）。\n"
                        "手写 HTTP 请求容易导致参数错误、端点遗漏、认证缺失，请改用技能封装函数。\n"
                        "如果不确定封装函数的调用方式，请用 read_file 查看技能的 SKILL.md 文档。"
                    )))

                # ── 连续空输出检测 ──
                # 即使代码不同，如果多次 produce 空/无意义输出，也说明陷入了无效循环
                consecutive_empty = state.get("consecutive_empty_result", 0)
                is_empty_output = (
                    "[执行成功-输出为空]" in content
                    or (tool_status == "成功" and len(content.strip()) < 20)
                )
                if is_empty_output:
                    consecutive_empty += 1
                else:
                    consecutive_empty = 0

                if consecutive_empty >= 3 and not force_no_tools:
                    force_no_tools = True
                    logger.warning("连续空输出检测触发 | consecutive_empty=%d | 强制软着陆", consecutive_empty)
                    reflection_msg = SystemMessage(content=(
                        f"⚠️ 系统检测：你已连续 {consecutive_empty} 次执行代码但均未获得有效输出。"
                        "你可能在反复尝试同一个无法成功的 API 或数据源。\n\n"
                        "请立即停止重试，换用以下策略：\n"
                        "  1. 使用模拟数据或本地知识库完成分析\n"
                        "  2. 直接基于已有知识给出结论性回复\n"
                        "工具调用已被系统禁用，请立即向用户回复当前进展。"
                    ))
                    extra_messages.append(reflection_msg)

                skill_id = state.get("selected_skill", "")
                prev_status = state.get("tool_status", "")
                is_retry = bool(prev_status and prev_status != "成功")
                self.stats.record_execution(skill_id, tool_status, is_retry=is_retry)
            # read_file: 保持原有 tool_status 不变，不记录执行统计
            else:
                # 非 execute_code 的工具调用，不更新连续计数
                consecutive_same = 0
                consecutive_empty = 0
                last_execute_code = ""
                # api_domain_counts 保持函数级初始值（非 execute_code 无 [API] 行，无需更新）

                # ── 非 execute_code 工具的 ErrorMemory 记录 ──
                # 统一捕获 read_file / retrieve_skills / update_task_status 的错误
                tool_content = last_msg.content if hasattr(last_msg, "content") else ""
                if (
                    tool_content
                    and isinstance(tool_content, str)
                    and ("[错误]" in tool_content or "[执行失败]" in tool_content)
                ):
                    try:
                        em = get_error_memory()
                        count = em.record(tool_content, tool_name=tool_name)
                        if count >= em.threshold:
                            learning_hint = em.build_learning_hint(tool_content)
                            extra_messages.append(SystemMessage(content=learning_hint))
                            logger.info(
                                "ErrorMemory 升级（非execute_code）| tool=%s | count=%d | hint 已注入",
                                tool_name, count,
                            )
                    except Exception as e:
                        logger.warning("ErrorMemory 记录失败（非execute_code）: %s", e)

        # ── 迭代即将耗尽时强制软着陆 ──
        # 当迭代接近上限时，无条件触发 force_no_tools，让 LLM 在下一轮
        # 用无工具模式生成总结回复（报告进度 + 建议用户输入"继续"）。
        # 这确保 Agent 永远不会静默结束，总会给用户一个交代。
        if iteration >= self.max_iterations - 2 and not force_no_tools:
            tool_status = "未完成（迭代耗尽）"
            force_no_tools = True
            # 收集当前子任务进度信息，注入到总结提示中
            subtasks = state.get("subtasks", [])
            progress_info = ""
            if subtasks:
                done_steps = [s.get("description", "") for s in subtasks if s.get("status") == "done"]
                pending_steps = [s.get("description", "") for s in subtasks if s.get("status") != "done"]
                if done_steps:
                    progress_info += f"\n已完成 {len(done_steps)} 步。"
                if pending_steps:
                    progress_info += f"未完成 {len(pending_steps)} 步。"

            truncation_msg = SystemMessage(content=(
                f"⚠️ 系统通知：已达到最大迭代次数 ({self.max_iterations})，任务尚未完成。"
                f"{progress_info}\n\n"
                "请立即生成一份进度总结回复，包含以下内容：\n"
                "1. 已完成的分析步骤及关键发现\n"
                "2. 尚未完成的步骤\n"
                "3. 建议用户输入'继续'以完成剩余分析\n\n"
                "工具调用已被系统禁用，请直接回复用户。"
            ))
            extra_messages.append(truncation_msg)
            logger.warning("迭代耗尽软着陆 | iteration=%d/%d | 强制 LLM 生成总结", iteration, self.max_iterations)

        result = {
            "tool_status": tool_status,
            "consecutive_same_result": consecutive_same,
            "consecutive_empty_result": consecutive_empty,
            "last_execute_code": last_execute_code,
            "api_domain_counts": api_domain_counts,
            "recent_code_window": _code_window,
            # 粘性 force_no_tools：一旦触发就保持，防止后续迭代重置
            "force_no_tools": force_no_tools or state.get("force_no_tools", False),
        }
        
        # ── 拦截 update_task_status 并更新状态 ──
        if state["messages"] and state["messages"][-1].type == "tool":
            last_msg = state["messages"][-1]

            # ── 上下文长度防御（修复 C，双保险）：在 post_tools_node 写入 state 时截断最新 ToolMessage ──
            # 与 B 方案（agent_node 调 LLM 前截断「读」侧）配合，C 在「写」侧截断，
            # 保证 state 不会因超长 execute_code 输出（如 EuropePMC 全文 XML/JSON）持续膨胀。
            # LangGraph add_messages 会按 message.id 去重：保留同 id 即覆盖原 ToolMessage。
            _last_content = getattr(last_msg, "content", None)
            if isinstance(_last_content, str) and len(_last_content) > 4000:
                _omitted_c = len(_last_content) - 2000 - 500
                _new_content_c = (
                    f"{_last_content[:2000]}\n\n"
                    f"... [✂️ post_tools 截断 | 原始 {len(_last_content)} 字符 | 中间 {_omitted_c} 字符已省略] ...\n\n"
                    f"{_last_content[-500:]}"
                )
                _trimmed_tool_msg = ToolMessage(
                    content=_new_content_c,
                    tool_call_id=getattr(last_msg, "tool_call_id", ""),
                    name=getattr(last_msg, "name", ""),
                )
                # 保持相同的 id 以让 add_messages 按 id 去重覆盖原 ToolMessage
                if getattr(last_msg, "id", None):
                    _trimmed_tool_msg.id = last_msg.id
                result["messages"] = [_trimmed_tool_msg]
                logger.info(
                    "post_tools 截断 | tool=%s | 原始 %d 字符 → 截断后 %d 字符 | 防 state 膨胀",
                    getattr(last_msg, "name", "?"), len(_last_content), len(_new_content_c),
                )

            if getattr(last_msg, "name", "") == "update_task_status":
                subtasks = list(state.get("subtasks", []))  # 拷贝一份避免直接修改
                for msg in reversed(state["messages"]):
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            if tc["name"] == "update_task_status":
                                args = tc.get("args", {})
                                # step_number: 1-indexed; task_index: 0-indexed（兼容旧调用）
                                raw_index = args.get("step_number") or args.get("task_index")
                                new_status = args.get("status")
                                if raw_index is not None and new_status:
                                    # step_number 是 1-indexed → 转为 0-indexed
                                    task_index = raw_index - 1 if args.get("step_number") is not None else raw_index
                                    if 0 <= task_index < len(subtasks):
                                        new_task = dict(subtasks[task_index])
                                        new_task["status"] = new_status
                                        if args.get("result_summary"):
                                            new_task["result_summary"] = args["result_summary"]
                                        subtasks[task_index] = new_task
                        break
                result["subtasks"] = subtasks

        if extra_messages:
            result["messages"] = extra_messages
        return result

    # ═══════════════════════════════════════════════════
    #  节点 3/5：auto_retrieve_node — 系统自动检索
    # ═══════════════════════════════════════════════════

    def _detect_llm_retrieve_request(self, state: AgentState) -> str | None:
        """检测 LLM 是否在上轮主动调用了 retrieve_skills，提取其查询词。
            
        用于支持跨领域二次检索：当 LLM 发现需要另一个领域的技能时，
        会主动调用 retrieve_skills(query=...)，此时系统应允许再次检索。
            
        Returns:
            查询词字符串，如果未检测到则返回 None
        """
        # 从后往前找最近的 AI 消息（带 tool_calls）
        for msg in reversed(state["messages"]):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc["name"] == "retrieve_skills":
                        query = tc.get("args", {}).get("query", "")
                        if query:
                            return query
                # 找到了带 tool_calls 的消息但不是 retrieve_skills，停止
                break
        return None
    
    def auto_retrieve_node(self, state: AgentState) -> dict:
        """在 Agent 决策前自动检索相关技能并注入上下文。
    
        触发条件：
        - 首次进入时（iteration == 0）：使用用户原始查询
        - 后续循环：仅当 LLM 主动调用了 retrieve_skills(query=...) 时触发
            
        通过距离阈值过滤不相关结果，日常问候等不会触发注入。
        """
        iteration = state.get("iteration", 0)
        user_query = ""
            
        if iteration == 0:
            # 首次进入：使用用户最新一条消息（多轮对话中取最后一条 HumanMessage）
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    user_query = msg.content
                    break
        else:
            # 后续循环：检查 LLM 是否主动请求检索
            llm_query = self._detect_llm_retrieve_request(state)
            if llm_query:
                user_query = llm_query
                logger.info("LLM 主动请求二次检索: %s", user_query)
            else:
                return {}  # LLM 未请求检索，跳过
            
        if not user_query:
            return {}

        # 2. 语义检索 + 距离分数
        skills, distances = self.retriever.retrieve_with_scores(
            user_query, k=self.top_k
        )

        if not skills:
            logger.debug("自动检索: 无匹配技能")
            return {}

        # 3. 相关性阈值过滤
        relevant = [
            s for s, d in zip(skills, distances) if d <= AUTO_RETRIEVE_MAX_DISTANCE
        ]

        # 4. Skill ID 兜底：查询中包含完整 skill ID 时强制加入（与 retrieve_skills 一致）
        relevant_ids = {s["id"] for s in relevant}
        query_lower = user_query.lower()
        for skill_id in self.retriever.registry_by_id:
            if skill_id in relevant_ids:
                continue
            if re.search(r'(?<![a-z0-9-])' + re.escape(skill_id) + r'(?![a-z0-9-])', query_lower):
                forced = self.retriever.get(skill_id)
                if forced:
                    relevant.insert(0, {
                        "id": forced["id"],
                        "description": forced.get("description", ""),
                        "path": forced.get("path", ""),
                    })
                    relevant_ids.add(forced["id"])
                    logger.info("自动检索 → 兜底命中: %s", forced["id"])

        if relevant:
            logger.info(
                "自动检索 → 召回 %d 个技能 (阈值过滤后 %d 个): %s | 距离: %s",
                len(skills), len(relevant),
                [s["id"] for s in relevant],
                [f"{d:.3f}" for s, d in zip(skills, distances) if s in relevant],
            )
        else:
            best_dist = distances[0] if distances else 0.0
            logger.info(
                "自动检索 → 召回 %d 个技能 (阈值过滤后 0 个，均不相关): 最佳距离 %.3f > 阈值 %.2f",
                len(skills), best_dist, AUTO_RETRIEVE_MAX_DISTANCE,
            )
            return {}

        # 4. 构建注入消息（SystemMessage，含强约束）
        ctx_parts = []
        for skill in relevant:
            skeleton = self.ctx.load_skeleton(
                skill["id"]
            )
            ctx_parts.append(f"## 技能: {skill['id']}\n{skeleton}")

        context = "\n\n".join(ctx_parts)

        hint_msg = SystemMessage(content=(
            f"🔍 系统已自动检索到 {len(relevant)} 个相关技能"
            f"（基于你的查询自动匹配）：\n\n"
            f"{'=' * 60}\n\n{context}\n\n"
            f"{'=' * 60}\n\n"
            f"⚠️ **重要提示**：\n"
            f"1. 上述技能已为你预加载完整文档（含 API 地址、代码示例），请在后续操作中严格遵循\n"
            f"2. **禁止凭记忆直接编写 API URL 或端点路径**——"
            f"必须从上方 SKILL.md 文档中获取最新地址和调用方式\n"
            f"3. 如果技能目录提供了封装脚本（如 tu.tools.xxx、scripts/fda_query.py），"
            f"必须优先使用而非自己从零实现\n"
            f"4. 代码示例中的参数（如 limit、pageSize）不要随意修改"
        ))

        self.stats.record_retrieval([s["id"] for s in relevant])

        return {
            "messages": [hint_msg],
            "retrieved_skills": [s["id"] for s in relevant],
        }

    # ═══════════════════════════════════════════════════
    #  节点 4/5：planner_node — 多步骤计划生成
    # ═══════════════════════════════════════════════════

    def planner_node(self, state: AgentState) -> dict:
        """在有技能可用时，为多步骤任务生成结构化执行计划（Codex 做法）

        触发条件：state 中有 retrieved_skills（无论来自 auto_retrieve 还是 retrieve_skills）
        检测到多步骤任务时，使用 LLM 生成执行计划并注入到 AgentState。
        计划会在后续 agent_node 的系统提示词中显示，引导 LLM 按步骤执行，
        避免"中间步骤停止"问题。
        """
        plan = state.get("plan", "")

        # 如果已有计划，不重复生成
        if plan:
            return {}

        # 获取可用技能列表（优先从 LLM 工具调用获取，其次从 auto_retrieve 获取）
        retrieved_skills = []

        # 路径 1: LLM 主动调用了 retrieve_skills
        for msg in reversed(state["messages"]):
            if msg.type == "tool" and getattr(msg, "name", "") == "retrieve_skills":
                match = re.search(r"已检索到 \d+ 个相关技能[：:]\s*(.+)", msg.content)
                if match:
                    retrieved_skills = [s.strip() for s in match.group(1).split(",")]
                break
            elif msg.type == "tool":
                break
            elif hasattr(msg, "tool_calls") and msg.tool_calls:
                break

        # 路径 2: auto_retrieve 注入的技能（state 中的 retrieved_skills）
        if not retrieved_skills:
            retrieved_skills = state.get("retrieved_skills", [])

        if not retrieved_skills:
            return {}

        # 已有 subtasks 时不再重新生成，保留 post_tools 更新的进度
        if state.get("subtasks"):
            return {}

        # 获取用户原始请求（取最新一条，与 auto_retrieve_node 一致）
        user_query = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break

        if not user_query:
            return {}

        # 始终生成执行计划（单步骤任务会生成 1-2 步，多步骤任务生成 3-6 步）
        plan_prompt = f"""基于用户请求和检索到的技能，制定一个简洁的执行计划。

用户请求: {user_query}
可用技能: {', '.join(retrieved_skills)}

请根据任务复杂度输出 1-6 个步骤的执行计划：
- 简单查询/信息检索 → 1 步
- 需要执行代码但单一操作 → 1-2 步
- 多步骤分析/Pipeline → 3-6 步

每步一行，格式：1. [步骤类型] 简要描述要做什么

步骤类型：查询/检索、数据准备、建库、搜索/比对、分析计算、可视化、结果解读

只输出计划，不要解释。"""

        try:
            plan_response = self.llm_raw.invoke([HumanMessage(content=plan_prompt)])
            plan = plan_response.content
            logger.info("Planner 生成执行计划:\n%s", plan)
            
            # 解析步骤以初始化子任务状态
            subtasks = []
            for line in plan.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    subtasks.append({"description": line, "status": "pending"})

            # max_iterations 为用户配置的硬上限，Planner 不再动态扩容
                    
        except Exception as e:
            logger.warning("Planner 生成失败: %s", e)
            return {}

        return {"plan": plan, "subtasks": subtasks}

    # ═══════════════════════════════════════════════════
    #  路由函数（决定图的边走向）
    # ═══════════════════════════════════════════════════

    def route_agent(self, state: AgentState) -> str:
        """Agent 节点路由：LLM 发出 tool_calls → 进入 tools，否则结束本轮。"""
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return END

    def route_tools(self, state: AgentState) -> str:
        """Tools 节点路由：未达迭代上限 → 回到 agent 继续决策，否则结束。"""
        iteration = state.get("iteration", 0)
        if iteration >= self.max_iterations:
            # 迭代耗尽且 force_no_tools 已生效 → LLM 已生成总结，正常结束
            if state.get("force_no_tools", False):
                return END
            # 二次软着陆兜底：迭代耗尽但 force_no_tools 未设置
            # 路由回 agent，agent_node 会用 llm_raw 生成总结回复
            logger.warning(
                "route_tools 软着陆兜底 | iteration=%d/%d | 强制路由回 agent 生成总结",
                iteration, self.max_iterations,
            )
            return "agent"
        return "agent"

    def route_post_tools(self, state: AgentState) -> str:
        """Post-tools 节点路由：已有计划 → 直接回 agent，否则走 planner 生成计划。"""
        # 如果已有执行计划，跳过 planner 节点（避免浪费 LLM 调用）
        if state.get("plan"):
            return "agent"
        return "planner"

    # ═══════════════════════════════════════════════════
    #  LangGraph 图构建
    # ═══════════════════════════════════════════════════

    def _build_graph(self):
        """构建 LangGraph 有向图：5 个节点 + 3 个条件路由 + MemorySaver 检查点。

        图的完整拓扑：
            auto_retrieve ──→ agent ──→ tools ──→ post_tools ──→ planner ──→ agent/END
                              │                                    │
                              └── END（无 tool_calls）              └── agent（已有 plan）
        """
        # LangGraph 流程：auto_retrieve 自动检索 → agent 决策 → tools 执行
        # → post_tools 修正状态 → planner 生成/维持计划 → 回到 agent/结束。
        builder = StateGraph(AgentState)

        builder.add_node("auto_retrieve", self.auto_retrieve_node)
        builder.add_node("agent", self.agent_node)
        builder.add_node("tools", self.tool_node)
        builder.add_node("post_tools", self.post_tools_node)
        builder.add_node("planner", self.planner_node)

        # ★ 入口改为 auto_retrieve（系统强制检索），而非 agent（LLM 决策）
        builder.set_entry_point("auto_retrieve")

        # ★ auto_retrieve → agent（无条件边，确保自动检索先于 LLM）
        builder.add_edge("auto_retrieve", "agent")

        builder.add_conditional_edges("agent", self.route_agent, {
            "tools": "tools",
            END: END,
        })
        builder.add_edge("tools", "post_tools")
        builder.add_conditional_edges("post_tools", self.route_post_tools, {
            "planner": "planner",
            "agent": "agent",
        })
        builder.add_conditional_edges("planner", self.route_tools, {
            "agent": "agent",
            END: END,
        })

        # MemorySaver：支持多轮对话状态持久化（Claude Code 做法）
        # 同一 thread_id 下的 messages 会跨轮次累积，Agent 拥有"记忆"。
        checkpointer = MemorySaver() if MemorySaver is not None else None
        return builder.compile(checkpointer=checkpointer)

    # ═══════════════════════════════════════════════════
    #  对外接口：stream() 流式执行 / run() 同步执行
    # ═══════════════════════════════════════════════════

    def stream(self, user_input: str, thread_id: str = ""):
        """流式执行 Agent，支持多轮对话（通过 thread_id 持久化状态）

        使用 MemorySaver checkpointer，同一 thread_id 下的 messages
        会跨轮次累积，实现真正的多轮对话"记忆"。
        每次调用只重置执行状态（iteration 等），保留对话历史。
        """
        config = {"configurable": {"thread_id": thread_id}} if thread_id else {}
        for event in self.graph.stream(
            {"messages": [HumanMessage(content=user_input)],
             "selected_skill": "", "tool_status": "", "iteration": 0,
             "consecutive_same_result": 0, "consecutive_empty_result": 0, "last_execute_code": "",
             "force_no_tools": False, "plan": "", "subtasks": [],
             "retrieved_skills": [], "auto_resume_count": 0,
             "recent_code_window": []},
            config=config,
            stream_mode="updates",
        ):
            yield event

    def run(self, user_input: str, thread_id: str = "") -> str:
        """同步执行 Agent，支持多轮对话"""
        config = {"configurable": {"thread_id": thread_id}} if thread_id else {}
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=user_input)],
             "selected_skill": "", "tool_status": "", "iteration": 0,
             "consecutive_same_result": 0, "consecutive_empty_result": 0, "last_execute_code": "",
             "force_no_tools": False, "plan": "", "subtasks": [],
             "retrieved_skills": [], "auto_resume_count": 0,
             "recent_code_window": []},
            config=config,
        )
        return result["messages"][-1].content
