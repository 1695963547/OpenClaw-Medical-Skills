# OpenClaw Medical Skills — LangGraph Agent 项目交接文档

> **最后更新**：2026-06-02  
> **运行平台**：Windows · conda `longxia` · Python 3.11  
> **仓库地址**：[FreedomIntelligence/OpenClaw-Medical-Skills](https://github.com/FreedomIntelligence/OpenClaw-Medical-Skills)

---

## 目录

- [一、项目概述](#一项目概述)
- [二、环境搭建](#二环境搭建)
- [三、配置文件](#三配置文件)
- [四、启动与使用](#四启动与使用)
- [五、核心架构](#五核心架构)
- [六、源代码文件索引](#六源代码文件索引)
- [七、中间件层（Middleware）](#七中间件层middleware)
- [八、工具系统](#八工具系统)
- [九、基准测试](#九基准测试)
- [十、依赖管理说明](#十依赖管理说明)
- [十一、已知问题与注意事项](#十一已知问题与注意事项)
- [十二、已完成的关键改进](#十二已完成的关键改进)
- [十三、建议后续方向](#十三建议后续方向)
- [十四、交接文件清单](#十四交接文件清单)

---

## 一、项目概述

本项目将 **869 个医学/生物信息技能** 接入 LangGraph，构建一个自主的医疗 AI 智能体助手。

Agent 能够根据用户的自然语言问题，自动完成：

```
语义检索匹配技能 → 生成执行计划 → 调用工具执行 → 错误修正 → 输出结构化结果
```

| 能力维度 | 说明 |
|---------|------|
| 技能库 | 869 个技能（临床、基因组、药物发现、生物信息学、医疗器械等） |
| 检索 | ChromaDB + sentence-transformers（paraphrase-multilingual-MiniLM-L12-v2）语义匹配 |
| 决策 | LangGraph ReAct 状态图 + LLM 动态规划 |
| 执行 | 隔离沙箱（per-session venv / conda 环境）执行 Python / Bash / R 代码 |
| 容错 | 中间件层（参数归一化、错误记忆、Schema 自愈、韧性执行） |

---

## 二、环境搭建

### 方式一：Conda 完整还原（推荐，100% 复现 longxia 环境）

```powershell
# 1. 用导出的 environment.yml 创建环境（含全部 921 个包）
conda env create -f environment.yml -n longxia

# 2. 激活环境
conda activate longxia

# 3. 补装可能新增的 pip 依赖
pip install -r requirements.txt
```

### 方式二：轻量安装（已有 Python 3.11 环境）

```powershell
# 1. 创建 conda 环境
conda create -n longxia python=3.11 -y
conda activate longxia

# 2. 安装 pip 依赖
pip install -r requirements.txt
```

### 验证安装

```powershell
python -c "import langgraph, langchain_openai, chromadb; print('OK')"
```

---

## 三、配置文件

### 3.1 llm_local_config.py（核心配置，必须修改）

项目根目录下的 `llm_local_config.py` 是唯一的配置文件：

```python
LLM_API_KEY = "your-api-key"                          # API 密钥
LLM_BASE_URL = "http://your-gateway/api/ai-gateway/v1" # API 网关地址（兼容 OpenAI 协议）
LLM_MODEL = "Gemini-3.1-pro-preview"                   # 模型名称
LLM_PROFILE = "custom"                                  # 配置档案
SKILL_TOP_K = "3"                                      # 每次检索返回的技能数
MAX_ITERATIONS = "40"                                   # Agent 最大迭代轮数
MAX_DISPLAY_LENGTH = "0"                               # 终端显示最大字符数（0 = 不截断）
SENTENCE_TRANSFORMER_MODEL = r"E:\paraphrase-multilingual-MiniLM-L12-v2"  # 嵌入模型路径
LANGSMITH_API_KEY = ""                                  # LangSmith 追踪密钥（可选）
LANGSMITH_PROJECT = "869skills"                         # LangSmith 项目名
```

### 3.2 配置参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `LLM_API_KEY` | — | API 密钥（支持 OpenAI 官方或兼容网关） |
| `LLM_BASE_URL` | — | API 网关地址 |
| `LLM_MODEL` | `Gemini-3.1-pro-preview` | 可切换为 `DeepSeek-V3.2-ALi` 等 |
| `SKILL_TOP_K` | `3` | 检索返回的技能数量 |
| `MAX_ITERATIONS` | `40` | Agent 单次任务最大迭代次数 |
| `MAX_DISPLAY_LENGTH` | `0` | 终端输出截断长度，0 表示完整显示 |
| `SENTENCE_TRANSFORMER_MODEL` | 本地路径 | 嵌入模型路径，见下方重要提示 |

> **⚠️ 重要 — 嵌入模型路径**  
> `SENTENCE_TRANSFORMER_MODEL` 当前指向本地绝对路径 `E:\paraphrase-multilingual-MiniLM-L12-v2`。  
> 交接后需要：  
> ① 将 `paraphrase-multilingual-MiniLM-L12-v2` 模型文件夹拷贝到你的机器  
> ② 修改此路径指向新位置  
> ③ 或者改为 HuggingFace 模型名（会自动下载）：`paraphrase-multilingual-MiniLM-L12-v2`

### 3.3 环境变量覆盖

`llm_local_config.py` 中的每个参数都支持环境变量覆盖（优先级更高）：

```powershell
python main.py
```

---

## 四、启动与使用

### 4.1 首次启动前：构建技能索引

```powershell
python scripts/build_registry.py
```

此命令扫描 `skills/` 目录，生成 `skill_registry.json`。  
首次启动 Agent 时会自动构建 `chroma_db/` 向量数据库。

### 4.2 启动 Agent

```powershell
conda activate longxia
python main.py
```

启动后进入交互式命令行循环，Agent 对每轮问题自动执行：检索 → 计划 → 执行 → 回复。  
输入 `quit` 或 `exit` 退出（退出时自动清理所有临时沙箱环境）。

### 4.3 测试示例

```
# 纯知识型（直接回答）
HIPAA 审计日志需要记录哪些信息？

# 脚本调用型
使用 arboreto 从 expr.tsv 推断基因调控网络

# API 代码生成型
帮我写一个 BLAST 搜索序列 ATGGTG 的代码并执行
```

### 4.4 日志

- 控制台实时输出 + 文件双通道日志（`logs/` 目录）
- 每轮请求落盘 JSONL 记录，便于回放和统计分析

---

## 五、核心架构

### 5.1 LangGraph 有向图

```
┌────────────────────────────────────────────────────────────────┐
│                    LangGraph 有向图流程                          │
│                                                                │
│  ┌──────────────┐    ┌──────────┐    ┌──────────┐             │
│  │ auto_retrieve │───→│  agent   │───→│  tools   │             │
│  │  (自动检索)   │    │ (LLM决策) │    │ (工具执行) │             │
│  └──────────────┘    └──────────┘    └──────────┘             │
│                           ↑    │           │                   │
│                           │    ↓           ↓                   │
│                      ┌──────────┐    ┌────────────┐            │
│                      │ planner  │←───│ post_tools  │            │
│                      │ (计划生成) │    │ (后处理修正) │            │
│                      └──────────┘    └────────────┘            │
│                           │                                    │
│                           ↓                                    │
│                      agent (循环) 或 END                        │
└────────────────────────────────────────────────────────────────┘
```

### 5.2 五个核心节点

| 节点 | 文件位置 | 职责 | 关键特性 |
|------|---------|------|----------|
| `auto_retrieve_node` | agent.py | 系统级自动检索，ChromaDB 语义匹配预加载技能 | 每轮对话首次进入时触发 |
| `agent_node` | agent.py | LLM 决策核心，决定调用工具还是直接回复 | 含上下文截断保护（B 方案） |
| `tools` (ToolNode) | tools.py | 执行 LLM 请求的工具调用 | 4 个工具（详见第八章） |
| `post_tools_node` | agent.py | 解析执行结果、分类错误、注入修正建议 | 含输出截断（C 方案）+ Stuck 检测 + 软着陆 |
| `planner_node` | agent.py | 为多步骤任务生成结构化执行计划 | 动态迭代上限 = max(20, plan_steps × 4) |

### 5.3 关键设计特性

| 特性 | 说明 |
|------|------|
| **MemorySaver 多轮对话** | 同一 thread_id 跨轮记忆上下文 |
| **Stuck Detector** | 检测 Agent 陷入死循环时触发软着陆 |
| **软着陆（force_no_tools）** | 强制 Agent 基于已有信息生成总结回复，而非无限循环 |
| **上下文截断（B+C 方案）** | B：发送 LLM 前截断历史 ToolMessage（head 2K + tail 500）；C：写 state 时截断工具输出 |
| **13 种错误分类** | ErrorType 枚举覆盖各类故障场景 |
| **3 层依赖自动安装** | execute_code 内部自动检测并安装缺失 Python 包 |

---

## 六、源代码文件索引

### 6.1 核心模块（src/）

```
src/
├── agent.py              (1703 行) LangGraph ReAct 状态图编排，5 节点核心
├── tools.py              (700 行)  4 个工具定义 + 3 层依赖自动安装
├── code_executor.py      (1445 行) Python/Bash/R/JS 安全沙箱执行器
├── conda_manager.py      (442 行)  Conda 环境管理（模板克隆 + per-session 隔离）
├── llm_factory.py        (237 行)  LLM 构建工厂（含 API 网关错误 patch）
├── skill_retriever.py    (137 行)  ChromaDB 语义检索
├── skill_context.py      (122 行)  技能上下文骨架加载器（SKILL.md + 文件清单）
├── skill_stats.py        (180 行)  技能使用频率统计
└── middleware/                     稳健执行中间层（见第七章）
    ├── __init__.py
    ├── error_memory.py   (264 行)  错误记忆与策略阶梯
    ├── schema_healer.py  (355 行)  GraphQL Schema 自愈层
    ├── resilient_executor.py (346 行) 平台/路径/网络韧性
    ├── tool_param_adapter.py (190 行) 工具参数归一化
    └── prompts/
        ├── __init__.py             系统提示加载器
        └── system_addendum.md      System Prompt 增强约束
```

### 6.2 运维脚本（scripts/）

```
scripts/
├── build_registry.py     (90 行)   扫描 skills/ 生成 skill_registry.json
├── benchmark_test.py     (1422 行) 基准测试（参数化 + 并发 + 断点续跑）
├── skill_audit.py        (153 行)  技能目录完整性审计
├── skill_runtime_smoke.py (337 行) 技能运行时冒烟测试
└── validate_skill.py     (99 行)   单个技能格式校验
```

### 6.3 入口与配置

```
main.py                   (428 行)  命令行交互入口（日志、LLM 加载、对话循环）
llm_local_config.py       (27 行)   项目本地配置文件
requirements.txt          (87 行)   pip 依赖清单（分层结构）
environment.yml           (434 行)  conda 完整环境导出
openclaw.plugin.json      (878 行)  OpenClaw 插件描述（869 个技能路径）
```

### 6.4 数据与缓存

```
skills/                   869 个技能目录（每个含 SKILL.md）
skill_registry.json       技能元数据缓存（build_registry.py 生成）
skill_stats.json          技能使用统计（运行时自动生成）
chroma_db/                ChromaDB 向量数据库（首次启动自动构建）
workspace/                execute_code 安全沙箱目录
fda_cache/                FDA 数据库查询缓存
data/                     示例数据集（如 pbmc3k_raw.h5ad）
TestQuestion/             基准测试题库（MD 格式）
logs/                     运行日志 + 测试报告
```

---

## 七、中间件层（Middleware）

`src/middleware/` 是 Agent 的「防错缓冲层」，与具体 skill / API 解耦，新增任何技能自动受益。

| 模块 | 解决的问题 | 核心机制 | 状态 |
|------|-----------|----------|------|
| `tool_param_adapter.py` | LLM 调用工具时使用幻觉参数名 | 通用别名表 + kwargs 归一化 + 失败重试 | ✅ 已实施 |
| `error_memory.py` | 同一错误反复出现导致失控 | 错误指纹追踪 + 策略阶梯（重试→换工具→切知识库→终止） | ✅ 已实施 |
| `schema_healer.py` | GraphQL 字段重命名导致 400 错误 | 解析 "Did you mean?" 提示 + 查询重写 | ✅ 已实施 |
| `resilient_executor.py` | 沙箱路径断裂 / Windows bash 缺失 / API 超时 | 平台适配 + 路径注入 + 网络韧性 preamble | ✅ 已实施 |
| `prompts/system_addendum.md` | Agent 行为规范约束 | 启动时拼接到 system message 末尾 | ✅ 已实施 |

---

## 八、工具系统

Agent 决策时可调用 4 个工具：

| 工具名 | 功能 | 实现位置 |
|--------|------|---------|
| `retrieve_skills` | 语义检索 869 个生物医学技能（ChromaDB） | tools.py |
| `read_file` | 读取技能目录下的 SKILL.md 或参考文件 | tools.py |
| `execute_code` | 在隔离 venv 中执行 Python / Bash / R / JS 代码 | tools.py → code_executor.py |
| `update_task_status` | 更新多步骤计划的子任务状态（pending/in_progress/completed/failed） | tools.py |

### 依赖自动安装（3 层防御）

```
第 1 层：LLM 在代码中写 `# pip install xxx` → CodeExecutor 自动预安装
第 2 层：post_tools 检测到 ModuleNotFoundError → auto_install_skill_deps() 批量安装
第 3 层：从代码 import 语句解析包名 → auto_install_from_code() 兜底安装
```

### 代码执行安全机制

- Windows bash 命令自动适配（Git Bash / WSL / PowerShell）
- 文件路径白名单（仅允许 workspace 目录读写）
- subprocess 调用工具白名单（仅允许生物信息学相关 CLI 工具）
- AST 静态分析检测文件路径越权和命令越权
- 虚拟环境隔离（venv + conda 双层架构）

---

## 九、基准测试

`scripts/benchmark_test.py` 支持对题库中的问题进行自动化批量测试。

### 9.1 常用命令

```powershell
# 快速验证（2 条 × 1 次）
python scripts/benchmark_test.py --questions 1,2 --runs 1

# 指定题库 + 采样测试
python scripts/benchmark_test.py --test-file "TestQuestion/TRQA测试案例（200条核心差异化数据）.md" --questions 1,10,50,100 --runs 1

# 全量测试（200 条 × 1 次，3 并发）
python scripts/benchmark_test.py --test-file "TestQuestion/TRQA测试案例（200条核心差异化数据）.md" --runs 1 --workers 3

# 断点续跑（利用 checkpoint）
python scripts/benchmark_test.py --runs 1 --resume logs/benchmark_checkpoint_20260529_174354.json

# 跳过 LLM 评分（节省成本）
python scripts/benchmark_test.py --runs 3 --skip-judge
```

### 9.2 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--runs N` | 3 | 每题重复运行次数 |
| `--questions 1,2,3` | 全部 | 指定题号（逗号分隔） |
| `--workers N` | 3 | 并发线程数（1 = 串行） |
| `--resume FILE` | — | 从 checkpoint JSON 断点续跑 |
| `--skip-judge` | false | 跳过 LLM-as-Judge 评分 |
| `--test-file FILE` | 问题模板 | 指定测试题库 MD 文件路径 |

### 9.3 测试题库

```
TestQuestion/
├── 问题模板和衍生问题.md                          # 默认题库（10 题）
├── TRQA测试案例（200条核心差异化数据）.md          # 200 条核心测试集
├── LAB-Bench_SuppQA 测试集.md                     # LAB-Bench 补充问答
├── LAB-Bench_LitQA2 测试集.md                     # 文献问答
├── LAB-Bench_ProtocolQA 测试集.md                 # 实验方案问答
├── LAB-Bench_SeqQA 测试集.md                      # 序列问答
├── LAB-Bench_DbQA 测试集.md                       # 数据库问答
└── HumanityLastExam.md                            # 高难度测试
```

---

## 十、依赖管理说明

### 10.1 两层依赖体系

| 文件 | 用途 | 覆盖范围 |
|------|------|---------|
| `environment.yml` | conda 完整环境导出 | 全部 921 个包（含 Python 解释器、系统库） |
| `requirements.txt` | pip 依赖清单 | 16 个核心 pip 包 |

### 10.2 requirements.txt 分层结构

```
第 1 组 · 核心依赖（8 个）：代码直接 import，禁止移除
  langgraph, langchain-core, langchain-openai, openai
  chromadb, sentence-transformers, PyYAML, httpx

第 2 组 · 隐式依赖（6 个）：运行时必需，缺失会 ImportError
  pydantic, typing-extensions, tenacity, tiktoken, numpy, requests

第 3 组 · 可观测（1 个）：可选调试工具
  langsmith

第 4 组 · 兼容垫片：版本冲突时的回退方案（注释形式）
```

### 10.3 版本基线（longxia 实测 2026-06-02）

```
langchain-core 1.4.0    langgraph 1.2.1       chromadb 1.5.9
openai 2.38.0           sentence-transformers 5.5.1
pydantic 2.12.3         numpy 2.4.6           httpx 0.28.1
PyYAML 6.0.3            tiktoken 0.12.0       requests 2.33.1
tenacity 9.1.4          langsmith 0.8.3
```

---

## 十一、已知问题与注意事项

### 1. 嵌入模型路径（交接后必须修改）

`llm_local_config.py` 中 `SENTENCE_TRANSFORMER_MODEL` 使用 Windows 绝对路径。  
→ 需拷贝模型文件夹到新机器，或改为 HuggingFace 模型名自动下载。

### 2. API 网关地址

`LLM_BASE_URL` 当前指向内网网关 `http://221.5.60.136:30100/...`。  
→ 如果不在同一网络，需改为可用的 API 地址，或使用 OpenAI 官方：`https://api.openai.com/v1`。

### 3. Windows 环境适配

`code_executor.py` 内有 Windows 适配逻辑（bash 缺失处理等）。  
但如果技能脚本依赖 Linux 工具（如 samtools、bedtools），需要额外安装 WSL 或对应 Windows 版本。

### 4. ChromaDB 版本兼容

当前 longxia 使用 chromadb 1.5.9。如果遇到 API 不兼容，可 pin 到 `chromadb>=0.5.0,<1.0.0`。

### 5. Conda 环境管理

`src/conda_manager.py` 会在运行时创建 per-session conda 环境（模板克隆策略），如果目标机器没有 conda，会自动降级到 venv + pip 路径。

### 6. 可重新生成的文件

以下文件不需要手动维护，可自动重新生成：

- `skill_registry.json` → 运行 `python scripts/build_registry.py`
- `chroma_db/` → 首次启动 Agent 时自动构建
- `skill_stats.json` → 运行时自动更新
- `workspace/` → 运行时动态创建沙箱目录

---

## 十二、已完成的关键改进

### 1. B+C 上下文截断方案

解决了超长 ToolMessage 撑爆 LLM 上下文窗口的问题：
- **B 方案（读端压缩）**：`agent.py` 第 485-525 行，发送给 LLM 前截断历史消息（保留 head 2K + tail 500）
- **C 方案（写端截断）**：`agent.py` 第 1307+ 行，post_tools_node 写入 state 时截断超长工具输出

### 2. requirements.txt 分层重构

从 9 个裸依赖升级为 4 层结构（核心 / 隐式 / 可观测 / 兼容垫片），每组带中文注释和版本基线。

### 3. environment.yml 完整导出

longxia 环境全量 921 个包，可 100% 还原开发环境。

### 4. 中间件层

5 个独立模块：参数归一化、错误记忆、Schema 自愈、韧性执行、系统提示增强，与具体 skill 解耦。

### 5. 基准测试脚本

支持参数化（`--questions --runs --workers`）、断点续跑（`--resume`）、LLM-as-Judge 评分、多题库切换。

### 6. LLM 工厂

统一 LLM 构建入口，含 API 网关错误 patch（解决网关返回 `choices: null` 时错误信息被吞掉的问题），支持多模型切换。

---

## 十三、建议后续方向

| 方向 | 优先级 | 说明 |
|------|--------|------|
| 压缩替代截断 | 中 | B+C 目前用截断处理超长上下文，可升级为 LLM 摘要或关键词提取压缩，减少信息丢失 |
| 模型对比 | 中 | 当前使用 `Gemini-3.1-pro-preview`，可对比 `DeepSeek-V3.2-ALi`、`gpt-4o` 效果 |
| LangSmith 链路追踪 | 低 | 填入 `LANGSMITH_API_KEY` 即可启用，便于调试和分析 Agent 行为轨迹 |
| 嵌入模型云端化 | 低 | 当前依赖本地模型文件，可改用 HuggingFace Hub 自动下载 |
| 技能动态注册 | 低 | 当前 `skill_registry.json` 是静态扫描生成，可考虑热加载机制 |

---

## 十四、交接文件清单

| 文件 | 用途 | 是否必须 |
|------|------|----------|
| `requirements.txt` | pip 依赖清单 | ✅ 必须 |
| `environment.yml` | conda 完整环境 | ✅ 强烈推荐 |
| `llm_local_config.py` | LLM 配置（含密钥） | ⚠️ 需修改后使用 |
| `openclaw.plugin.json` | 869 个技能路径映射 | ✅ 必须 |
| `skill_registry.json` | 技能元数据缓存 | 可重新生成 |
| `chroma_db/` | 向量数据库 | 可重新生成 |
| `paraphrase-multilingual-MiniLM-L12-v2/` | 本地嵌入模型 | ⚠️ 需单独拷贝或改为在线模型名 |
| `skills/` | 869 个技能目录 | ✅ 必须 |
| `TestQuestion/` | 基准测试题库 | 按需保留 |
| `logs/` | 历史日志 | 可丢弃 |

---

> **快速启动检查清单**  
> ① 安装 conda 环境 → ② 修改 `llm_local_config.py` → ③ 确保嵌入模型可用 → ④ `python scripts/build_registry.py` → ⑤ `python main.py`
