# OpenClaw Medical Skills — 医疗 AI 智能体平台

<div align="center">

[![技能数量](https://img.shields.io/badge/技能数量-869-brightgreen?style=for-the-badge)](https://github.com/1695963547/OpenClaw-Medical-Skills/tree/main/skills)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-ReAct-orange?style=for-the-badge)](https://github.com/langchain-ai/langgraph)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-向量检索-purple?style=for-the-badge)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/License-MIT-gray?style=for-the-badge)](LICENSE)

**基于 LangGraph 构建的自主医疗 AI 智能体平台，集成 869 个生物医学技能，具备语义检索、多层错误容错与安全代码执行能力。**

*临床医学 · 基因组学 · 药物发现 · 生物信息学 · 医疗器械*

**[English](README_en.md) | 中文**

</div>

---

## 项目背景

大语言模型在医学场景中面临两个核心痛点：

> **痛点一：专业领域知识调用难**
> 通用 LLM 没有与真实医学数据库（PubMed、ClinicalTrials.gov、ChEMBL 等）的连接能力，只能依赖训练数据中的静态知识作答，无法完成需要实时查询的临床与科研任务。

> **痛点二：推理链不稳定**
> 面对复杂多步骤的生物信息分析任务（如 RNA-seq 流程、变异注释、蛋白质结构预测），通用 LLM 容易出现幻觉、步骤遗漏、代码执行失败等问题，缺乏系统性的错误自愈机制。

本项目是在企业 AGI 战略方向上**从 0 到 1** 自研的医疗大模型 Agent 平台，目标是通过 LangGraph 编排框架 + 869 个专业技能库，从根本上解决上述两个痛点，实现"会思考、能执行、懂医学"的自主 AI 智能体。

---

## 项目简介

OpenClaw Medical Skills 将通用大语言模型转化为专业医学与科研研究助手。平台将 **869 个生物医学技能**（涵盖临床工作流、基因组学、药物发现、生物信息学等领域）集成到统一的 LangGraph 智能体框架中。

Agent 能够自主检索相关技能、生成多步骤执行计划、在沙箱中执行代码，并从错误中自愈——全程无需人工干预。

### 能力对比

| 不使用本平台 | 使用本平台后 |
|---|---|
| 对医学问题的通用回答 | 真实查询 PubMed / ClinicalTrials.gov / FDA |
| 无生物信息学能力 | RNA-seq、scRNA-seq、GWAS、变异注释流程 |
| 无药物情报 | ChEMBL、DrugBank、DDI 预测、药物警戒 |
| 无临床文档 | SOAP 记录、出院小结、预先授权决策 |
| 无基因组学支持 | VCF 注释、ACMG 分类、多基因风险评分 |
| 无法规指导 | FDA、CE 认证、IEC 62304、ISO 14971 合规 |

---

## 核心特性

| 特性 | 说明 |
|------|------|
| 🧠 **自主技能检索** | 对话前自动语义检索最相关技能，无需用户手动指定，降低使用门槛 |
| 📋 **多步骤计划生成** | 复杂任务自动分解为结构化子步骤，实时展示执行进度 |
| 🛡️ **5 层错误容错** | 覆盖 13 类错误场景，API 异常下 Agent 自动切换策略保持稳定运行 |
| 🔒 **安全代码执行** | AST 静态分析 + 路径/命令白名单 + per-session 虚拟环境隔离 |
| 💬 **多轮对话记忆** | MemorySaver 跨轮次持久化上下文，同一会话内 Agent 具备"记忆" |
| 🔄 **卡死软着陆** | Stuck Detector 检测循环后触发降级响应，避免无限重试耗尽 Token |
| 📊 **自动化评测** | 内置 LAB-Bench / TRQA Benchmark，支持断点续跑和 LLM-as-Judge 评分 |
| 🔌 **多模型兼容** | 支持 DeepSeek V3/V4、GLM-4.6 及任意 OpenAI 兼容 API，一键切换 |

---

## 架构设计

### 整体架构

智能体基于 **LangGraph 5 节点状态图**（含条件路由）构建：

```
┌──────────────────────────────────────────────────────────┐
│                   LangGraph 有向状态图                     │
│                                                          │
│  ┌──────────────┐    ┌──────────┐    ┌──────────┐       │
│  │ auto_retrieve │───→│  agent   │───→│  tools   │       │
│  │  (语义检索    │    │ (LLM决策) │    │ (代码执行) │      │
│  │   预加载)     │    │          │    │          │       │
│  └──────────────┘    └──────────┘    └──────────┘       │
│                           ↑    │           │             │
│                           │    ↓           ↓             │
│                      ┌──────────┐    ┌────────────┐     │
│                      │ planner  │←───│ post_tools  │     │
│                      │ (计划生成) │    │ (错误修正)  │     │
│                      └──────────┘    └────────────┘     │
│                           │                              │
│                           ↓                              │
│                      agent（循环）或 END                  │
└──────────────────────────────────────────────────────────┘
```

### 节点职责

| 节点 | 职责 |
|------|------|
| **auto_retrieve** | LLM 决策前，通过 ChromaDB 语义匹配预加载相关技能 |
| **agent** | LLM 决策核心——决定调用工具还是直接回复 |
| **tools** | 执行工具调用：`retrieve_skills` / `read_file` / `execute_code` / `update_task_status` |
| **post_tools** | 错误分类（13 种）、修正建议注入、卡死检测、子任务进度追踪 |
| **planner** | 为复杂任务生成结构化多步骤执行计划 |

### 关键设计决策

**1. 为什么用 LangGraph 而非简单链式调用？**

简单链（LangChain Chains）在医学任务中面临致命缺陷：复杂任务需要动态决策（"是否需要再查一次数据库"、"代码报错了怎么办"），而链式调用是静态的。LangGraph 的有状态图允许 Agent 在执行过程中根据结果动态路由，实现真正的自主循环推理。

**2. 为什么在 LLM 决策之前先执行 auto_retrieve？**

如果等 LLM 自己决定"要不要检索技能"，召回时机会滞后，且 LLM 在没有上下文的情况下往往选择直接回答而非调用工具。`auto_retrieve` 节点在对话一开始就把最相关的技能注入 System Prompt，大幅提升 LLM 选择正确工具的概率，这是对 RAG 思路的主动式改进。

**3. 为什么设计 5 层容错而非简单重试？**

医学 API（NCBI、ChEMBL、UniProt 等）故障模式千差万别：认证失败、Schema 变更、包依赖缺失都需要不同的处理策略。简单重试只解决瞬时抖动，无法应对结构性错误。分层容错通过错误记忆避免重复踩坑，通过策略升级逐步降级，最大化任务完成率。

**4. 为什么需要独立的 planner 节点？**

对于单步问答，`agent` 节点直接回复即可。但对于"分析这组测序数据"这类需要 5-10 个步骤的复杂任务，没有显式计划会导致 Agent 在执行过程中迷失方向，出现步骤遗漏或循环。`planner` 生成的结构化子任务列表相当于给 Agent 一份"执行合同"，并通过 `post_tools` 实时追踪完成状态。

### 关键设计特性

- **MemorySaver 多轮持久化** — 同一 `thread_id` 跨轮次保留对话上下文
- **Stuck Detector 软着陆** — 循环检测触发 `force_no_tools` 降级响应
- **动态迭代上限** — Planner 生成 N 步计划时，自动设置 `max(20, N×4)` 轮迭代
- **结构化错误分类** — 13 种错误类型枚举，配套策略升级机制

---

## 技能检索系统

检索系统采用**双路径架构**，确保技能召回的稳定性：

| 检索路径 | 方法 | 适用场景 |
|---------|------|---------|
| **语义检索** | ChromaDB + sentence-transformers | 自然语言查询 |
| **ID 兜底** | 技能 ID 直接查找 | 语义召回失效时 |

- **869 个技能**的描述信息已索引入 ChromaDB
- **余弦距离阈值**过滤，排除不相关召回
- **Top-K 可配置**（默认召回 8 个技能）

---

## 多层错误容错中间件

**5 层容错框架**确保 API 异常下 Agent 仍稳定运行：

```
第 1 层：参数归一化          → 修正格式异常的工具参数
第 2 层：错误记忆            → 记录失败历史，避免重复犯错
第 3 层：工具替换            → 切换至备用工具
第 4 层：知识库降级          → 回退至缓存响应
第 5 层：GraphQL Schema 自愈  → 自动修复损坏的 API Schema
```

覆盖 **13 类错误**：脚本错误、导入失败、包安装失败、API 认证错误等。

---

## 安全代码执行沙箱

平台内置多语言代码执行沙箱，配备完整安全控制：

| 特性 | 实现方式 |
|-----|---------|
| **支持语言** | Python、Bash、R、JavaScript |
| **静态分析** | AST 解析 + import 白名单校验 |
| **路径限制** | 文件系统访问白名单 |
| **命令限制** | Shell 命令白名单 |
| **会话隔离** | per-session 独立虚拟环境 |
| **依赖管理** | venv/conda 双层自动安装 |

---

## 自动化 Benchmark 评估

平台内置并发 Benchmark 框架，支持模型迭代验证：

- **评测数据集**：LAB-Bench（SuppQA、LitQA2、SeqQA、DbQA、ProtocolQA）、TRQA（200 道核心问题）
- **断点续跑**：Checkpoint 机制保存/恢复评测进度
- **自动评分**：LLM-as-Judge 多维评分
- **并发执行**：多线程并行加速评测

---

## 技能总览

| 分类 | 数量 | 技能亮点 |
|-----|------|---------|
| 通用与核心工具 | 10 | 浏览器、搜索、文档处理、开发者工作流 |
| 临床医学 | 119 | 临床报告、CDS、肿瘤学、医学影像、医疗 AI |
| 科学数据库 | 43 | 基因组、蛋白质、药物数据库，生物医学知识检索 |
| 生物信息学 | 239 | 变异分析、测序 QC、差异表达、通路分析、单细胞 |
| 多组学与计算生物学 | 59 | 单细胞/空间组学、蛋白质组学、化学信息学、蛋白质设计 |
| ClawBio 流程 | 21 | scRNA、GWAS、祖先分析、结构生物学流程编排 |
| BioOS 扩展套件 | 285 | 肿瘤学、免疫学、临床 AI、基础设施 |
| 数据科学与工具 | 93 | 统计分析、可视化、自动化、模拟计算 |
| **合计** | **869** | |

---

## 快速开始

### 环境要求

- Python 3.10+
- Git

### 安装

```bash
# 克隆仓库
git clone https://github.com/1695963547/OpenClaw-Medical-Skills.git
cd OpenClaw-Medical-Skills

# 安装依赖（二选一）
pip install -r requirements.txt

# 或使用 conda
conda env create -f environment.yml
conda activate openclaw-medical
```

### 配置 LLM

创建 `llm_local_config.py`（参考 `llm_config.example.py`）：

```python
LLM_CONFIG = {
    "model": "deepseek-v4-flash",              # 模型名称
    "base_url": "https://api.deepseek.com",    # API 地址
    "api_key": "your-api-key-here",            # API 密钥
    "skill_top_k": 8,                          # 检索技能数量
    "max_iterations": 20,                       # Agent 最大迭代次数
}
```

**兼容模型**：DeepSeek V3/V4、GLM-4.6，以及任意 OpenAI 兼容 API。

### 初始化向量索引

```bash
# 首次运行时构建 ChromaDB 索引（根据 skill_registry.json）
python -c "from src.skill_retriever import SkillRetriever; SkillRetriever('skill_registry.json')"
```

### 启动

```bash
python main.py
```

```
Medical Skills Agent 就绪。输入 exit 退出。

> 华法林与哪些药物存在相互作用？
👤 User: 华法林与哪些药物存在相互作用？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 执行计划
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ⬜ Step 1: 检索药物相互作用相关技能
  ⬜ Step 2: 查询药物相互作用数据库
  ⬜ Step 3: 分析并汇总结果
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅ [1/3] done: 检索药物相互作用相关技能
  ✅ [2/3] done: 查询药物相互作用数据库
  ✅ [3/3] done: 分析并汇总结果

🤖 Agent: 华法林的主要药物相互作用包括...
```

---

## 项目结构

```
OpenClaw-Medical-Skills/
├── main.py                    # 入口：CLI 对话循环、日志、会话管理
├── skill_registry.json        # 869 个技能元数据索引
├── src/
│   ├── agent.py               # LangGraph ReAct 智能体（5 节点状态图）
│   ├── skill_retriever.py     # ChromaDB 语义检索器
│   ├── skill_context.py       # 技能上下文组装器
│   ├── tools.py               # 4 个工具函数（检索/读取/执行/状态更新）
│   ├── code_executor.py       # 多语言沙箱执行器
│   ├── conda_manager.py       # Conda 环境管理
│   ├── llm_factory.py         # LLM 实例工厂
│   ├── skill_stats.py         # 技能使用统计追踪
│   └── middleware/            # 错误容错中间件
│       ├── error_memory.py        # 错误记忆 + 策略升级器
│       ├── resilient_executor.py  # 弹性执行包装器
│       ├── schema_healer.py       # GraphQL Schema 自愈
│       └── tool_param_adapter.py  # 参数归一化适配器
├── skills/                    # 869 个技能模块（SKILL.md 文件）
├── scripts/                   # Benchmark 与工具脚本
│   ├── benchmark_test.py      # 并发 Benchmark 运行器
│   ├── build_registry.py      # 技能注册表构建器
│   ├── skill_audit.py         # 技能校验审计
│   └── validate_skill.py      # 单技能校验
├── TestQuestion/              # 评测数据集（LAB-Bench、TRQA）
└── chroma_db/                 # 预构建向量索引
```

---

## 技术栈

| 层级 | 技术 |
|-----|------|
| **智能体框架** | LangGraph（ReAct 状态图） |
| **向量数据库** | ChromaDB |
| **嵌入模型** | sentence-transformers |
| **LLM 接口** | LangChain + ChatOpenAI（兼容 OpenAI API） |
| **代码执行** | subprocess + AST 分析 + venv/conda |
| **可观测性** | LangSmith 链路追踪 + JSONL 日志 |
| **开发语言** | Python 3.10+ |

---

## 致谢

本项目在 [FreedomIntelligence](https://github.com/FreedomIntelligence) 的 [OpenClaw Medical Skills](https://github.com/FreedomIntelligence/OpenClaw-Medical-Skills) 技能库基础上进行了深度二次开发，构建了完整的 LangGraph 智能体框架、语义检索系统、错误容错中间件和安全代码执行沙箱。

技能来源于 12 个以上开源仓库，完整致谢见原始仓库。

## 开源协议

MIT License — 详见 [LICENSE](LICENSE)
