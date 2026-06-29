# OpenClaw Medical Skills

<div align="center">

[![Skills Count](https://img.shields.io/badge/Skills-869-brightgreen?style=for-the-badge)](https://github.com/1695963547/OpenClaw-Medical-Skills/tree/main/skills)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-ReAct-orange?style=for-the-badge)](https://github.com/langchain-ai/langgraph)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Search-purple?style=for-the-badge)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/License-MIT-gray?style=for-the-badge)](LICENSE)

**An autonomous medical AI agent platform powered by LangGraph, integrating 869 biomedical skills with semantic retrieval, multi-layer error resilience, and safe code execution.**

*Clinical · Genomics · Drug Discovery · Bioinformatics · Medical Devices*

</div>

---

## Overview

OpenClaw Medical Skills is an autonomous medical AI agent platform that transforms a general-purpose LLM into a specialized medical and scientific research assistant. It integrates **869 curated biomedical skills** — covering clinical workflows, genomics, drug discovery, and bioinformatics — into a unified LangGraph-based agent framework.

The agent autonomously retrieves relevant skills via semantic search, generates multi-step execution plans, executes code in a sandboxed environment, and self-heals from errors — all within an interactive conversational interface.

### Key Capabilities

| Without This Platform | With This Platform |
|---|---|
| Generic AI responses about medicine | Real PubMed / ClinicalTrials.gov / FDA queries |
| No bioinformatics capability | RNA-seq, scRNA-seq, GWAS, variant calling pipelines |
| No drug intelligence | ChEMBL, DrugBank, DDI prediction, pharmacovigilance |
| No clinical documentation | SOAP notes, discharge summaries, prior auth decisions |
| No genomics support | VCF annotation, ACMG classification, PRS calculation |
| No regulatory guidance | FDA, CE mark, IEC 62304, ISO 14971 compliance |

---

## Architecture

The agent is built on a **LangGraph 5-node state graph** with conditional routing:

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph State Graph                     │
│                                                             │
│  ┌──────────────┐    ┌──────────┐    ┌──────────┐          │
│  │ auto_retrieve │───→│  agent   │───→│  tools   │          │
│  │  (semantic    │    │ (LLM     │    │ (code    │          │
│  │   retrieval)  │    │  decision)│   │  execution)│        │
│  └──────────────┘    └──────────┘    └──────────┘          │
│                           ↑    │           │                │
│                           │    ↓           ↓                │
│                      ┌──────────┐    ┌────────────┐        │
│                      │ planner  │←───│ post_tools  │        │
│                      │ (plan    │    │ (error      │        │
│                      │  generation)│  │  recovery) │        │
│                      └──────────┘    └────────────┘        │
│                           │                                 │
│                           ↓                                 │
│                      agent (loop) or END                   │
└─────────────────────────────────────────────────────────────┘
```

### Node Responsibilities

| Node | Function |
|------|----------|
| **auto_retrieve** | Pre-loads relevant skills via ChromaDB semantic search before LLM decision |
| **agent** | LLM decision core — decides whether to call tools or respond directly |
| **tools** | Executes tool calls: `retrieve_skills`, `read_file`, `execute_code`, `update_task_status` |
| **post_tools** | Error classification (13 types), correction injection, stuck detection, subtask tracking |
| **planner** | Generates structured multi-step execution plans for complex tasks |

### Design Features

- **MemorySaver** persistence for multi-turn conversations (same `thread_id` across turns)
- **Stuck Detector** with soft-landing mechanism (`force_no_tools` fallback)
- **Dynamic iteration limit** — Planner N steps → `max(20, N×4)` iterations
- **Structured error classification** — 13 error types with escalation strategies

---

## Skill Retrieval System

The retrieval system uses a **dual-path architecture** for robust skill matching:

| Path | Method | Use Case |
|------|--------|----------|
| **Semantic** | ChromaDB + sentence-transformers | Natural language queries |
| **ID Fallback** | Direct skill ID lookup | When semantic recall fails |

- **869 skills** indexed in ChromaDB with embedded descriptions
- **Cosine distance threshold** filtering to eliminate irrelevant recalls
- **Top-K** configurable retrieval (default: 8)

---

## Error Resilience Middleware

A **5-layer fault-tolerant framework** ensures the agent stays operational under API failures:

```
Layer 1: Parameter Normalization      → Fix malformed tool arguments
Layer 2: Error Memory                 → Record failures, avoid repeating
Layer 3: Tool Replacement             → Switch to alternative tools
Layer 4: Knowledge Base Degradation   → Fallback to cached responses
Layer 5: GraphQL Schema Self-Healing → Auto-repair broken API schemas
```

Covers **13 error categories** including script errors, import failures, package installation failures, API authentication errors, and more.

---

## Safe Code Execution Sandbox

The platform includes a multi-language code execution sandbox with security controls:

| Feature | Implementation |
|---------|---------------|
| **Languages** | Python, Bash, R, JavaScript |
| **Static Analysis** | AST parsing + import validation |
| **Path Whitelist** | Restricted file system access |
| **Command Whitelist** | Approved shell commands only |
| **Session Isolation** | Per-session virtual environments |
| **Dependency Management** | Dual venv/conda layer with auto-install |

---

## Benchmark & Evaluation

The platform includes an automated benchmark framework:

- **Datasets**: LAB-Bench (SuppQA, LitQA2, SeqQA, DbQA, ProtocolQA), TRQA (200 core questions)
- **Checkpoint Resume**: Save and restore evaluation progress
- **LLM-as-Judge**: Multi-dimensional automated scoring
- **Concurrent Execution**: Parallel evaluation for faster iteration

---

## Skills Overview

| Category | Count | Highlights |
|---|---|---|
| General & Core | 10 | Browser, search, document tools, developer workflows |
| Medical & Clinical | 119 | Clinical reports, CDS, oncology, imaging, healthcare AI |
| Scientific Databases | 43 | Genomics, protein, drug databases, knowledge retrieval |
| Bioinformatics | 239 | Variant analysis, sequencing QC, DE, pathways, single-cell |
| Omics & Computational Biology | 59 | Single-cell/spatial, proteomics, cheminformatics, protein design |
| ClawBio Pipelines | 21 | Orchestration for scRNA, GWAS, ancestry, structural workflows |
| BioOS Extended Suite | 285 | Oncology, immunology, clinical AI, infrastructure |
| Data Science & Tools | 93 | Statistics, visualization, automation, simulation |
| **Total** | **869** | |

---

## Installation

### Requirements

- Python 3.10+
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/1695963547/OpenClaw-Medical-Skills.git
cd OpenClaw-Medical-Skills

# Install dependencies
pip install -r requirements.txt

# Or use conda
conda env create -f environment.yml
conda activate openclaw-medical
```

### Configure LLM

Create a `llm_local_config.py` file (or copy from `llm_config.example.py`):

```python
LLM_CONFIG = {
    "model": "deepseek-v4-flash",        # Model name
    "base_url": "https://api.deepseek.com",  # API endpoint
    "api_key": "your-api-key-here",       # API key
    "skill_top_k": 8,                    # Number of skills to retrieve
    "max_iterations": 20,                 # Max agent iterations
}
```

**Supported models**: DeepSeek V3/V4, GLM-4.6, and any OpenAI-compatible API.

### Initialize Vector Index

```bash
# Build ChromaDB index from skill_registry.json (first run only)
python -c "from src.skill_retriever import SkillRetriever; SkillRetriever('skill_registry.json')"
```

### Run

```bash
python main.py
```

```
Medical Skills Agent 就绪。输入 exit 退出。

> What drugs interact with warfarin?
👤 User: What drugs interact with warfarin?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 执行计划
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ⬜ Step 1: Retrieve drug interaction skills
  ⬜ Step 2: Query drug interaction database
  ⬜ Step 3: Analyze and summarize results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅ [1/3] done: Retrieve drug interaction skills
  ✅ [2/3] done: Query drug interaction database
  ✅ [3/3] done: Analyze and summarize results

🤖 Agent: Warfarin has major interactions with...
```

---

## Project Structure

```
OpenClaw-Medical-Skills/
├── main.py                    # Entry point: CLI loop, logging, session management
├── skill_registry.json        # 869 skill metadata index
├── src/
│   ├── agent.py               # LangGraph ReAct agent (5-node state graph)
│   ├── skill_retriever.py     # ChromaDB semantic retrieval
│   ├── skill_context.py       # Skill context assembly
│   ├── tools.py               # 4 tool functions (retrieve/read/execute/status)
│   ├── code_executor.py       # Multi-language sandbox executor
│   ├── conda_manager.py       # Conda environment management
│   ├── llm_factory.py         # LLM instance factory
│   ├── skill_stats.py         # Usage statistics tracker
│   └── middleware/            # Error resilience framework
│       ├── error_memory.py        # Error memory + strategy escalator
│       ├── resilient_executor.py # Resilient execution wrapper
│       ├── schema_healer.py       # GraphQL schema self-healing
│       └── tool_param_adapter.py # Parameter normalization
├── skills/                    # 869 skill modules (SKILL.md files)
├── scripts/                   # Benchmark & utility scripts
│   ├── benchmark_test.py      # Concurrent benchmark runner
│   ├── build_registry.py      # Skill registry builder
│   ├── skill_audit.py         # Skill validation auditor
│   └── validate_skill.py      # Single skill validator
├── TestQuestion/              # Benchmark datasets (LAB-Bench, TRQA)
└── chroma_db/                 # Pre-built vector index
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Agent Framework** | LangGraph (ReAct state graph) |
| **Vector Database** | ChromaDB |
| **Embedding Model** | sentence-transformers |
| **LLM Interface** | LangChain + ChatOpenAI (OpenAI-compatible) |
| **Code Execution** | subprocess + AST analysis + venv/conda |
| **Observability** | LangSmith tracing + JSONL logging |
| **Language** | Python 3.10+ |

---

## Acknowledgments

This project builds upon the [OpenClaw Medical Skills](https://github.com/FreedomIntelligence/OpenClaw-Medical-Skills) skill collection by [FreedomIntelligence](https://github.com/FreedomIntelligence). The original skill library has been extended with a LangGraph-based agent framework, semantic retrieval system, error resilience middleware, and safe code execution sandbox.

Skills are aggregated from 12+ open-source repositories. Full credits available in the original repo.

## License

MIT License — see [LICENSE](LICENSE) for details.
