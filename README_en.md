# OpenClaw Medical Skills — Medical AI Agent Platform

<div align="center">

[![Skills Count](https://img.shields.io/badge/Skills-869-brightgreen?style=for-the-badge)](https://github.com/1695963547/OpenClaw-Medical-Skills/tree/main/skills)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-ReAct-orange?style=for-the-badge)](https://github.com/langchain-ai/langgraph)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Search-purple?style=for-the-badge)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/License-MIT-gray?style=for-the-badge)](LICENSE)

**An autonomous medical AI agent platform powered by LangGraph, integrating 869 biomedical skills with semantic retrieval, multi-layer error resilience, and safe code execution.**

*Clinical · Genomics · Drug Discovery · Bioinformatics · Medical Devices*

**English | [中文](README.md)**

</div>

---

## Background

General-purpose LLMs face two core challenges in medical settings:

> **Challenge 1: Difficulty accessing specialized domain knowledge**
> General LLMs lack connections to real medical databases (PubMed, ClinicalTrials.gov, ChEMBL, etc.) and can only answer based on static training knowledge — making them incapable of real-time clinical and research queries.

> **Challenge 2: Unstable reasoning chains**
> For complex multi-step biomedical tasks (RNA-seq workflows, variant annotation, protein structure prediction), general LLMs suffer from hallucinations, missed steps, and code execution failures, with no systematic self-healing mechanisms.

This project was built **from scratch** as an enterprise AGI strategic initiative to solve both challenges through a LangGraph orchestration framework combined with 869 specialized biomedical skills — delivering an autonomous AI agent that can "reason, execute, and understand medicine."

---

## Overview

OpenClaw Medical Skills transforms a general-purpose LLM into a specialized medical and scientific research assistant. It integrates **869 curated biomedical skills** — covering clinical workflows, genomics, drug discovery, and bioinformatics — into a unified LangGraph agent framework.

The agent autonomously retrieves relevant skills via semantic search, generates multi-step execution plans, executes code in a sandboxed environment, and self-heals from errors — all without human intervention.

### Capability Comparison

| Without This Platform | With This Platform |
|---|---|
| Generic AI responses about medicine | Real PubMed / ClinicalTrials.gov / FDA queries |
| No bioinformatics capability | RNA-seq, scRNA-seq, GWAS, variant calling pipelines |
| No drug intelligence | ChEMBL, DrugBank, DDI prediction, pharmacovigilance |
| No clinical documentation | SOAP notes, discharge summaries, prior auth decisions |
| No genomics support | VCF annotation, ACMG classification, PRS calculation |
| No regulatory guidance | FDA, CE mark, IEC 62304, ISO 14971 compliance |

---

## Core Features

| Feature | Description |
|---------|-------------|
| 🧠 **Autonomous Skill Retrieval** | Semantically retrieves the most relevant skills before every conversation — no manual skill selection needed |
| 📋 **Multi-Step Planning** | Automatically decomposes complex tasks into structured subtasks with real-time progress display |
| 🛡️ **5-Layer Error Resilience** | Covers 13 error types — agent automatically switches strategies under API failures |
| 🔒 **Safe Code Execution** | AST static analysis + path/command whitelists + per-session venv isolation |
| 💬 **Multi-Turn Memory** | MemorySaver persists conversation context across turns within the same session |
| 🔄 **Stuck Soft-Landing** | Stuck Detector triggers graceful degradation instead of infinite retry loops |
| 📊 **Automated Benchmarking** | Built-in LAB-Bench / TRQA evaluation with checkpoint resume and LLM-as-Judge scoring |
| 🔌 **Multi-Model Compatible** | Supports DeepSeek V3/V4, GLM-4.6, and any OpenAI-compatible API |

---

## Architecture

### System Architecture

The agent is built on a **LangGraph 5-node state graph** with conditional routing:

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph State Graph                     │
│                                                             │
│  ┌──────────────┐    ┌──────────┐    ┌──────────┐          │
│  │ auto_retrieve │───→│  agent   │───→│  tools   │          │
│  │  (semantic    │    │ (LLM     │    │ (code    │          │
│  │   pre-load)   │    │  decision)│   │  execution)│        │
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
| **tools** | Executes: `retrieve_skills`, `read_file`, `execute_code`, `update_task_status` |
| **post_tools** | Error classification (13 types), correction injection, stuck detection, subtask tracking |
| **planner** | Generates structured multi-step execution plans for complex tasks |

### Key Design Decisions

**1. Why LangGraph instead of simple chains?**

Simple chains are static — they can't handle the dynamic decision-making required for medical tasks ("should I query the database again?", "how to recover from a code error?"). LangGraph's stateful graph enables true autonomous loop reasoning with dynamic routing based on execution results.

**2. Why pre-retrieve skills before LLM decision (auto_retrieve)?**

Without context, LLMs often choose to answer directly rather than invoke tools, missing the opportunity to use specialized skills. The `auto_retrieve` node proactively injects the most relevant skills into the System Prompt at the start of every conversation, dramatically increasing the probability that the LLM selects the right tool — a proactive improvement over standard RAG.

**3. Why 5-layer error resilience instead of simple retries?**

Medical APIs (NCBI, ChEMBL, UniProt, etc.) fail in many ways: authentication errors, schema changes, missing dependencies — each requiring a different strategy. Simple retries only handle transient failures. Layered resilience uses error memory to avoid repeated mistakes and strategy escalation to progressively degrade, maximizing task completion rates.

**4. Why a dedicated planner node?**

For simple Q&A, the `agent` node can respond directly. But for complex tasks like "analyze this sequencing dataset" requiring 5-10 steps, the agent loses direction without an explicit plan. The structured subtask list generated by `planner` serves as an "execution contract" for the agent, with `post_tools` tracking real-time completion status.

### Additional Design Features

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
Layer 5: GraphQL Schema Self-Healing  → Auto-repair broken API schemas
```

Covers **13 error categories** including script errors, import failures, package installation failures, API authentication errors, and more.

---

## Safe Code Execution Sandbox

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

## Quick Start

### Requirements

- Python 3.10+
- Git

### Installation

```bash
git clone https://github.com/1695963547/OpenClaw-Medical-Skills.git
cd OpenClaw-Medical-Skills
pip install -r requirements.txt
```

### Configure LLM

Create `llm_local_config.py`:

```python
LLM_CONFIG = {
    "model": "deepseek-v4-flash",
    "base_url": "https://api.deepseek.com",
    "api_key": "your-api-key-here",
    "skill_top_k": 8,
    "max_iterations": 20,
}
```

### Run

```bash
python main.py
```

---

## Project Structure

```
OpenClaw-Medical-Skills/
├── main.py                    # Entry: CLI loop, logging, session management
├── skill_registry.json        # 869 skill metadata index
├── src/
│   ├── agent.py               # LangGraph ReAct agent (5-node state graph)
│   ├── skill_retriever.py     # ChromaDB semantic retrieval
│   ├── skill_context.py       # Skill context assembly
│   ├── tools.py               # 4 tool functions
│   ├── code_executor.py       # Multi-language sandbox executor
│   ├── llm_factory.py         # LLM instance factory
│   └── middleware/            # Error resilience framework
│       ├── error_memory.py
│       ├── resilient_executor.py
│       ├── schema_healer.py
│       └── tool_param_adapter.py
├── skills/                    # 869 skill modules
├── scripts/                   # Benchmark & utility scripts
├── TestQuestion/              # Benchmark datasets
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

Built upon the [OpenClaw Medical Skills](https://github.com/FreedomIntelligence/OpenClaw-Medical-Skills) skill collection by [FreedomIntelligence](https://github.com/FreedomIntelligence). Extended with a LangGraph agent framework, semantic retrieval system, error resilience middleware, and safe code execution sandbox.

## License

MIT License — see [LICENSE](LICENSE) for details.
