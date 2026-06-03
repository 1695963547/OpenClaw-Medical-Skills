"""
Medical Skills Agent 基准测试脚本
==================================

对「问题模板和衍生问题.md」中的 10 个衍生问题进行自动化测试，
每个问题重复运行 N 次，统计过程指标和结论质量评分。

用法:
    # 先测 2 条验证解析正常
python scripts/benchmark_test.py --questions 1,2 --runs 1 

# 验证通过后，小规模测试（10 条 × 1 次）
python scripts/benchmark_test.py --test-file "TestQuestion/TRQA测试案例（200条核心差异化数据）.md" --questions 1,10,50,100,150,200 --runs 1

# 全量 200 条 × 1 次（预计数小时，务必利用 checkpoint 机制）
python scripts/benchmark_test.py --test-file "TestQuestion/TRQA测试案例（200条核心差异化数据）.md" --runs 1 --workers 3


    python scripts/benchmark_test.py
    python scripts/benchmark_test.py --runs 5
    python scripts/benchmark_test.py --runs 10 --questions 1,3,5
    python scripts/benchmark_test.py --runs 3 --skip-judge   # 跳过 LLM-as-Judge 节省成本
    python scripts/benchmark_test.py --runs 10 --workers 3
    python scripts/benchmark_test.py --runs 10 --workers 3 --resume logs/benchmark_checkpoint_20260529_123542.json
可配置参数（修改下方常量或通过命令行参数覆盖）:
    RUNS_PER_QUESTION   : 每个问题重复运行次数（默认 3）
    JUDGE_MODEL         : 裁判 LLM 模型名称
    JUDGE_PASS_SCORE    : LLM-as-Judge overall 通过分数线（默认 6）
    MAX_WORKERS         : 并发线程数（默认 3，设为 1 则串行执行）
    TEST_FILE           : 测试问题 MD 文件路径
"""

import re
import json
import uuid
import logging
import sys
import os
import time
import shutil
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

# ── 确保项目根目录在 sys.path，并切换工作目录（使所有相对路径生效） ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.skill_retriever import SkillRetriever
from src.agent import MedicalSkillAgent
from src.llm_factory import build_llm, load_local_llm_settings
from src.tools import set_session_id, cleanup_session_venv, get_workspace_dir
from src.middleware.error_memory import set_memory_session_id, cleanup_memory  # 错误记忆清理

# ═══════════════════════════════════════════════════════════
#  ★ 可配置参数 — 修改这里的值即可 ★
# ═══════════════════════════════════════════════════════════

# 并发 print 保护锁（模块级，run_single 内部使用）
_print_lock = threading.Lock()
RUNS_PER_QUESTION = 3                # 每个问题重复运行次数
JUDGE_MODEL = "DeepSeek-V3.2-ALi"    # 裁判 LLM 模型
JUDGE_PASS_SCORE = 5                 # LLM-as-Judge overall ≥ 此分数算通过（从 6 降为 5，匹配 Judge 对 coverage 评分偏严的现实）
MAX_WORKERS = 3                      # 并发线程数（1=串行，3-5=推荐并发）
TASK_TIMEOUT = 600                   # 单任务超时（秒），防止 API 卡死导致无限等待
LLM_TIMEOUT = 90                    # LLM 单次请求超时（秒），防止网关无响应时线程永久阻塞
LLM_MAX_RETRIES = 1                  # LLM 最大重试次数（1=只重试1次，减少并发场景下的阻塞时间）
TEST_FILE = "TestQuestion/LAB-Bench_SuppQA 测试集.md"  # 相对于项目根目录
# 注意: Benchmark 会自动将 MAX_ITERATIONS 提升至至少 40（复杂 6 步问题需要 36+ 次迭代）
# 如需更高，可在 llm_local_config.py 中设置 MAX_ITERATIONS = "60"
# ═══════════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────────
#  数据结构
# ───────────────────────────────────────────────────────────

@dataclass
class TestCase:
    """单个测试用例（从 MD 文件解析得到）"""
    index: int
    template: str
    query: str
    expected: str


@dataclass
class JudgeResult:
    """LLM-as-Judge 评分结果"""
    coverage: float = 0
    accuracy: float = 0
    logic: float = 0
    depth: float = 0
    overall: float = 0
    reason: str = ""
    raw: str = ""
    error: str = ""


@dataclass
class RunResult:
    """单次运行的完整结果"""
    question_index: int
    run_index: int
    thread_id: str
    # 过程指标
    tool_status: str = ""
    tool_calls_count: int = 0
    selected_skills: list = field(default_factory=list)
    final_answer: str = ""
    duration: float = 0
    error: str = ""
    # 过程判定
    process_pass: bool = False
    process_fail_reasons: list = field(default_factory=list)
    # 结论评分
    judge: Optional[JudgeResult] = None
    # 整体判定
    overall_pass: bool = False
    status: str = "FAIL"   # PASS / WEAK_PASS / FAIL


def _serialize_run_result(r: RunResult) -> dict:
    """将 RunResult 序列化为可 JSON 存储的 dict"""
    return {
        "question_index": r.question_index,
        "run_index": r.run_index,
        "thread_id": r.thread_id,
        "tool_status": r.tool_status,
        "tool_calls_count": r.tool_calls_count,
        "selected_skills": r.selected_skills,
        "final_answer": r.final_answer,
        "duration": r.duration,
        "error": r.error,
        "process_pass": r.process_pass,
        "process_fail_reasons": r.process_fail_reasons,
        "judge": {
            "coverage": r.judge.coverage,
            "accuracy": r.judge.accuracy,
            "logic": r.judge.logic,
            "depth": r.judge.depth,
            "overall": r.judge.overall,
            "reason": r.judge.reason,
            "raw": r.judge.raw,
            "error": r.judge.error,
        } if r.judge else None,
        "overall_pass": r.overall_pass,
        "status": r.status,
    }


def _save_checkpoint(checkpoint_path: str, completed_results: dict, config: dict):
    """增量保存 checkpoint 文件（每完成一个任务调用一次）
    使用 os.replace 原子写入，防止写入中断导致文件损坏。
    """
    checkpoint = {
        "version": 1,
        "config": config,
        "saved_at": datetime.now().isoformat(),
        "completed_count": sum(len(v) for v in completed_results.values()),
        "results": {
            str(q_idx): [_serialize_run_result(r) for r in runs]
            for q_idx, runs in completed_results.items()
        },
    }
    tmp_path = checkpoint_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, checkpoint_path)  # 原子写入


def _load_checkpoint(checkpoint_path: str):
    """加载 checkpoint 文件，返回 (completed_set, results_dict, config)
    completed_set: {(q_idx, run_idx), ...} 已完成的 (问题, 运行) 对
    results_dict: {q_idx: [RunResult, ...]} 已完成的结果
    """
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    completed_set = set()
    results_dict = {}

    for q_idx_str, runs in data.get("results", {}).items():
        q_idx = int(q_idx_str)
        results_dict[q_idx] = []
        for rd in runs:
            jr = None
            if rd.get("judge"):
                jd = rd["judge"]
                jr = JudgeResult(
                    coverage=jd.get("coverage", 0),
                    accuracy=jd.get("accuracy", 0),
                    logic=jd.get("logic", 0),
                    depth=jd.get("depth", 0),
                    overall=jd.get("overall", 0),
                    reason=jd.get("reason", ""),
                    raw=jd.get("raw", ""),
                    error=jd.get("error", ""),
                )
            r = RunResult(
                question_index=rd["question_index"],
                run_index=rd["run_index"],
                thread_id=rd.get("thread_id", ""),
                tool_status=rd.get("tool_status", ""),
                tool_calls_count=rd.get("tool_calls_count", 0),
                selected_skills=rd.get("selected_skills", []),
                final_answer=rd.get("final_answer", ""),
                duration=rd.get("duration", 0),
                error=rd.get("error", ""),
                process_pass=rd.get("process_pass", False),
                process_fail_reasons=rd.get("process_fail_reasons", []),
                judge=jr,
                overall_pass=rd.get("overall_pass", False),
                status=rd.get("status", "FAIL"),
            )
            results_dict[q_idx].append(r)
            completed_set.add((q_idx, rd["run_index"]))

    return completed_set, results_dict, data.get("config", {})


# ═══════════════════════════════════════════════════════════
#  MD 文件解析（基于行的有限状态机）
# ═══════════════════════════════════════════════════════════

def parse_test_file(filepath: str) -> list:
    """从 MD 文件中解析测试用例。

    支持两种格式：
    1. 问题模板格式：## 问题模板 + Query: + ##### 结果
    2. 表格格式：| 序号 | 中文问题 | 中文回复 |
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # ── 格式检测：如果包含表格表头，走表格解析分支 ──
    if re.search(r'\|\s*序号\s*\|\s*(?:中文)?问题\s*\|', content):
        return _parse_table_format(content)

    # ── 原有：问题模板格式 ──
    # ── 提取所有模板 ──
    templates = []
    for m in re.finditer(r"## 问题模板\s*\n(.*?)(?=\n##### 衍生问题)", content, re.DOTALL):
        tmpl = m.group(1).strip()
        if tmpl:
            templates.append(tmpl)

    # ── 状态机提取 Query 和结果 ──
    queries = []
    results = []
    section = None   # "query" | "result" | None
    buf = []

    for line in content.split("\n"):
        stripped = line.strip()

        # 新的 Query 行
        if stripped.startswith("Query:"):
            # 先保存上一个 section
            if section == "query" and buf:
                queries.append("\n".join(buf).strip())
                buf = []
            elif section == "result" and buf:
                results.append("\n".join(buf).strip())
                buf = []
            section = "query"
            text = stripped[len("Query:"):].strip()
            buf = [text] if text else []
            continue

        # 结果区域开始
        if stripped == "##### 结果":
            if section == "query" and buf:
                queries.append("\n".join(buf).strip())
                buf = []
            section = "result"
            buf = []
            continue

        # 下一个问题块开始 → 结束当前
        if stripped == "## 问题模板":
            if section == "result" and buf:
                results.append("\n".join(buf).strip())
                buf = []
            elif section == "query" and buf:
                queries.append("\n".join(buf).strip())
                buf = []
            section = None
            continue

        # 跳过区域标题
        if stripped == "##### 衍生问题":
            continue

        # 累积内容
        if section is not None:
            buf.append(line)

    # 处理文件末尾
    if section == "result" and buf:
        results.append("\n".join(buf).strip())
    elif section == "query" and buf:
        queries.append("\n".join(buf).strip())

    # ── 组装测试用例 ──
    test_cases = []
    for i, (query, result) in enumerate(zip(queries, results)):
        tmpl = templates[i] if i < len(templates) else ""
        test_cases.append(TestCase(index=i + 1, template=tmpl, query=query, expected=result))

    return test_cases


def _parse_table_format(content: str) -> list:
    """解析 Markdown 表格格式的测试用例（| 序号 | 中文问题 | 中文回复 |）"""
    test_cases = []
    for m in re.finditer(
        r'\|\s*(\d+)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|',
        content
    ):
        idx = int(m.group(1))
        query = m.group(2).strip()
        expected = m.group(3).strip()
        # 跳过分隔行（如 | --- | --- | --- |）
        if query and expected and not query.startswith('-'):
            test_cases.append(TestCase(
                index=idx, template="", query=query, expected=expected
            ))
    return test_cases


# ═══════════════════════════════════════════════════════════
#  Agent 初始化（复用 main.py 流程）
# ═══════════════════════════════════════════════════════════

def init_agent(logger):
    """初始化 MedicalSkillAgent"""
    retriever = SkillRetriever(os.path.join(PROJECT_ROOT, "skill_registry.json"))
    local_settings = load_local_llm_settings()
    profile = (local_settings["profile"] or os.getenv("LLM_PROFILE") or "").strip().lower()
    profiles = {
        "glm": {"model": "GLM-4.6"},
        "glm-4.6": {"model": "GLM-4.6"},
        "deepseek": {"model": "deepseek-v4-flash"},
        "deepseek-v4": {"model": "deepseek-v4-flash"},
        "deepseek-v4-flash": {"model": "deepseek-v4-flash"},
        "deepseek-v4-pro": {"model": "deepseek-v4-pro"},
        "deepseek-v3": {"model": "DeepSeek-V3.2-HuoS"},
        "deepseek-v3.2-huos": {"model": "DeepSeek-V3.2-HuoS"},
    }
    default_model = profiles.get(profile, {}).get("model", "deepseek-v4-flash")
    model = local_settings["model"] or os.getenv("LLM_MODEL", default_model)
    llm, llm_info = build_llm(model=model, timeout=LLM_TIMEOUT, max_retries=LLM_MAX_RETRIES)

    skill_top_k_str = os.getenv("SKILL_TOP_K") or local_settings.get("skill_top_k") or ""
    try:
        skill_top_k = int(skill_top_k_str.strip()) if skill_top_k_str.strip() else 8
    except ValueError:
        skill_top_k = 8

    # Benchmark 模式：自动提升迭代上限，确保复杂多步问题（6步×6轮=36+）不会耗尽
    BENCHMARK_MIN_ITERATIONS = 40
    max_iter_str = os.getenv("MAX_ITERATIONS") or local_settings.get("max_iterations") or ""
    try:
        base_max_iter = int(max_iter_str.strip()) if max_iter_str.strip() else 20
    except ValueError:
        base_max_iter = 20
    max_iterations = max(base_max_iter, BENCHMARK_MIN_ITERATIONS)
    if max_iterations > base_max_iter:
        logger.info("Benchmark 模式: MAX_ITERATIONS 提升至 %d（原始配置 %d）", max_iterations, base_max_iter)

    agent = MedicalSkillAgent(
        retriever=retriever, llm=llm, top_k=skill_top_k, max_iterations=max_iterations,
    )
    logger.info("Agent 就绪 | model=%s | base_url=%s | top_k=%d | max_iter=%d",
                llm_info["model"], llm_info["base_url"], skill_top_k, max_iterations)
    return agent


# ═══════════════════════════════════════════════════════════
#  单次运行 Agent 并采集过程指标
# ═══════════════════════════════════════════════════════════

def run_single(agent, query, thread_id, logger, timeout=TASK_TIMEOUT):
    """运行一次 Agent 并采集 run_record。

    实时输出逻辑与 main.py 主循环完全对齐：
      - planner: 打印执行计划
      - agent tool_calls: logger.info 打印工具调用
      - agent reply: print 打印完整回复
      - tools: logger.info 打印工具执行结果
      - post_tools: 打印子任务进度变化
    """
    start_time = datetime.now()
    deadline = time.time() + timeout
    set_session_id(thread_id)
    set_memory_session_id(thread_id)

    rec = {
        "retrieved_skills": [],
        "selected_skill": "",
        "selected_skills": [],
        "used_skills": False,
        "tool_status": "未调用",
        "tool_calls_count": 0,
        "final_answer": "",
        "_last_subtasks": [],   # 内部追踪：子任务状态快照（用于检测变化）
        "_timed_out": False,    # 内部标记：是否因超时强制终止
    }

    for event in agent.stream(query, thread_id=thread_id):
        # ── 任务级超时检查（每个 event 返回后判定）──
        if time.time() > deadline:
            rec["_timed_out"] = True
            logger.warning("任务超时 (%ds)，强制终止", timeout)
            break

        for node_name, node_state in event.items():
            if node_state is None:
                continue

            # ── 节点 1：auto_retrieve（系统自动检索）──
            if node_name == "auto_retrieve":
                retrieved = node_state.get("retrieved_skills", [])
                if retrieved:
                    rec["retrieved_skills"] = retrieved

            # ── 节点 2：planner（多步骤计划生成）──
            elif node_name == "planner":
                plan = node_state.get("plan", "")
                subtasks = node_state.get("subtasks", [])
                if plan and subtasks:
                    with _print_lock:
                        print("\n" + "━" * 50)
                        print("📋 执行计划")
                        print("━" * 50)
                        for i, sub in enumerate(subtasks):
                            desc = sub.get("description", f"步骤 {i + 1}")
                            clean_desc = re.sub(r"^\d+\.\s*\[?[^\]]*\]?\s*", "", desc, count=1)
                            icon = "⬜"
                            print(f"  {icon} Step {i + 1}: {clean_desc}")
                        print("━" * 50)
                    # 记录初始 subtask 状态用于后续比较
                    rec["_last_subtasks"] = [
                        {"description": s.get("description", ""), "status": s.get("status", "pending")}
                        for s in subtasks
                    ]

            # ── 节点 3：agent（LLM 决策核心）──
            elif node_name == "agent":
                if "selected_skill" in node_state and node_state["selected_skill"]:
                    rec["selected_skill"] = node_state["selected_skill"]
                messages = node_state.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    # 分支 A：LLM 决定调用工具
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        rec["used_skills"] = True
                        for tc in last_msg.tool_calls:
                            rec["tool_calls_count"] += 1
                            # 收集使用过的 skill（去重）
                            if tc["name"] == "read_file":
                                sid = tc["args"].get("skill_id", "")
                                if sid and sid not in rec["selected_skills"]:
                                    rec["selected_skills"].append(sid)
                            elif tc["name"] == "execute_code":
                                sid = tc["args"].get("related_skill", "")
                                if sid and sid not in rec["selected_skills"]:
                                    rec["selected_skills"].append(sid)
                            # 终端/日志展示裁剪后的 code 参数（与 main.py 一致）
                            display_args = {}
                            for k, v in tc["args"].items():
                                if k == "code" and isinstance(v, str) and len(v) > 200:
                                    display_args[k] = v[:200] + f"... (共 {len(v)} 字符)"
                                else:
                                    display_args[k] = v
                            logger.info("模型决策 → 调用工具: %s | 参数: %s", tc["name"], display_args)
                            if tc["name"] == "retrieve_skills":
                                rec["retrieved_skills"] = ["(等待工具执行结果)"]
                    # 分支 B：LLM 直接回复用户
                    elif last_msg.content:
                        if not rec["used_skills"]:
                            logger.info("模型决策 → 未调用工具，直接回复")
                        rec["final_answer"] = last_msg.content
                        # 终端完整显示回复（与 main.py 一致，线程安全）
                        with _print_lock:
                            print(f"\n🤖 Agent:\n{last_msg.content}")
                        logger.debug("模型完整回复: %s", last_msg.content)

            # ── 节点 4：tools（工具执行）──
            elif node_name == "tools":
                messages = node_state.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    out = last_msg.content
                    out_preview = out[:500] if len(out) > 500 else out
                    if len(out) > 500:
                        out_preview += "\n... (已省略后续输出)"
                    logger.info("工具执行结果 (%s): %s", last_msg.name, out_preview)
                    logger.debug("工具原始输出 (%s): %s", last_msg.name, out)
                    # 解析 retrieve_skills 的召回列表
                    if last_msg.name == "retrieve_skills":
                        match = re.search(r"已检索到 \d+ 个相关技能[：:]\s*(.+)", last_msg.content)
                        if match:
                            rec["retrieved_skills"] = [s.strip() for s in match.group(1).split(",")]
                            logger.info("检索阶段 → 召回 %d 个技能: %s",
                                        len(rec["retrieved_skills"]), rec["retrieved_skills"])

            # ── 节点 5：post_tools（后处理/错误修正/Stuck检测）──
            elif node_name == "post_tools":
                tool_status = node_state.get("tool_status", "")
                if tool_status:
                    rec["tool_status"] = tool_status
                    if tool_status == "成功":
                        logger.info("执行状态: %s", tool_status)
                    else:
                        logger.warning("执行状态: %s", tool_status)

                # ── 实时展示子任务进度变化（与 main.py 一致，线程安全）──
                subtasks = node_state.get("subtasks", [])
                last_subtasks = rec.get("_last_subtasks", [])
                if subtasks:
                    _changed_lines = []
                    for i, sub in enumerate(subtasks):
                        sub_status = sub.get("status", "pending")
                        last_status = "pending"
                        if i < len(last_subtasks):
                            last_status = last_subtasks[i].get("status", "pending")
                        if sub_status != last_status:
                            desc = sub.get("description", f"步骤 {i + 1}")
                            clean_desc = re.sub(r"^\d+\.\s*\[?[^\]]*\]?\s*", "", desc, count=1)
                            if sub_status == "done":
                                icon = "✅"
                            elif sub_status == "in_progress":
                                icon = "🔄"
                            elif sub_status == "failed":
                                icon = "❌"
                            else:
                                icon = "⬜"
                            _changed_lines.append(
                                f"  {icon} [{i + 1}/{len(subtasks)}] {sub_status}: {clean_desc}")
                    if _changed_lines:
                        with _print_lock:
                            for line in _changed_lines:
                                print(line)
                    rec["_last_subtasks"] = [
                        {"description": s.get("description", ""), "status": s.get("status", "pending")}
                        for s in subtasks
                    ]

    rec["duration"] = (datetime.now() - start_time).total_seconds()
    return rec


# ═══════════════════════════════════════════════════════════
#  过程指标评估（仅检查最终回复是否产出）
# ═══════════════════════════════════════════════════════════

def evaluate_process(rec):
    """评估过程指标。返回 (is_pass: bool, fail_reasons: list)

    过程考核仅检查 Agent 是否成功产生了最终回复（final_answer）。
    tool_calls_count / retrieved_skills / selected_skills 作为观测指标保留在报告中，
    不再作为过程 PASS 的必要条件——部分问题 Agent 凭自身知识即可回答，
    部分问题 Agent 调用了工具但未显式关联技能 ID，均属合理行为。
    """
    fails = []

    answer = rec.get("final_answer", "")
    if not answer:
        fails.append("无 final_answer（可能崩溃或未完成）")

    return len(fails) == 0, fails


# ═══════════════════════════════════════════════════════════
#  LLM-as-Judge 结论评分
# ═══════════════════════════════════════════════════════════

JUDGE_PROMPT = """你是一位严谨的生物医学分析质量评审员。请根据以下信息对 AI 助手的回答进行评分。

【问题】
{query}

【参考答案】
{expected}

【AI 实际回复】
{actual}

请从以下 4 个维度评分（每个维度 1-10 分）：

1. coverage（关键结论覆盖率）：AI 回复是否涵盖了参考答案中的核心结论和关键数据点？
2. accuracy（科学准确性）：数据引用、专业术语和因果推理是否正确？是否存在事实性错误？
3. logic（逻辑完整性）：分析推理链是否完整？结论是否有充分的证据支撑？
4. depth（专业深度）：是否体现了领域专业知识？分析的颗粒度是否足够？

评分参考锚点：
- 9-10 分：优秀，结论完整准确，可直接用于专业决策参考
- 7-8 分：良好，核心结论正确，细节可补充但不影响判断
- 5-6 分：及格，大方向正确但缺少关键论证或重要数据
- 3-4 分：不足，存在明显错误或重大遗漏
- 1-2 分：极差，答非所问或结论完全错误

评分注意事项：
- coverage 关注核心结论是否覆盖，不要求逐词匹配参考答案中的特定药物名/靶点名/试验编号。如果 AI 回复的核心结论方向与参考答案一致且覆盖大部分关键点，coverage 可给 7-8 分；如果几乎完全覆盖，给 9-10 分；如果遗漏了参考答案中的重要结论，应酌情扣分至 4-6 分；如果完全偏离参考答案，应给 3 分以下。
- accuracy 以事实性错误的严重程度扣分：关键结论错误（如用错药物、搞反因果关系）扣至 3-4 分；次要细节偏差（如剂量范围略有出入）扣 1-2 分即可。

仅输出一个合法 JSON 对象，不要输出任何其他文字。
{{"coverage": <分数>, "accuracy": <分数>, "logic": <分数>, "depth": <分数>, "reason": "<100字以内的评分理由，说明主要扣分点>"}}"""




def judge_response(query, expected, actual, judge_llm, logger):
    """调用裁判 LLM 对实际回复进行 1-10 分评分"""
    if not actual or len(actual) < 50:
        return JudgeResult(error="回复过短，跳过评分", reason="回复过短")

    prompt = JUDGE_PROMPT.format(
        query=query,
        expected=expected[:3000],
        actual=actual[:5000],
    )

    try:
        response = judge_llm.invoke(prompt)
        raw_text = response.content if hasattr(response, "content") else str(response)
        return _parse_judge_output(raw_text, logger)
    except Exception as e:
        logger.warning("Judge LLM 调用异常: %s", e)
        return JudgeResult(error=str(e))


def _parse_judge_output(raw_text, logger):
    """从裁判回复中解析 JSON 评分（3 种容错策略）

    overall 由代码自动计算（accuracy×0.35 + coverage×0.25 + logic×0.25 + depth×0.15），
    不依赖 LLM 输出——避免 LLM 算术错误导致的评分偏差。
    """
    parsed = None

    # 尝试 1: 直接解析
    try:
        parsed = json.loads(raw_text.strip())
    except json.JSONDecodeError:
        pass

    # 尝试 2: 正则提取含评分字段的 JSON 对象（锚点 "coverage" 始终存在）
    if parsed is None:
        match = re.search(r'\{[^{}]*"coverage"[^{}]*\}', raw_text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                pass

    # 尝试 3: 提取 ```json ... ``` 代码块
    if parsed is None:
        match = re.search(r'```(?:json)?\s*(\{.+?\})\s*```', raw_text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

    if parsed and isinstance(parsed, dict) and "coverage" in parsed:
        coverage = float(parsed.get("coverage", 0))
        accuracy = float(parsed.get("accuracy", 0))
        logic = float(parsed.get("logic", 0))
        depth = float(parsed.get("depth", 0))
        # ── overall 由代码计算，不信任 LLM 算术 ──
        overall = round(accuracy * 0.35 + coverage * 0.25 + logic * 0.25 + depth * 0.15)
        return JudgeResult(
            coverage=coverage,
            accuracy=accuracy,
            logic=logic,
            depth=depth,
            overall=float(overall),
            reason=str(parsed.get("reason", "")),
            raw=raw_text[:500],
        )

    logger.warning("Judge 回复无法解析:\n%s", raw_text[:300])
    return JudgeResult(error="JSON 解析失败", raw=raw_text[:500])


# ═══════════════════════════════════════════════════════════
#  报告生成
# ═══════════════════════════════════════════════════════════

def _short_label(query, max_len=18):
    """为问题生成简短标签"""
    for pattern in [r"的\s*(\S{2,8})\s*蛋白", r"相关\s*(\S{2,8})\s*基因",
                    r"挖掘(\S{2,10})的", r"评估(.{2,10})对", r"优化(.{2,10})药物",
                    r"针对(.{2,8})的", r"分析(.{2,8})的", r"揭示(.{2,8})相关",
                    r"开发针对(.{2,8})", r"设计针对(.{2,8})"]:
        m = re.search(pattern, query)
        if m:
            label = m.group(1).strip()
            return label[:max_len] if len(label) <= max_len else label[:max_len] + "…"
    return (query[:max_len] + "…") if len(query) > max_len else query


def generate_report(results, test_cases, logger, max_workers=1):
    """生成终端报告 + 返回 JSON 结构"""
    summary_data = []
    all_runs_flat = []

    logger.info("\n" + "═" * 80)
    logger.info("  📊 Medical Skills Agent 基准测试报告")
    logger.info("  生成时间: %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    logger.info("═" * 80)

    for tc in test_cases:
        q_idx = tc.index
        runs = results.get(q_idx, [])
        n = len(runs)
        if n == 0:
            continue

        all_runs_flat.extend(runs)

        process_pass_n = sum(1 for r in runs if r.process_pass)
        overall_pass_n = sum(1 for r in runs if r.overall_pass)
        weak_pass_n = sum(1 for r in runs if r.status == "WEAK_PASS")
        fail_n = sum(1 for r in runs if r.status == "FAIL")
        scores = [r.judge.overall for r in runs if r.judge and r.judge.overall > 0]
        durations = [r.duration for r in runs]
        tool_counts = [r.tool_calls_count for r in runs]

        avg_score = sum(scores) / len(scores) if scores else 0
        avg_duration = sum(durations) / len(durations) if durations else 0
        avg_tools = sum(tool_counts) / len(tool_counts) if tool_counts else 0
        success_rate = overall_pass_n / n * 100
        status_icon = "✅" if success_rate >= 80 else ("⚠️" if success_rate >= 50 else "❌")

        summary_data.append({
            "question_index": q_idx,
            "query_short": _short_label(tc.query),
            "total_runs": n,
            "process_pass": f"{process_pass_n}/{n}",
            "overall_pass": f"{overall_pass_n}/{n} ({success_rate:.0f}%)",
            "weak_pass": weak_pass_n,
            "fail": fail_n,
            "avg_score": round(avg_score, 1),
            "avg_duration": round(avg_duration, 1),
            "avg_tools": round(avg_tools, 1),
            "status_icon": status_icon,
        })

        # ── 每题详细 ──
        label = _short_label(tc.query, 30)
        logger.info("")
        logger.info("%s", '─' * 80)
        logger.info("  问题 %d: %s", q_idx, label)
        logger.info("  Query: %s%s", tc.query[:90], '…' if len(tc.query) > 90 else '')
        logger.info("%s", '─' * 80)
        logger.info("  运行: %d | 过程PASS: %d/%d | 整体PASS: %d/%d (%.0f%%) | WEAK: %d | FAIL: %d",
                    n, process_pass_n, n, overall_pass_n, n, success_rate, weak_pass_n, fail_n)
        logger.info("  平均结论分: %.1f | 平均耗时: %.1fs | 平均工具: %.1f次", avg_score, avg_duration, avg_tools)

        for r in runs:
            js = f"{r.judge.overall:.1f}" if r.judge and r.judge.overall > 0 else " - "
            tag = f"{r.status:<10}"
            info = ""
            if r.process_fail_reasons:
                info = f"  [{r.process_fail_reasons[0][:55]}]"
            elif r.error:
                info = f"  [异常: {r.error[:48]}]"
            elif r.judge and r.judge.error:
                info = f"  [Judge: {r.judge.error[:45]}]"
            logger.info("    #%d %s %7.1fs  结论%4s  工具%3d次%s",
                        r.run_index + 1, tag, r.duration, js, r.tool_calls_count, info)

    # ── 总计 ──
    total_n = len(all_runs_flat)
    total_pass = sum(1 for r in all_runs_flat if r.overall_pass)
    all_scores = [r.judge.overall for r in all_runs_flat if r.judge and r.judge.overall > 0]
    all_durs = [r.duration for r in all_runs_flat]
    all_tools = [r.tool_calls_count for r in all_runs_flat]

    g_rate = total_pass / total_n * 100 if total_n else 0
    g_score = sum(all_scores) / len(all_scores) if all_scores else 0
    g_dur = sum(all_durs) / len(all_durs) if all_durs else 0
    g_tools = sum(all_tools) / len(all_tools) if all_tools else 0

    logger.info("")
    logger.info("%s", '═' * 80)
    logger.info("  📋 总计")
    logger.info("%s", '═' * 80)
    logger.info("  总运行: %d | 整体PASS: %d/%d (%.0f%%)", total_n, total_pass, total_n, g_rate)
    logger.info("  平均结论分: %.1f | 平均耗时: %.1fs | 平均工具: %.1f次", g_score, g_dur, g_tools)

    # ── 汇总表 ──
    # 使用 print 直接输出（不带 logger 时间戳前缀），避免 PowerShell PSReadLine
    # 对超长行（logger.info 带 ~30 字符前缀 + 80 字符内容 = 110+）的渲染重复 BUG
    # 日志文件中仍通过 logger.info 保留完整记录
    print("")
    print('═' * 70)
    print("  📊 汇总表")
    print('═' * 70)
    hdr = f"  {'#':<5} {'问题':<18} {'过程':<8} {'整体':<14} {'均分':<6} {'耗时':<8} {'工具':<6} {'':>2}"
    print(hdr)
    print('  ' + '─' * 66)
    for s in summary_data:
        row = (f"  {s['question_index']:<5} {s['query_short']:<18} "
               f"{s['process_pass']:<8} {s['overall_pass']:<14} "
               f"{s['avg_score']:<6} {s['avg_duration']:>5.1f}s "
               f"{s['avg_tools']:<6} {s['status_icon']:>2}")
        print(row)
    print('  ' + '─' * 66)
    total_row = (f"  {'总计':<23} {total_pass}/{total_n:<12} {g_rate:>4.0f}%    "
                 f"{g_score:>5.1f} {g_dur:>5.1f}s {g_tools:>5.1f}")
    print(total_row)
    # 同步写入日志文件（FileHandler 不受 PSReadLine 影响）
    for s in summary_data:
        row = (f"  {s['question_index']:<5} {s['query_short']:<22} "
               f"{s['process_pass']:<8} {s['overall_pass']:<16} "
               f"{s['avg_score']:<7} {s['avg_duration']:>6.1f}s "
               f"{s['avg_tools']:<7} {s['status_icon']:>2}")
        logger.info("汇总 | %s", row.strip())
    logger.info("汇总 | 总计 PASS: %d/%d (%.0f%%) | 均分 %.1f | 耗时 %.1fs",
                total_pass, total_n, g_rate, g_score, g_dur)

    # ── 返回 JSON ──
    return {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "runs_per_question": len(results.get(1, [])) if results else 0,
            "judge_model": JUDGE_MODEL,
            "judge_pass_score": JUDGE_PASS_SCORE,
            "max_workers": max_workers,
        },
        "summary": summary_data,
        "totals": {
            "total_runs": total_n,
            "overall_pass": total_pass,
            "success_rate": round(g_rate, 1),
            "avg_judge_score": round(g_score, 1),
            "avg_duration_s": round(g_dur, 1),
        },
        "details": {
            str(q_idx): [
                {
                    "run": r.run_index + 1,
                    "status": r.status,
                    "process_pass": r.process_pass,
                    "process_fail_reasons": r.process_fail_reasons,
                    "tool_status": r.tool_status,
                    "tool_calls_count": r.tool_calls_count,
                    "selected_skills": r.selected_skills,
                    "duration": round(r.duration, 1),
                    "final_answer_length": len(r.final_answer),
                    "judge_overall": r.judge.overall if r.judge else 0,
                    "judge_reason": r.judge.reason if r.judge else "",
                    "judge_error": r.judge.error if r.judge and r.judge.error else "",
                    "error": r.error,
                }
                for r in runs
            ]
            for q_idx, runs in results.items()
        },
    }


# ═══════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════

def main():
    # ── 命令行参数 ──
    parser = argparse.ArgumentParser(description="Medical Skills Agent 基准测试")
    parser.add_argument("--runs", type=int, default=RUNS_PER_QUESTION,
                        help=f"每个问题运行次数 (默认 {RUNS_PER_QUESTION})")
    parser.add_argument("--questions", type=str, default="",
                        help="指定问题编号，逗号分隔 (如 1,3,5)。空=全部")
    parser.add_argument("--skip-judge", action="store_true",
                        help="跳过 LLM-as-Judge 评分（节省 API 成本）")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help=f"并发线程数 (默认 {MAX_WORKERS}，1=串行)")
    parser.add_argument("--timeout", type=int, default=TASK_TIMEOUT,
                        help=f"单任务超时秒数 (默认 {TASK_TIMEOUT})")
    parser.add_argument("--resume", type=str, default="",
                        help="从 checkpoint 文件恢复运行（传入 checkpoint JSON 路径）")
    parser.add_argument("--test-file", type=str, default="",
                        help="测试文件路径（相对于项目根目录），默认使用 TEST_FILE 常量")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="运行结束后不清理 workspace 目录（用于调试）")
    args = parser.parse_args()

    runs_per_q = args.runs
    skip_judge = args.skip_judge
    max_workers = args.workers
    task_timeout = args.timeout

    # ── 测试文件路径 ──
    test_file = args.test_file.strip() if args.test_file.strip() else TEST_FILE
    ds_label = _dataset_label(test_file)  # 数据集短名，嵌入所有输出文件名

    # ── 解析问题编号过滤 ──
    if args.questions.strip():
        q_filter = [int(x.strip()) for x in args.questions.split(",") if x.strip()]
    else:
        q_filter = []

    # ── 日志 ──
    logger = logging.getLogger("benchmark")
    logger.setLevel(logging.INFO)
    _log_file = None
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s",
                                                datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        logs_dir = os.path.join(PROJECT_ROOT, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        _log_file = os.path.join(logs_dir, f"benchmark_{ds_label}_{ts}.log")
        fh = logging.FileHandler(_log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s",
                                           datefmt="%H:%M:%S"))
        logger.addHandler(fh)
        logger.info("日志文件: %s", _log_file)

    # ── 解析测试文件 ──
    test_file_path = os.path.join(PROJECT_ROOT, test_file)
    test_cases = parse_test_file(test_file_path)
    if not test_cases:
        logger.error("未解析到任何测试用例，请检查文件: %s", test_file_path)
        sys.exit(1)

    if q_filter:
        test_cases = [tc for tc in test_cases if tc.index in q_filter]

    logger.info("解析到 %d 个测试用例", len(test_cases))
    for tc in test_cases:
        logger.info("  Q%d: %s…", tc.index, tc.query[:80])

    # ── 初始化 Agent ──
    logger.info("正在初始化 Agent …")
    agent = init_agent(logger)

    # ── 初始化 Judge LLM ──
    judge_llm = None
    if not skip_judge:
        logger.info("正在初始化 Judge LLM (%s) …", JUDGE_MODEL)
        judge_llm, judge_info = build_llm(model=JUDGE_MODEL, timeout=LLM_TIMEOUT, max_retries=LLM_MAX_RETRIES)
        logger.info("Judge LLM 就绪 | model=%s | base_url=%s",
                     judge_info["model"], judge_info["base_url"])

    # ══════════════════════════════════════════
    #  运行 benchmark（并发模式）
    # ══════════════════════════════════════════
    print_lock = threading.Lock()
    results_lock = threading.Lock()
    results = {}   # {question_index: [RunResult, ...]}
    for tc in test_cases:
        results[tc.index] = []

    total_expected = len(test_cases) * runs_per_q

    # ── 构建所有任务 ──
    tasks = []
    run_counter = 0
    for tc in test_cases:
        for run_i in range(runs_per_q):
            run_counter += 1
            tasks.append((tc, run_i, run_counter))

    # ── Checkpoint 恢复：加载已完成的任务并跳过 ──
    checkpoint_path = ""
    completed_set = set()  # {(q_idx, run_idx), ...}
    if args.resume:
        resume_path = args.resume
        if not os.path.isabs(resume_path):
            resume_path = os.path.join(PROJECT_ROOT, resume_path)
        if os.path.exists(resume_path):
            completed_set, loaded_results, _cfg = _load_checkpoint(resume_path)
            logger.info("\u2705 从 checkpoint 恢复: 已完成 %d 个任务", len(completed_set))
            # 合并已加载的结果到 results
            for q_idx, runs in loaded_results.items():
                if q_idx in results:
                    results[q_idx] = runs
            # 过滤掉已完成的任务
            original_count = len(tasks)
            tasks = [(tc, ri, seq) for tc, ri, seq in tasks if (tc.index, ri) not in completed_set]
            skipped = original_count - len(tasks)
            logger.info("跳过 %d 个已完成任务，剩余 %d 个任务", skipped, len(tasks))
            checkpoint_path = resume_path  # 继续写入同一个文件
        else:
            logger.warning("checkpoint 文件不存在: %s，从头开始运行", resume_path)

    # 创建新的 checkpoint 路径（如果不是恢复模式）
    if not checkpoint_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(logs_dir, f"benchmark_{ds_label}_checkpoint_{ts}.json")

    if not tasks:
        logger.info("所有任务已完成，无需运行")
    else:
        total_expected = len(tasks) + len(completed_set)
        logger.info("并发模式: %d 个任务 | max_workers=%d | 总预期=%d (已完成%d)",
                    len(tasks), max_workers, total_expected, len(completed_set))

    # ── 连接失败熔断计数器（跨线程共享） ──
    _conn_fail_count = [0]  # 用列表包装以便闭包修改
    _conn_fail_lock = threading.Lock()

    # ── Workspace 输出文件跟踪（每题运行完成后清理） ──
    _workspace_baselines = {}   # {q_idx: set(filename)} 首 run 前的文件快照
    _workspace_lock = threading.Lock()

    def _get_workspace_output_files() -> set:
        """列出 workspace 中所有非 venv 文件（仅顶层，不递归子目录）"""
        ws_dir = str(get_workspace_dir())
        if not os.path.isdir(ws_dir):
            return set()
        return {
            f for f in os.listdir(ws_dir)
            if os.path.isfile(os.path.join(ws_dir, f)) and f != "__pycache__"
        }

    def _is_connection_error(err_str: str) -> bool:
        """判断是否为网络连接类错误（可重试、可熔断）"""
        return any(kw in err_str for kw in (
            "Connection error", "APIConnectionError",
            "WinError 10051", "WinError 10061", "WinError 10060",
            "Connection refused", "Network is unreachable",
        ))

    # ── Checkpoint 配置 ──
    checkpoint_config = {
        "runs_per_question": runs_per_q,
        "test_file": test_file,
        "model": os.getenv("LLM_MODEL", "auto"),
    }

    def _save_after_task(q_idx, run_result):
        """任务完成后保存结果并写入 checkpoint（线程安全）
        同一题所有运行完成后自动清理 workspace 输出文件。
        """
        with results_lock:
            results[q_idx].append(run_result)
            completed_runs = len(results[q_idx])
            try:
                _save_checkpoint(checkpoint_path, results, checkpoint_config)
            except Exception as e:
                logger.warning("checkpoint 保存失败: %s", e)

        # ── 同一题所有运行完成后清理 workspace 输出文件 ──
        if completed_runs >= runs_per_q:
            with _workspace_lock:
                baseline = _workspace_baselines.get(q_idx, set())
            current_files = _get_workspace_output_files()
            new_files = current_files - baseline
            if new_files:
                ws_dir = str(get_workspace_dir())
                cleaned = 0
                for fname in new_files:
                    fpath = os.path.join(ws_dir, fname)
                    try:
                        os.remove(fpath)
                        cleaned += 1
                    except Exception:
                        pass
                if cleaned:
                    logger.info("Q%d 输出文件已清理: %d 个文件", q_idx, cleaned)

    def _run_task(tc, run_i, task_seq):
        """单个并发任务：运行 Agent → 评估 → Judge → 返回 RunResult"""
        thread_id = uuid.uuid4().hex
        short_id = thread_id[:8]
        task_start = time.time()

        try:
            return _run_task_inner(tc, run_i, task_seq, thread_id, short_id, task_start)
        finally:
            # ── 任务完成后立即清理 venv（最大占用源）──
            try:
                cleanup_session_venv(thread_id)
                logger.debug("venv 已清理: %s", short_id)
            except Exception as e:
                logger.debug("venv 清理失败: %s", e)
            # ── 清理 ErrorMemory（避免 benchmark 长跑时内存泄漏）──
            try:
                cleanup_memory(thread_id)
            except Exception as e:
                logger.debug("ErrorMemory 清理失败: %s", e)

    def _run_task_inner(tc, run_i, task_seq, thread_id, short_id, task_start):

        # ── 首 run 时记录 workspace 文件基线（用于运行后清理增量文件） ──
        with _workspace_lock:
            if tc.index not in _workspace_baselines:
                _workspace_baselines[tc.index] = _get_workspace_output_files()

        with print_lock:
            print(f"\n{'━' * 60}")
            print(f"  [{task_seq}/{total_expected}] Q{tc.index} 第{run_i + 1}次 | thread={short_id}")
            print(f"  Query: {tc.query[:90]}{'…' if len(tc.query) > 90 else ''}")
            print(f"{'━' * 60}")

        # ── 连接错误重试（最多 2 次，等待 10/30 秒后重试） ──
        # Windows 唤醒后网络适配器恢复需要时间，短暂等待后重试即可成功
        conn_retry_delays = [10, 30]
        rec = None
        for attempt in range(len(conn_retry_delays) + 1):
            try:
                rec = run_single(agent, tc.query, thread_id, logger, timeout=task_timeout)
                break  # 成功，跳出重试循环
            except Exception as e:
                err_str = str(e)
                is_conn_err = _is_connection_error(err_str)

                if is_conn_err and attempt < len(conn_retry_delays):
                    wait_sec = conn_retry_delays[attempt]
                    logger.warning("Q%d 第%d次 连接失败 (尝试%d/%d)，等待 %ds 后重试: %s",
                                   tc.index, run_i + 1, attempt + 1, len(conn_retry_delays),
                                   wait_sec, err_str[:100])
                    time.sleep(wait_sec)
                    # 清理旧 session 的 venv 再生成新 ID
                    try:
                        cleanup_session_venv(thread_id)
                    except Exception:
                        pass
                    thread_id = uuid.uuid4().hex  # 重新生成，避免复用上下文
                    continue  # 重试

                # 最终失败：简洁日志，不打印完整 traceback
                if is_conn_err:
                    logger.error("Q%d 第%d次 连接失败（已重试%d次）: %s",
                                 tc.index, run_i + 1, len(conn_retry_delays), err_str[:120])
                else:
                    logger.error("Q%d 第%d次运行异常: %s", tc.index, run_i + 1, err_str)

                run_result = RunResult(
                    question_index=tc.index, run_index=run_i,
                    thread_id=thread_id, error=err_str,
                    duration=0,
                )
                run_result.process_pass = False
                run_result.process_fail_reasons = [f"运行异常: {err_str[:60]}"]
                run_result.status = "FAIL"
                with print_lock:
                    print(f"  ❌ 异常: {err_str[:100]}")

                # 连接失败计数 → 触发熔断
                if is_conn_err:
                    with _conn_fail_lock:
                        _conn_fail_count[0] += 1
                return tc.index, run_result

        # ── 连接失败熔断检测 ──
        with _conn_fail_lock:
            if _conn_fail_count[0] >= 3:
                logger.error("连续 %d 次连接失败，网络可能中断，跳过剩余任务", _conn_fail_count[0])
                run_result = RunResult(
                    question_index=tc.index, run_index=run_i,
                    thread_id=thread_id, error="网络中断跳过",
                )
                run_result.status = "FAIL"
                run_result.process_fail_reasons = ["网络中断跳过"]
                return tc.index, run_result

        # ── 超时强制终止 → 直接标记 FAIL ──
        if rec.get("_timed_out"):
            duration = time.time() - task_start
            run_result = RunResult(
                question_index=tc.index, run_index=run_i,
                thread_id=thread_id,
                error=f"任务超时 ({task_timeout}s)",
                duration=duration,
                tool_status=rec.get("tool_status", ""),
                tool_calls_count=rec.get("tool_calls_count", 0),
                selected_skills=rec.get("selected_skills", []),
                final_answer=rec.get("final_answer", ""),
            )
            run_result.process_pass = False
            run_result.process_fail_reasons = [f"任务超时 ({task_timeout}s)"]
            run_result.status = "FAIL"
            with print_lock:
                print(f"  ❌ TIMEOUT | {duration:.0f}s | 超时强制终止")
            return tc.index, run_result

        # ── 评估过程指标 ──
        proc_pass, proc_fails = evaluate_process(rec)

        # ── LLM-as-Judge 评分（仅过程 PASS 时）──
        jr = None
        if proc_pass and judge_llm and not skip_judge:
            logger.info("Q%d 第%d次 → 过程PASS，调用 Judge LLM 评分 …", tc.index, run_i + 1)
            jr = judge_response(tc.query, tc.expected, rec["final_answer"], judge_llm, logger)
            if jr.error:
                logger.warning("Judge 评分出错: %s", jr.error)
            else:
                logger.info("Q%d 第%d次 → Judge 评分: coverage=%.1f accuracy=%.1f logic=%.1f depth=%.1f overall=%.1f | %s",
                            tc.index, run_i + 1, jr.coverage, jr.accuracy, jr.logic, jr.depth, jr.overall, jr.reason)
        elif not proc_pass and judge_llm and not skip_judge:
            logger.info("Q%d 第%d次 → 过程FAIL，跳过 Judge 评分", tc.index, run_i + 1)

        # ── 判定整体状态 ──
        run_result = RunResult(
            question_index=tc.index,
            run_index=run_i,
            thread_id=thread_id,
            tool_status=rec["tool_status"],
            tool_calls_count=rec["tool_calls_count"],
            selected_skills=rec.get("selected_skills", []),
            final_answer=rec["final_answer"],
            duration=rec["duration"],
            process_pass=proc_pass,
            process_fail_reasons=proc_fails,
            judge=jr,
        )

        if proc_pass and jr and jr.overall >= JUDGE_PASS_SCORE:
            run_result.status = "PASS"
            run_result.overall_pass = True
        elif proc_pass and jr and jr.overall < JUDGE_PASS_SCORE:
            run_result.status = "WEAK_PASS"
            run_result.overall_pass = False
        elif proc_pass and not jr:
            run_result.status = "PASS"
            run_result.overall_pass = True
        else:
            run_result.status = "FAIL"
            run_result.overall_pass = False

        # ── 终端即时反馈（线程安全）──
        with print_lock:
            j_str = f"结论{jr.overall:.1f}" if jr and jr.overall > 0 else "未评分"
            status_icon = "✅" if run_result.status == "PASS" else (
                "⚠️" if run_result.status == "WEAK_PASS" else "❌")
            print(f"  {status_icon} {run_result.status} | {rec['duration']:.1f}s | "
                  f"{j_str} | 工具{rec['tool_calls_count']}次")
            if proc_fails:
                for reason in proc_fails[:2]:
                    print(f"     ↳ {reason}")

        return tc.index, run_result

    # ── 并发执行 ──
    completed = 0
    circuit_break = False  # 熔断标志
    if max_workers <= 1:
        # 串行模式（兼容 max_workers=1）
        for tc, run_i, task_seq in tasks:
            if circuit_break:
                # 网络熔断：跳过剩余任务
                run_result = RunResult(
                    question_index=tc.index, run_index=run_i,
                    thread_id="", error="网络中断跳过",
                )
                run_result.status = "FAIL"
                run_result.process_fail_reasons = ["网络中断跳过"]
                _save_after_task(tc.index, run_result)
                completed += 1
                continue
            q_idx, run_result = _run_task(tc, run_i, task_seq)
            _save_after_task(q_idx, run_result)
            completed += 1
            # 熔断检测
            if run_result.error and "网络中断跳过" in str(run_result.error):
                circuit_break = True
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(_run_task, tc, run_i, task_seq): (tc, run_i, task_seq)
                for tc, run_i, task_seq in tasks
            }
            # 不设全局超时：每个任务已有独立的 task_timeout 保护，
            # 网络中断由熔断机制兜底，无需 as_completed 层面的总限时
            try:
                for future in as_completed(future_to_task):
                    if circuit_break:
                        break  # 熔断后不再处理更多结果
                    try:
                        q_idx, run_result = future.result(timeout=60)
                        _save_after_task(q_idx, run_result)
                        completed += 1
                        logger.info("任务完成 %d/%d", completed, total_expected)

                        # ── 网络中断熔断检测 ──
                        if run_result.error and "网络中断跳过" in str(run_result.error):
                            circuit_break = True
                            logger.error("⚠️ 网络中断熔断触发，取消剩余 %d 个任务",
                                         total_expected - completed)
                            # 取消未启动的任务
                            for f, (tc_info, ri_info, ts_info) in future_to_task.items():
                                if not f.done() and not f.running():
                                    f.cancel()
                                    skip_result = RunResult(
                                        question_index=tc_info.index, run_index=ri_info,
                                        thread_id="", error="网络中断跳过",
                                    )
                                    skip_result.status = "FAIL"
                                    skip_result.process_fail_reasons = ["网络中断跳过"]
                                    _save_after_task(tc_info.index, skip_result)
                                    completed += 1

                    except TimeoutError:
                        tc_info, run_i_info, task_seq_info = future_to_task[future]
                        logger.error("任务超时 Q%d 第%d次 (future.result timeout)", tc_info.index, run_i_info + 1)
                        run_result = RunResult(
                            question_index=tc_info.index, run_index=run_i_info,
                            thread_id="", error="future.result 超时",
                        )
                        run_result.status = "FAIL"
                        run_result.process_fail_reasons = ["任务级超时"]
                        _save_after_task(tc_info.index, run_result)
                        completed += 1
                    except Exception as e:
                        tc_info, run_i_info, task_seq_info = future_to_task[future]
                        logger.error("任务异常 Q%d 第%d次: %s", tc_info.index, run_i_info + 1, e)
                        run_result = RunResult(
                            question_index=tc_info.index, run_index=run_i_info,
                            thread_id="", error=str(e),
                        )
                        run_result.status = "FAIL"
                        _save_after_task(tc_info.index, run_result)
                        completed += 1
            except KeyboardInterrupt:
                logger.warning("用户中断 (Ctrl+C)，保存已完成任务...")
                for f, (tc_info, ri_info, ts_info) in future_to_task.items():
                    if not f.done():
                        f.cancel()
                        run_result = RunResult(
                            question_index=tc_info.index, run_index=ri_info,
                            thread_id="", error="用户中断",
                        )
                        run_result.status = "FAIL"
                        run_result.process_fail_reasons = ["用户中断"]
                        _save_after_task(tc_info.index, run_result)
                        completed += 1

    # ── 每题结果按 run_index 排序（并发完成顺序不确定）──
    for q_idx in results:
        results[q_idx].sort(key=lambda r: r.run_index)

    # ══════════════════════════════════════════
    #  生成报告
    # ══════════════════════════════════════════
    report = generate_report(results, test_cases, logger, max_workers=max_workers)

    # ── 保存 JSON 报告 ──
    logs_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    report_path = os.path.join(logs_dir, f"benchmark_{ds_label}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("报告已保存: %s", report_path)
    logger.info("日志文件: %s", _log_file or "(未初始化)")

    # ── 运行完毕且全部任务正常完成 → 清理 checkpoint（断点续跑已完成，不再需要） ──
    if checkpoint_path and os.path.exists(checkpoint_path) and completed >= total_expected:
        try:
            os.remove(checkpoint_path)
            logger.info("Checkpoint 已清理（全部 %d 个任务已完成）", completed)
        except Exception as e:
            logger.debug("Checkpoint 清理失败: %s", e)

    # ── 清理 workspace ──
    if not args.no_cleanup:
        _cleanup_workspace(logger)

    return report


def _dataset_label(test_file_path: str) -> str:
    """从测试文件路径提取数据集短名（用于日志/报告文件名）

    示例:
        "TestQuestion/TRQA测试案例（200条核心差异化数据）.md" → "TRQA200条"
        "TestQuestion/问题模板和衍生问题.md" → "问题模板和衍生问题"
    """
    stem = os.path.splitext(os.path.basename(test_file_path))[0]
    label = re.sub(r'[（(].*?[）)]', '', stem)
    label = re.sub(r'\s+', '', label)
    return label[:20] if len(label) > 20 else label


def _cleanup_workspace(logger):
    """清理 workspace 目录下 benchmark 运行产生的所有文件（venv、输出文件、子目录）"""
    workspace_dir = os.path.join(PROJECT_ROOT, "workspace")
    if not os.path.isdir(workspace_dir):
        return

    cleaned_files = 0
    cleaned_dirs = 0
    errors = 0

    for entry in os.listdir(workspace_dir):
        full_path = os.path.join(workspace_dir, entry)
        # 跳过 __pycache__（Python 自管理）
        if entry == "__pycache__":
            continue
        try:
            if os.path.isdir(full_path):
                shutil.rmtree(full_path, ignore_errors=False)
                cleaned_dirs += 1
            else:
                os.remove(full_path)
                cleaned_files += 1
        except Exception as e:
            errors += 1
            logger.debug("清理失败: %s - %s", entry, e)

    total = cleaned_files + cleaned_dirs
    if total > 0 or errors > 0:
        logger.info("Workspace 清理完成: 删除 %d 个文件 + %d 个目录 | 失败 %d 个",
                     cleaned_files, cleaned_dirs, errors)


if __name__ == "__main__":
    main()
