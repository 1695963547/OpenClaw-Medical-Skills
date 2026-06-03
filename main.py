"""
医疗 AI 智能体 — 主入口模块
=============================

本模块是整个系统的启动入口，负责：
1. 初始化日志系统（控制台 + 文件双通道）
2. 加载 LLM 模型配置（支持本地配置文件 + 环境变量覆盖）
3. 初始化技能检索器（基于 ChromaDB 的语义检索，869 个生物医学技能）
4. 启动交互式命令行循环（支持多轮对话、上下文记忆）
5. 实时展示 Agent 执行过程（自动检索 → 计划生成 → 工具调用 → 结果输出）
6. 每轮请求落盘 JSONL 记录，便于后续回放和统计分析

架构定位：
    main.py ──→ MedicalSkillAgent (agent.py)
                    ├── auto_retrieve_node  自动检索
                    ├── agent_node          LLM 决策
                    ├── tools_node          工具执行 (tools.py)
                    ├── post_tools_node     后处理/错误修正
                    └── planner_node        多步骤计划生成
"""
import os
import json
import logging
import re
import uuid
from datetime import datetime

# ── 内部模块导入 ──
from src.skill_retriever import SkillRetriever      # 技能语义检索器（ChromaDB + sentence-transformers）
from src.agent import MedicalSkillAgent              # 核心 LangGraph ReAct 智能体
from src.llm_factory import build_llm, DEFAULT_MODEL, load_local_llm_settings  # LLM 工厂（构建 ChatOpenAI 实例）
from src.tools import set_session_id, _executor      # 工具运行时（会话 ID 管理 + 代码执行器）
from src.middleware.error_memory import set_memory_session_id, cleanup_memory  # 错误记忆（与 tools 共用 session_id）

# ─── 日志配置 ───

def setup_logging(level: str = "INFO") -> logging.Logger:
    """配置结构化日志，替代散落的 print 调用。

    日志格式：时间戳 [级别] 来源 - 消息
    - INFO: 正常流程（模型决策、工具调用、执行状态）
    - DEBUG: 详细输出（工具原始返回、完整代码参数）
    - WARNING: 异常但可继续（API 限流、包安装失败、迭代耗尽）
    - ERROR: 严重错误（LLM 调用失败、未捕获异常）
    """
    logger = logging.getLogger("agent")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 避免重复添加 handler（main 可能被多次调用）
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # 同时写入文件，保留完整运行记录
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(logs_dir, "agent_run.log"), encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)  # 文件记录所有级别
    logger.addHandler(file_handler)

    return logger


def log_run(record: dict, logger: logging.Logger):
    """每轮请求落一条 JSON 记录，便于后续回放和自动化对比。"""
    # 写入独立的 JSONL 文件
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    with open(os.path.join(logs_dir, "agent_run.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    # 同时用 logger 记录摘要
    logger.info(
        "请求完成 | skills=%s | 选中=%s | 状态=%s | 用时=%s",
        record.get("retrieved_skills", [])[:3],  # 只展示前3个
        record.get("selected_skill") or "无",
        record.get("tool_status") or "未调用",
        record.get("duration", "未知"),
    )


def setup_langsmith(logger: logging.Logger):
    """如果存在 LangSmith 配置，则开启链路观测。"""
    local = load_local_llm_settings()
    api_key = os.getenv("LANGSMITH_API_KEY") or local.get("langsmith_api_key") or ""
    if not api_key:
        return False
    project = os.getenv("LANGSMITH_PROJECT") or local.get("langsmith_project") or "869skills"
    endpoint = os.getenv("LANGSMITH_ENDPOINT") or local.get("langsmith_endpoint") or "https://api.smith.langchain.com"
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_PROJECT"] = project
    os.environ["LANGSMITH_ENDPOINT"] = endpoint
    logger.info("LangSmith Tracing enabled | project=%s", project)
    return True


def main():
    """主函数：初始化系统并启动交互式对话循环。

    启动流程：
    1. 初始化日志系统
    2. 加载技能注册表（skill_registry.json，869 个技能的索引）
    3. 解析 LLM 配置（llm_local_config.py → 环境变量 → 默认值，三级覆盖）
    4. 构建 LLM 实例（通过 llm_factory 统一创建 ChatOpenAI）
    5. 解析运行时参数（SKILL_TOP_K、MAX_ITERATIONS、MAX_DISPLAY_LENGTH）
    6. 初始化 MedicalSkillAgent（含 LangGraph 图构建）
    7. 进入 while 循环等待用户输入
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    logger = setup_logging(log_level)

    # ── 步骤 1：初始化技能检索器 ──
    # skill_registry.json 包含 869 个生物医学技能的元数据（ID、描述、路径）
    # 检索器使用 ChromaDB + sentence-transformers 实现语义匹配
    retriever = SkillRetriever("skill_registry.json")

    # ── 步骤 2：解析 LLM 模型配置 ──
    # 三级覆盖优先级：llm_local_config.py → 环境变量 → 模型 profile 预设 → 默认值
    local_settings = load_local_llm_settings()
    profile = (local_settings["profile"] or os.getenv("LLM_PROFILE") or "").strip().lower()
    # 预设模型 profile：用户可通过 LLM_PROFILE 快速切换模型族
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

    # ── 步骤 3：初始化 LangSmith 链路观测（可选）──
    # 配置后可以在 LangSmith 平台上追踪每次 LLM 调用的完整链路
    setup_langsmith(logger)

    # ── 步骤 4：构建 LLM 实例 ──
    # build_llm() 根据模型名称自动选择 API base_url 和参数配置
    default_model = profiles.get(profile, {}).get("model", DEFAULT_MODEL)
    model = local_settings["model"] or os.getenv("LLM_MODEL", default_model)
    llm, llm_info = build_llm(model=model)

    # ── 步骤 5：解析运行时参数 ──
    # 参数优先级：环境变量 > llm_local_config.py 配置文件 > 代码默认值
    skill_top_k_str = os.getenv("SKILL_TOP_K") or local_settings.get("skill_top_k") or ""
    try:
        skill_top_k = int(skill_top_k_str.strip()) if skill_top_k_str.strip() else 8
    except ValueError:
        skill_top_k = 8

    # 解析 MAX_ITERATIONS（优先环境变量，其次本地配置，默认 20）。
    max_iter_str = os.getenv("MAX_ITERATIONS") or local_settings.get("max_iterations") or ""
    try:
        max_iterations = int(max_iter_str.strip()) if max_iter_str.strip() else 20
    except ValueError:
        max_iterations = 20

    # 解析 MAX_DISPLAY_LENGTH（优先环境变量，其次本地配置，默认 0 = 不截断）。
    max_display_str = os.getenv("MAX_DISPLAY_LENGTH") or local_settings.get("max_display_length") or ""
    try:
        max_display_length = int(max_display_str.strip()) if max_display_str.strip() else 0
    except ValueError:
        max_display_length = 0

    # ── 步骤 6：初始化 Agent 核心实例 ──
    # MedicalSkillAgent 内部会构建 LangGraph 有向图（5 个节点 + 条件路由）
    agent = MedicalSkillAgent(retriever=retriever, llm=llm, top_k=skill_top_k, max_iterations=max_iterations)

    logger.info("Agent 就绪 | model=%s | base_url=%s | top_k=%d | max_iterations=%d | max_display_length=%d", llm_info["model"], llm_info["base_url"], skill_top_k, max_iterations, max_display_length)
    print("Medical Skills Agent 就绪。输入 exit 退出。")

    # ── 步骤 7：多轮对话会话管理 ──
    # 使用持久 thread_id 实现多轮对话（基于 LangGraph MemorySaver）
    # 同一 thread_id 下的 messages 会跨轮次累积，Agent 拥有"记忆"
    session_thread_id = uuid.uuid4().hex

    # ══════════════════════════════════════════════════
    #  交互式对话主循环
    #  每次用户输入 → Agent 完整执行 pipeline → 输出结果
    # ══════════════════════════════════════════════════
    while True:
        try:
            user_input = input("\n> ")
            if user_input.lower() in ("exit", "quit"):
                break
            if not user_input.strip():
                continue

            print(f"👤 User: {user_input}")

            start_time = datetime.now()
            set_session_id(session_thread_id)
            set_memory_session_id(session_thread_id)
            logger.debug("会话 | thread_id=%s | query=%s", session_thread_id[:8], user_input[:100])

            # ── 初始化本轮请求的观测记录（用于日志/JSONL/统计分析）──
            run_record = {
                "timestamp": start_time.isoformat(),
                "thread_id": session_thread_id,
                "user_input": user_input,
                "retrieved_skills": [],          # 自动检索召回的技能列表
                "selected_skill": "",             # 最终选中的单个技能
                "selected_skills": [],            # 所有使用过的技能（去重列表）
                "used_skills": False,             # 是否调用了工具
                "tool_status": "未调用",           # 工具执行状态（成功/失败/超时等）
                "tool_calls_count": 0,            # 工具调用总次数（含 retrieve_skills/read_file/execute_code/update_task_status）
                "final_answer": "",               # Agent 最终回复文本
                "steps": [],                      # 执行步骤轨迹（每个节点的输出）
                "duration": "",                   # 总耗时
            }

            # ══════════════════════════════════════════════════════════
            #  Agent 流式执行（核心 pipeline）
            #  agent.stream() 逐节点 yield 事件，main.py 负责实时展示
            #  LangGraph 图流程：
            #    auto_retrieve → agent → tools → post_tools → planner → agent/END
            # ══════════════════════════════════════════════════════════
            for event in agent.stream(user_input, thread_id=session_thread_id):
                for node_name, node_state in event.items():
                    if node_state is None:
                        continue
                    # ── 节点 1：auto_retrieve（系统自动检索）──
                    # 在 LLM 决策之前，系统用语义检索预加载相关技能到上下文
                    if node_name == "auto_retrieve":
                        # 从自动检索节点捕获召回列表
                        retrieved = node_state.get("retrieved_skills", [])
                        if retrieved:
                            run_record["retrieved_skills"] = retrieved
                            run_record["steps"].append({"node": "auto_retrieve", "skills": retrieved})

                    # ── 节点 2：planner（多步骤计划生成）──
                    # 当有技能可用且任务较复杂时，LLM 生成结构化执行计划
                    # 计划会注入到后续 agent 的系统提示词中，引导按步骤执行
                    elif node_name == "planner":
                        plan = node_state.get("plan", "")
                        subtasks = node_state.get("subtasks", [])
                        if plan and subtasks:
                            print("\n" + "━" * 50)
                            print("📋 执行计划")
                            print("━" * 50)
                            for i, sub in enumerate(subtasks):
                                desc = sub.get("description", f"步骤 {i + 1}")
                                # 去掉编号前缀（如 "1. [类型] "）让显示更干净
                                clean_desc = re.sub(r"^\d+\.\s*\[?[^\]]*\]?\s*", "", desc, count=1)
                                icon = "⬜"
                                print(f"  {icon} Step {i + 1}: {clean_desc}")
                            print("━" * 50)
                            run_record["steps"].append({"node": "planner", "plan": plan})
                            # 记录初始 subtask 状态用于后续比较
                            run_record["_last_subtasks"] = [
                                {"description": s.get("description", ""), "status": s.get("status", "pending")}
                                for s in subtasks
                            ]

                    # ── 节点 3：agent（LLM 决策核心）──
                    # LLM 根据上下文决定：调用工具 or 直接回复用户
                    # 如果发出 tool_calls → 进入 tools 节点执行
                    # 如果直接输出 content → 本轮结束
                    elif node_name == "agent":
                        if "selected_skill" in node_state and node_state["selected_skill"]:
                            run_record["selected_skill"] = node_state["selected_skill"]

                        messages = node_state.get("messages", [])
                        if messages:
                            last_msg = messages[-1]
                            # 分支 A：LLM 决定调用工具（retrieve_skills / read_file / execute_code / update_task_status）
                            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                                run_record["used_skills"] = True
                                for tc in last_msg.tool_calls:
                                    # 累计工具调用次数
                                    run_record["tool_calls_count"] += 1
                                    # 收集使用过的 skill（去重）
                                    if tc["name"] == "read_file":
                                        sid = tc["args"].get("skill_id", "")
                                        if sid and sid not in run_record["selected_skills"]:
                                            run_record["selected_skills"].append(sid)
                                    elif tc["name"] == "execute_code":
                                        sid = tc["args"].get("related_skill", "")
                                        if sid and sid not in run_record["selected_skills"]:
                                            run_record["selected_skills"].append(sid)
                                    # 终端/日志只展示裁剪后的 code 参数
                                    display_args = {}
                                    for k, v in tc["args"].items():
                                        if k == "code" and isinstance(v, str) and len(v) > 200:
                                            display_args[k] = v[:200] + f"... (共 {len(v)} 字符)"
                                        else:
                                            display_args[k] = v
                                    logger.info("模型决策 → 调用工具: %s | 参数: %s", tc["name"], display_args)
                                    # 完整参数记录到 DEBUG 级别
                                    logger.debug("工具调用完整参数: %s | args=%s", tc["name"], tc["args"])
                                    run_record["steps"].append({"node": "agent_tool_call", "tool": tc["name"], "args": tc["args"]})
                                    if tc["name"] == "retrieve_skills":
                                        run_record["retrieved_skills"] = ["(等待工具执行结果)"]
                            elif last_msg.content:
                                # 分支 B：LLM 直接回复用户（不调用工具，常见于日常对话或最终总结）
                                if not run_record["used_skills"]:
                                    logger.info("模型决策 → 未调用工具，直接回复")
                                    run_record["steps"].append({"node": "agent_decision"})

                                run_record["final_answer"] = last_msg.content
                                # 终端显示：受 max_display_length 控制截断
                                if max_display_length > 0 and len(last_msg.content) > max_display_length:
                                    display_answer = last_msg.content[:max_display_length] + f"\n... (共 {len(last_msg.content)} 字符，已截断。修改 MAX_DISPLAY_LENGTH 可调整)"
                                else:
                                    display_answer = last_msg.content
                                print(f"\n🤖 Agent:\n{display_answer}")
                                # 完整内容记录到日志 DEBUG 级别
                                logger.debug("模型完整回复: %s", last_msg.content)
                                run_record["steps"].append({"node": "agent_reply"})

                    # ── 节点 4：tools（工具执行）──
                    # ToolNode 自动执行 LLM 请求的工具调用，返回 ToolMessage
                    # 可能的工具：retrieve_skills / read_file / execute_code / update_task_status
                    elif node_name == "tools":
                        messages = node_state.get("messages", [])
                        if messages:
                            last_msg = messages[-1]
                            # 工具输出裁剪后 INFO，完整内容 DEBUG
                            out = last_msg.content
                            out_preview = out[:500] if len(out) > 500 else out
                            if len(out) > 500:
                                out_preview += "\n... (已省略后续输出)"
                            logger.info("工具执行结果 (%s): %s", last_msg.name, out_preview)
                            logger.debug("工具原始输出 (%s): %s", last_msg.name, out)
                            run_record["steps"].append({"node": "tools_result", "tool": last_msg.name})

                            # 解析 retrieve_skills 的召回列表
                            if last_msg.name == "retrieve_skills":
                                match = re.search(r"已检索到 \d+ 个相关技能[：:]\s*(.+)", last_msg.content)
                                if match:
                                    run_record["retrieved_skills"] = [s.strip() for s in match.group(1).split(",")]
                                    logger.info("检索阶段 → 召回 %d 个技能: %s", len(run_record["retrieved_skills"]), run_record["retrieved_skills"])
                                    run_record["steps"].append({"node": "retrieve", "skills": run_record["retrieved_skills"]})

                    # ── 节点 5：post_tools（后处理/错误修正/Stuck检测）──
                    # 解析工具执行结果，分类错误类型，注入修正建议
                    # 同时追踪子任务进度变化，更新 plan 中的步骤状态
                    elif node_name == "post_tools":
                        tool_status = node_state.get("tool_status", "")
                        if tool_status:
                            run_record["tool_status"] = tool_status
                            if tool_status == "成功":
                                logger.info("执行状态: %s", tool_status)
                            else:
                                logger.warning("执行状态: %s", tool_status)
                            run_record["steps"].append({"node": "post_tools", "status": tool_status})

                        # ── 实时展示子任务进度变化 ──
                        subtasks = node_state.get("subtasks", [])
                        last_subtasks = run_record.get("_last_subtasks", [])
                        if subtasks:
                            for i, sub in enumerate(subtasks):
                                sub_status = sub.get("status", "pending")
                                # 获取上次状态
                                last_status = "pending"
                                if i < len(last_subtasks):
                                    last_status = last_subtasks[i].get("status", "pending")
                                # 状态发生变化时打印
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
                                    print(f"  {icon} [{i + 1}/{len(subtasks)}] {sub_status}: {clean_desc}")
                            # 更新追踪状态
                            run_record["_last_subtasks"] = [
                                {"description": s.get("description", ""), "status": s.get("status", "pending")}
                                for s in subtasks
                            ]

            # ══════════════════════════════════════════
            #  本轮请求完成 → 输出总结日志
            # ══════════════════════════════════════════
            duration = (datetime.now() - start_time).total_seconds()
            run_record["duration"] = f"{duration:.1f}s"

            logger.info("=" * 40)
            logger.info("本次请求总结")
            logger.info("  召回 skills: %s", run_record["retrieved_skills"])
            logger.info("  是否使用 skills: %s", "是" if run_record["used_skills"] else "否")
            logger.info("  最终选择的 skill: %s", run_record["selected_skill"] or "无")
            if run_record["selected_skills"]:
                logger.info("  使用的 skills: %s", run_record["selected_skills"])
            logger.info("  工具调用次数: %d", run_record["tool_calls_count"])
            logger.info("  执行状态: %s", run_record["tool_status"])
            logger.info("  用时: %.1fs", duration)

            # ── 异常状态检测与告警 ──
            if run_record["tool_status"] in ("未完成（迭代耗尽）",):
                logger.warning("任务未完整执行：Agent 用尽了迭代次数")
            elif run_record["tool_status"] == "成功" and run_record["used_skills"] and not run_record["final_answer"]:
                logger.warning("工具执行成功但无最终回复")
            elif run_record["tool_status"] == "成功" and not run_record["final_answer"]:
                logger.warning("状态标记成功但未输出分析结果")
            logger.info("=" * 40)

            # ── 持久化：JSONL 记录 + 技能统计落盘 ──
            log_run(run_record, logger)
            agent.stats.save()
            # ── 虚拟环境生命周期管理（Codex 策略）──
            # venv 按 session_id 隔离，跨轮复用直到程序退出
            # cleanup_all_sessions() 在 exit/quit 后统一销毁，零残留
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error("未捕获异常: %s", e, exc_info=True)

    # 程序退出时清理所有残留 venv
    _executor.cleanup_all_sessions()


if __name__ == "__main__":
    main()
