"""项目配置模板（示例文件）。

使用步骤：
1. 复制本文件为 `llm_local_config.py`：
   - Windows (PowerShell):  Copy-Item llm_config.example.py llm_local_config.py
   - macOS / Linux:         cp llm_config.example.py llm_local_config.py

2. 打开 `llm_local_config.py`，把下方所有 `YOUR_*` 占位符替换为你的真实值。

3. 重新运行：
       python .\\main.py

注意：`llm_local_config.py` 已在 `.gitignore` 中被忽略，永远不会被推送到 GitHub。
"""

# ─── LLM Provider ────────────────────────────────────────────
# 在哪里调用 LLM（OpenAI 兼容协议）
LLM_BASE_URL = "https://api.openai.com/v1"          # 例如：https://api.openai.com/v1

# 你的 API Key（请妥善保管，不要提交到任何公开仓库）
LLM_API_KEY = "YOUR_API_KEY_HERE"

# 使用的模型名称
LLM_MODEL = "gpt-4o-mini"                            # 或 "deepseek-chat"、"gemini-1.5-pro" 等

# 配置档位（不同档位对应不同提示词策略，可选 "default" / "custom"）
LLM_PROFILE = "default"

# ─── Agent 行为参数 ─────────────────────────────────────────
# 每轮最多检索的技能数量（越大越慢，但工具越准）
SKILL_TOP_K = "3"

# Agent 最大推理步数（防止无限循环）
MAX_ITERATIONS = "40"

# 终端显示 Agent 回复的最大字符数（0 = 不截断）
MAX_DISPLAY_LENGTH = "0"

# ─── 本地向量模型 ────────────────────────────────────────────
# 技能检索使用的 sentence-transformers 模型路径
# 下载地址：https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
SENTENCE_TRANSFORMER_MODEL = r"./models/paraphrase-multilingual-MiniLM-L12-v2"

# ─── LangSmith 观测（可选）───────────────────────────────────
# 填入 API Key 后启用 LangSmith 追踪
# 申请地址：https://smith.langchain.com
LANGSMITH_API_KEY = ""                                # 留空表示不启用
LANGSMITH_PROJECT = "869skills"
LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
