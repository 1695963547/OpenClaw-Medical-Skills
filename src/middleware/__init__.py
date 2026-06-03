"""稳健执行中间层（Robust Execution Middleware）

本包是 Agent 系统的"防错缓冲层"，与具体 skill / 具体 API 解耦。
新增任何 skill 或接入任何新数据源时，本层自动复用。

子模块：
  - tool_param_adapter   工具参数归一化（P1 解决 LLM 参数幻觉）
  - error_memory         错误记忆与策略阶梯（P6 解决错误复读机）
  - schema_healer        GraphQL 字段自愈（待实施 P2）
  - resilient_executor   平台/路径/网络韧性（待实施 P3/P4/P5）
  - llm_retry_wrapper    LLM 客户端容错（待实施 P8）
"""
