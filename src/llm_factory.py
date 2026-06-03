import json
import logging
import os

from langchain_openai import ChatOpenAI

_logger = logging.getLogger("agent.llm_factory")


def _patch_langchain_openai_for_gateway_errors():
    """Monkey-patch langchain_openai 的 _create_chat_result 方法，
    在 choices 为 null 时打印 API 网关的错误详情（msg/code/success）。

    背景：某些 API 网关（如阿里云、vLLM）在请求失败时返回：
      {"choices": null, "code": "...", "success": false, "msg": "错误原因"}
    LangChain 只把 keys 打印到 TypeError 消息里，不输出 msg 的值，
    导致真正的错误原因被吞掉。此 patch 在 TypeError 抛出前打印网关错误。
    """
    try:
        from langchain_openai.chat_models.base import BaseChatOpenAI

        _original_create_chat_result = BaseChatOpenAI._create_chat_result

        def _patched_create_chat_result(self, response, generation_info=None):
            # 在原始方法抛 TypeError 之前，先检查并打印网关错误字段
            try:
                response_dict = (
                    response
                    if isinstance(response, dict)
                    else response.model_dump(
                        exclude={"choices": {"__all__": {"message": {"parsed"}}}}
                    )
                )
                if response_dict.get("choices") is None:
                    gw_msg = response_dict.get("msg", "")
                    gw_code = response_dict.get("code", "")
                    gw_success = response_dict.get("success", "")
                    if gw_msg:
                        _logger.error("Gateway Error: code=%s, success=%s, msg=%s", gw_code, gw_success, gw_msg)
                    else:
                        # 没有 msg 字段，打印完整响应供排查（截断防刷屏）
                        _logger.error("Gateway Error: null choices. Body: %s",
                              json.dumps(response_dict, ensure_ascii=False, default=str)[:2000])
            except Exception:
                pass

            return _original_create_chat_result(self, response, generation_info=generation_info)

        BaseChatOpenAI._create_chat_result = _patched_create_chat_result
    except Exception as e:
        _logger.warning("Failed to patch langchain_openai for gateway error logging: %s", e)


# 模块加载时自动应用 patch
_patch_langchain_openai_for_gateway_errors()


# Manual override entrypoint:
# edit these defaults if you want to switch gateway/model without env vars.
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"


def load_local_llm_settings() -> dict[str, str | None]:
    """Load optional project-local overrides from `llm_local_config.py`."""
    try:
        import llm_local_config as local_config
    except Exception:
        return {
            "api_key": None,
            "base_url": None,
            "model": None,
            "profile": None,
            "thinking": None,
            "reasoning_effort": None,
            "thinking_apply": None,
            "skill_top_k": None,
            "max_iterations": None,
            "max_display_length": None,
            "sentence_transformer_model": None,
        }

    def read_attr(name: str) -> str | None:
        value = getattr(local_config, name, None)
        if not isinstance(value, str):
            return None
        value = value.strip()
        return value or None

    return {
        "api_key": read_attr("LLM_API_KEY"),
        "base_url": read_attr("LLM_BASE_URL"),
        "model": read_attr("LLM_MODEL"),
        "profile": read_attr("LLM_PROFILE"),
        "thinking": read_attr("LLM_THINKING"),
        "reasoning_effort": read_attr("LLM_REASONING_EFFORT"),
        "thinking_apply": read_attr("LLM_THINKING_APPLY"),
        "langsmith_api_key": read_attr("LANGSMITH_API_KEY"),
        "langsmith_project": read_attr("LANGSMITH_PROJECT"),
        "langsmith_endpoint": read_attr("LANGSMITH_ENDPOINT"),
        "skill_top_k": read_attr("SKILL_TOP_K"),
        "max_iterations": read_attr("MAX_ITERATIONS"),
        "max_display_length": read_attr("MAX_DISPLAY_LENGTH"),
        "sentence_transformer_model": read_attr("SENTENCE_TRANSFORMER_MODEL"),
    }


def resolve_llm_settings(
    model: str | None = None,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> tuple[str, str | None, str | None]:
    local_settings = load_local_llm_settings()
    resolved_model = (
        model
        or local_settings["model"]
        or os.getenv("LLM_MODEL")
        or DEFAULT_MODEL
    )
    resolved_api_key = (
        api_key
        or local_settings["api_key"]
        or os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("ZHIPUAI_API_KEY")
    )
    resolved_base_url = (
        base_url
        or local_settings["base_url"]
        or os.getenv("LLM_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("ZHIPUAI_BASE_URL")
        or DEFAULT_BASE_URL
    )
    return resolved_model, resolved_api_key, resolved_base_url


def build_llm(
    model: str | None = None,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> tuple[ChatOpenAI, dict]:
    resolved_model, resolved_api_key, resolved_base_url = resolve_llm_settings(
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    local_settings = load_local_llm_settings()
    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
    timeout_s = timeout if timeout is not None else float(os.getenv("LLM_TIMEOUT", "180"))
    max_retries = max_retries if max_retries is not None else int(os.getenv("LLM_MAX_RETRIES", "2"))

    if not resolved_api_key:
        raise RuntimeError("Missing API key. Set LLM_API_KEY (or OPENAI_API_KEY / ZHIPUAI_API_KEY).")
    if not resolved_base_url:
        raise RuntimeError("Missing base url.")

    llm_kwargs = {
        "api_key": resolved_api_key,
        "base_url": resolved_base_url,
    }

    thinking_mode = (
        os.getenv("LLM_THINKING")
        or local_settings.get("thinking")
        or "auto"
    )
    thinking_apply = (
        os.getenv("LLM_THINKING_APPLY")
        or local_settings.get("thinking_apply")
        or "auto"
    )
    reasoning_effort = (
        os.getenv("LLM_REASONING_EFFORT")
        or local_settings.get("reasoning_effort")
        or None
    )

    thinking_mode_norm = thinking_mode.strip().lower()
    thinking_apply_norm = thinking_apply.strip().lower()

    model_lc = (resolved_model or "").lower()
    base_url_lc = (resolved_base_url or "").lower()
    is_deepseek = "deepseek" in model_lc or "deepseek" in base_url_lc
    should_apply_thinking = (
        thinking_apply_norm in {"always", "1", "true", "yes", "y"}
        or (thinking_apply_norm in {"auto", "deepseek"} and is_deepseek)
    )

    if should_apply_thinking and thinking_mode_norm not in {"auto", ""}:
        if thinking_mode_norm in {"disabled", "disable", "off", "false", "0"}:
            llm_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            reasoning_effort = None
        elif thinking_mode_norm in {"enabled", "enable", "on", "true", "1"}:
            llm_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

    if should_apply_thinking and reasoning_effort:
        llm_kwargs["reasoning_effort"] = reasoning_effort.strip()

    # ── 显式创建 httpx.Client 确保超时在底层 HTTP 客户端生效 ──
    # 仅传 timeout 数值给 ChatOpenAI 时，底层 HTTP 客户端可能忽略该设置
    try:
        import httpx
        _http_client = httpx.Client(timeout=httpx.Timeout(timeout_s, connect=30.0))
    except Exception:
        _http_client = None

    candidate_kwargs = [
        {**llm_kwargs, "model": resolved_model, "temperature": temperature, "timeout": timeout_s, "max_retries": max_retries},
        {**llm_kwargs, "model": resolved_model, "temperature": temperature, "request_timeout": timeout_s, "max_retries": max_retries},
        {**llm_kwargs, "model": resolved_model, "temperature": temperature, "timeout": timeout_s},
        {**llm_kwargs, "model": resolved_model, "temperature": temperature, "request_timeout": timeout_s},
        {**llm_kwargs, "model": resolved_model, "temperature": temperature},
    ]
    # 将 httpx.Client 注入到所有候选配置中
    if _http_client is not None:
        for _kw in candidate_kwargs:
            _kw["http_client"] = _http_client

    last_error: Exception | None = None
    for kwargs in candidate_kwargs:
        try:
            llm = ChatOpenAI(**kwargs)
            return llm, {
                "model": resolved_model,
                "base_url": resolved_base_url,
                "temperature": temperature,
            }
        except Exception as e:
            last_error = e

    raise last_error if last_error is not None else RuntimeError("Failed to initialize LLM client.")
