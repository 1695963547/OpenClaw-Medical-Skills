"""代码执行韧性层（Resilient Executor）—— 解决 P3/P4/P5

通用原则：
  沙箱（venv）+ 跨平台 + 网络 这三件事是基础设施问题，
  与具体 skill / 具体 API / 具体业务无关。
  本模块提供三组正交工具，按需在 `code_executor.py` 入口 hook：

  1. 平台适配    adapt_code_for_platform()    → 解决 P4（Windows bash 缺失）
  2. 路径注入    build_syspath_preamble()     → 解决 P3（沙箱路径断裂）
  3. 网络韧性    build_resilience_preamble()  → 解决 P5（API 超时无重试）

  全部基于标准库 + 第三方库（urllib3）实现，不绑定任何 skill。
  新增任何 skill 都自动获得这三层保护。
"""
from __future__ import annotations

import os
import re
import platform
import sys
from pathlib import Path
from typing import Optional


# ────────────────────────────────────────────────────────────────
# 平台适配 —— 解决 P4（Windows bash 缺失）
# ────────────────────────────────────────────────────────────────
# 场景：LLM 在 Python 代码中写 subprocess.run(["bash", "-c", "..."])
#       在 Windows 上没有 bash 命令 → WinError 2
# 解法：检测到 bash -c 时自动改写为 PowerShell 等价命令（或直接拒绝并提示）
_BASH_C_PATTERN = re.compile(
    r'\[\s*["\']bash["\']\s*,\s*["\'](?:-c|-l)["\']\s*,',
)


def adapt_code_for_platform(code: str, language: str) -> str:
    """通用平台适配。

    仅在 Windows + Python 时生效；其他平台 / 其他语言不改动。
    返回改写后的代码（若无需改写则原样返回）。
    """
    if platform.system() != "Windows":
        return code
    if language != "python":
        return code
    if not _BASH_C_PATTERN.search(code):
        return code

    # 仅做提示 + 改写：把 bash -c 换成 PowerShell -Command
    # 完整等价改写非常复杂（bash 语法 → powershell 语法），这里只解决"找不到 bash"这一关。
    # 真实业务逻辑建议 LLM 直接用 Python 替代。
    code = _BASH_C_PATTERN.sub('["powershell", "-Command",', code)

    # 在代码顶部注入一行警告，提醒 LLM 下次改用 Python
    if "[ResilientExecutor] Windows 平台" not in code:
        warn = (
            "# [ResilientExecutor] Windows 平台已自动将 'bash -c' 改写为 'powershell -Command'。\n"
            "# 注意：bash → PowerShell 语法并非完全等价（如 && 改 ;、cat 改 Get-Content）。\n"
            "# 建议尽量使用纯 Python 实现，避免跨平台 shell 差异。\n"
        )
        code = warn + code
    return code


# ────────────────────────────────────────────────────────────────
# 路径注入 —— 解决 P3（沙箱路径断裂）
# ────────────────────────────────────────────────────────────────
# 场景：venv 隔离后，sys.path 只包含 site-packages，
#       LLM 写的 `from scripts.query_opentargets import ...` 找不到模块
# 解法：扫描代码中的 import 模式，自动注入项目根目录 + 所有 skills/*/scripts 到 sys.path

# 匹配 "from X import Y" 或 "import X"，X 必须是非标准库（不含 .py/.json 等后缀）
_IMPORT_LINE_RE = re.compile(
    r'^\s*(?:from|import)\s+([A-Za-z_][\w.]*)',
    re.MULTILINE,
)
# 已知外部库（前缀匹配），import 它们时不需要注入 sys.path
_KNOWN_EXTERNAL_PREFIXES = (
    "requests", "numpy", "pandas", "scipy", "sklearn", "scikit",
    "torch", "tensorflow", "keras", "transformers",
    "scanpy", "anndata", "scrna", "scrnaseq",
    "pysam", "biopython", "Bio",
    "matplotlib", "seaborn", "plotly", "bokeh",
    "openai", "anthropic", "langchain", "langgraph",
    "tooluniverse", "tu",
    "pydeseq2", "gseapy", "gget", "cdsapi",
    "pymc", "pymoo", "cobrapy", "rdkit",
    "dask", "polars", "vaex", "zarr",
    "cryptography", "urllib3", "httpx", "aiohttp",
)
# 项目内"根命名空间"（这些名字必定来自项目目录，需注入 sys.path）
_PROJECT_NAMESPACES = (
    "scripts", "skills", "src", "tools", "tu", "tooluniverse",
)


def _is_external_lib(name: str) -> bool:
    """判断 import 名称是否为已知的外部库（前缀匹配）。"""
    base = name.split(".")[0]
    return any(base == p or base.startswith(p + "_") or base == p
               for p in _KNOWN_EXTERNAL_PREFIXES)


def _is_project_internal(name: str) -> bool:
    """判断 import 名称是否需要从项目目录查找。"""
    base = name.split(".")[0]
    if base in _PROJECT_NAMESPACES:
        return True
    # skills/<id>/scripts/xxx.py 中的导入也常见
    if "scripts." in name or name.startswith("scripts."):
        return True
    return False


def detect_needs_path_injection(code: str) -> bool:
    """检测代码是否需要 sys.path 注入。"""
    for m in _IMPORT_LINE_RE.finditer(code):
        name = m.group(1)
        if _is_project_internal(name) and not _is_external_lib(name):
            return True
        # tu / tooluniverse 这类"语义上可能来自项目"也注入
        if name.split(".")[0] in ("tu", "tooluniverse"):
            return True
    return False


def build_syspath_preamble(workspace: Optional[Path] = None) -> str:
    """构造一段 Python 注入代码，自动把项目目录加入 sys.path。

    设计：
      - 始终把 cwd 加入 sys.path（覆盖"从项目根目录运行"场景）
      - 探测项目根目录（含 skills/ 的目录），并把它加入 sys.path
      - 探测每个 skills/<id>/scripts/，加入 sys.path（允许 from scripts.xxx import）
      - 不破坏任何现有行为；如已存在同名路径，setdefault 不会重复添加
    """
    workspace = workspace or Path(os.getcwd()) / "workspace"
    # 候选根目录：workspace 自身、其父目录、祖父目录
    candidates: list[Path] = []
    for p in [workspace, workspace.parent, workspace.parent.parent, workspace.parent.parent.parent]:
        try:
            candidates.append(p.resolve())
        except Exception:
            pass
    # 去重
    seen = set()
    unique = []
    for p in candidates:
        s = str(p)
        if s not in seen:
            seen.add(s)
            unique.append(p)

    # 探测所有 skills/*/scripts 目录
    skill_scripts_dirs: list[Path] = []
    for cand in unique:
        skills_root = cand / "skills"
        if skills_root.is_dir():
            for sub in skills_root.iterdir():
                if not sub.is_dir():
                    continue
                scripts_dir = sub / "scripts"
                if scripts_dir.is_dir():
                    skill_scripts_dirs.append(scripts_dir.resolve())

    # 构造注入代码
    lines: list[str] = [
        "# [ResilientExecutor] 自动 sys.path 注入（解决沙箱路径断裂）",
        "import sys as _sys_re",
        "import os as _os_re",
    ]
    # 注入候选根目录
    for p in unique:
        lines.append(f"_sys_re.path.insert(0, {str(p)!r})")
    # 注入 skills/*/scripts 目录（让 LLM 可以 from scripts.xxx import yyy）
    for sd in skill_scripts_dirs:
        lines.append(f"_sys_re.path.insert(0, {str(sd)!r})")

    lines.extend([
        "# 显式清掉可能冲突的旧 scripts 模块（避免 venv site-packages 中的同名包遮蔽项目内脚本）",
        "for _k in [k for k in list(_sys_re.modules.keys()) if k == 'scripts' or k.startswith('scripts.')]:",
        "    del _sys_re.modules[_k]",
        "del _sys_re, _os_re",
        "",
    ])
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────
# 网络韧性 —— 解决 P5（API 超时无重试）
# ────────────────────────────────────────────────────────────────
# 场景：requests.get(url) 30s 超时后无重试 → 网络抖动时整次执行失败
# 解法：通过 monkey-patch 给 requests.Session 加上：
#       - 自动重试（3 次，指数退避）
#       - 按域名分级的 (connect, read) timeout
#       - 与现有 [API] 追踪日志兼容

# 按域名分级的超时（仅当用户没显式指定 timeout 时生效）
_DOMAIN_TIMEOUTS: list[tuple[str, tuple[float, float]]] = [
    # 已观察到 connect timeout=30 都不够的站点
    ("ebi.ac.uk",                (5.0, 120.0)),
    ("europepmc.org",            (5.0, 120.0)),
    # GraphQL API
    ("opentargets.org",          (10.0, 60.0)),
    ("api.platform.opentargets", (10.0, 60.0)),
    # 公共 API
    ("clinicaltrials.gov",       (10.0, 30.0)),
    ("ncbi.nlm.nih.gov",         (10.0, 60.0)),
    ("eutils.ncbi.nlm.nih.gov",  (10.0, 60.0)),
    ("rest.ensembl.org",         (10.0, 60.0)),
    # 兜底
    ("",                         (10.0, 60.0)),    # 默认
]


def split_timeout(url: str) -> tuple[float, float]:
    """根据 URL 域名返回 (connect_timeout, read_timeout)。"""
    url_lower = url.lower()
    for domain, t in _DOMAIN_TIMEOUTS:
        if not domain or domain in url_lower:
            return t
    return _DOMAIN_TIMEOUTS[-1][1]    # 兜底


def build_resilience_preamble() -> str:
    """构造一段 Python 注入代码，覆盖 requests 的 Session.request 方法，
    加上重试 + 智能 timeout + 保留原 [API] 追踪日志。

    兼容性：
      - 仅当用户没显式传 timeout 时，自动按域名填 (connect, read) timeout
      - 保留原有 [API] / [API-Body] 打印逻辑
      - 失败重试时不会重复打 [API]（避免日志噪声）

    实现细节：
      - 智能 timeout 通过包装 Session.request 实现（与原 _http_preamble 兼容）
      - 重试通过 *替换全局 HTTPAdapter 类* 实现（避免必须创建实例的陷阱），
        所有后续 Session() 自动获得重试配置
    """
    timeout_table_repr = repr([(d, t) for d, t in _DOMAIN_TIMEOUTS])
    return f'''
# [ResilientExecutor] requests 自动重试 + 智能 timeout（解决 API 超时无重试）
import sys as _sys_re2
try:
    import requests as _req_re
    from requests.adapters import HTTPAdapter as _HTTPAdapter_re
    try:
        from urllib3.util.retry import Retry as _Retry_re
    except ImportError:
        _Retry_re = None

    _RESILIENT_TIMEOUTS = {timeout_table_repr}

    def _resilient_get_timeout(_url):
        _u = _url.lower()
        for _d, _t in _RESILIENT_TIMEOUTS:
            if not _d or _d in _u:
                return _t
        return _RESILIENT_TIMEOUTS[-1][1]

    # ── 智能 timeout 注入：仅当用户没指定时，按域名自动填 ──
    if not hasattr(_req_re.Session, "_resilient_request_patched"):
        _orig_request_re = _req_re.Session.request

        def _resilient_request(self, method, url, **kw):
            if kw.get("timeout") is None:
                kw["timeout"] = _resilient_get_timeout(url)
            return _orig_request_re(self, method, url, **kw)

        _req_re.Session.request = _resilient_request
        _req_re.Session._resilient_request_patched = True

    # ── 重试配置：通过替换全局 HTTPAdapter 类实现 ──
    # 这样所有后续 requests.Session() 都自动获得 max_retries，
    # 避免必须创建实例才能调用 mount() 的陷阱。
    if (_Retry_re is not None
            and not hasattr(_req_re.adapters.HTTPAdapter, "_resilient_retry_patched")):
        _retry = _Retry_re(
            total=3,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET", "HEAD", "OPTIONS"]),
        )
        _OrigAdapter = _req_re.adapters.HTTPAdapter

        class _ResilientAdapter(_OrigAdapter):
            """全局替换的 HTTPAdapter：默认带 max_retries 配置。"""

            def __init__(self, *args, **kwargs):
                kwargs.setdefault("max_retries", _retry)
                kwargs.setdefault("pool_maxsize", 10)
                super().__init__(*args, **kwargs)

        _ResilientAdapter._resilient_retry_patched = True
        _req_re.adapters.HTTPAdapter = _ResilientAdapter
        # Session 内部引用了 HTTPAdapter 名称，也需要更新（避免 __init__ 重新创建非重试版本）
        try:
            _req_re.sessions.HTTPAdapter = _ResilientAdapter
            _req_re.Session.HTTPAdapter = _ResilientAdapter
        except (AttributeError, TypeError):
            pass
except ImportError:
    pass
finally:
    del _sys_re2
'''


# ────────────────────────────────────────────────────────────────
# 统一入口（推荐用法）
# ────────────────────────────────────────────────────────────────
def prepare_code_for_execution(
    code: str,
    language: str,
    workspace: Optional[Path] = None,
    *,
    inject_path: bool = True,
    inject_resilience: bool = True,
) -> str:
    """统一入口：一次性完成 平台适配 + 路径注入 + 网络韧性。

    仅对 Python 注入 sys.path 和 resilience preamble（这两个是 Python 专属）；
    平台适配对所有 language 都生效。
    """
    # 1. 平台适配（所有语言都做，但仅 Windows 实际改写）
    code = adapt_code_for_platform(code, language)

    # 2-3. Python 专属的 preamble 注入
    if language == "python":
        parts: list[str] = []
        if inject_path and detect_needs_path_injection(code):
            parts.append(build_syspath_preamble(workspace))
        if inject_resilience:
            parts.append(build_resilience_preamble())
        if parts:
            code = "\n".join(parts) + "\n" + code
    return code


__all__ = [
    "adapt_code_for_platform",
    "detect_needs_path_injection",
    "build_syspath_preamble",
    "build_resilience_preamble",
    "split_timeout",
    "prepare_code_for_execution",
]
