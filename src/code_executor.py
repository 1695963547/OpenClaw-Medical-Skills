"""安全执行 Python / Bash / R / JavaScript 代码

增强安全措施：
- Windows bash 命令自动适配（Git Bash / WSL / PowerShell）
- 文件路径白名单（仅允许 workspace 目录读写）
- subprocess 调用工具白名单（仅允许生物信息学相关 CLI 工具）
- AST 静态分析检测文件路径越权和命令越权
- API 调用结果细粒度追踪
- pip install 结果预追踪
- Node.js 脚本支持
- 虚拟环境隔离（venv + conda 双层架构）
"""
import os
import re
import ast
import platform
import subprocess
import sys
import time
import venv
import shutil
import logging
from pathlib import Path
from shutil import which
from typing import Optional

from src.conda_manager import CondaEnvManager

_logger = logging.getLogger("agent.code_executor")


# ─── 禁止操作（原有黑名单） ───

FORBIDDEN = [
    r"\bexec\b\s*\(", r"\beval\b\s*\(",
    r"\b__import__\b\s*\(", r"\bimportlib\b",
    r"\bshutil\.rmtree\b", r"\bos\.remove\b", r"\bos\.unlink\b",
    r"rm\s+-rf\b", r"del\s+/[fq]",
]


# ─── Windows bash 命令自动适配 ───

def _resolve_bash_cmd():
    """在 Windows 上自动查找可用的 bash 命令"""
    if platform.system() == "Windows":
        for candidate in ["bash", r"C:\Program Files\Git\bin\bash.exe",
                          r"C:\Program Files (x86)\Git\bin\bash.exe", "wsl"]:
            if which(candidate):
                return [candidate]
        return ["powershell", "-Command"]
    return ["bash"]


def _resolve_node_cmd():
    """查找可用的 Node.js 命令"""
    for candidate in ["node", "nodejs"]:
        if which(candidate):
            return [candidate]
    return None


LANG_CONFIG = {
    "python": {"cmd": ["python"], "suffix": ".py"},
    "bash":   {"cmd": _resolve_bash_cmd(), "suffix": ".sh"},
    "r":      {"cmd": ["Rscript"], "suffix": ".R"},
    "javascript": {"cmd": _resolve_node_cmd() or ["node"], "suffix": ".mjs"},
}


# ─── Conda 优先包分类（需要 C 编译 / bioconda 生态的包）───

# C 扩展 Python 包（pip 安装需要 C 编译器，conda 有预编译二进制）
CONDA_PREFERRED_PACKAGES = {
    "pysam", "pybigwig", "pybedtools", "tables",
    "pysamstats", "cyvcf2", "htslib",
}

# 这些包对应的纯 Python 替代方案（conda 不可用时的降级提示）
CONDA_FALLBACK_MESSAGES = {
    "pysam": ("需要编译 htslib C 扩展，pip 安装极易挂起。"
              "替代方案: 使用 Biopython 的 SeqIO + 手动构建 .fai 索引（纯 Python 实现）。"),
    "tables": ("需要编译 HDF5 C 扩展，pip 安装可能失败或挂起。"
               "替代方案: 使用 h5py 或 pandas 的 HDFStore。"),
    "pybigwig": ("需要编译 libBigWig C 库。"
                 "替代方案: 确认是否需要此库；若必须使用，请下载预编译 wheel。"),
    "pybedtools": ("需要编译 BEDTools C++ 源码。替代方案: 用纯 Python bed_utils 或 pybedlite。"),
    "pysamstats": ("需要编译 htslib C 扩展。替代方案: 用 pysam 读取 BAM + numpy 手动统计。"),
    "cyvcf2": ("需要编译 htslib C 扩展。替代方案: 使用 PyVCF 或 scikit-allel。"),
    "htslib": ("需要编译 htslib C 库。替代方案: 通过 pysam 间接使用 htslib。"),
}


def _classify_packages(packages: list[str]) -> tuple[list[str], list[str]]:
    """将包列表分为 pip 可安装（纯 Python）和 conda 优先（C 扩展）两类

    Args:
        packages: 从代码 # pip install 行提取的包名列表

    Returns:
        (pip_pkgs, conda_pkgs) — conda_pkgs 为空时走 venv 路径，
        conda_pkgs 非空且有 conda 时走 conda 路径
    """
    pip_pkgs, conda_pkgs = [], []
    for pkg in packages:
        normalized = pkg.lower().replace("-", "_")
        if normalized in CONDA_PREFERRED_PACKAGES:
            conda_pkgs.append(pkg)
        else:
            pip_pkgs.append(pkg)
    return pip_pkgs, conda_pkgs


# ─── R 语言包提取 & conda 包名映射 ───

def _extract_r_packages(code: str) -> list[str]:
    """从 R 代码中提取 library() 和 require() 调用的包名

    过滤掉 R 内置包（base/ stats/ utils/ methods/ graphics/ grDevices/ datasets）
    """
    _R_BUILTINS = {'base', 'stats', 'utils', 'methods', 'graphics', 'grdevices', 'datasets'}
    packages = []
    for m in re.finditer(r'(?:library|require)\s*\(\s*["\']?(\w+(?:\.\w+)*)["\']?\s*\)', code):
        pkg = m.group(1)
        if pkg.lower() not in _R_BUILTINS and pkg not in packages:
            packages.append(pkg)
    return packages


# R 包名 → conda 包名映射（Bioconductor → bioconductor-*, CRAN → r-*）
R_PACKAGE_TO_CONDA: dict[str, str] = {
    # ── Bioconductor packages ──
    "DESeq2":                 "bioconductor-deseq2",
    "edgeR":                  "bioconductor-edger",
    "limma":                  "bioconductor-limma",
    "apeglm":                 "bioconductor-apeglm",
    "IHW":                    "bioconductor-ihw",
    "tximport":               "bioconductor-tximport",
    "BiocParallel":           "bioconductor-biocparallel",
    "GenomicRanges":          "bioconductor-genomicranges",
    "SummarizedExperiment":   "bioconductor-summarizedexperiment",
    "MatrixGenerics":         "bioconductor-matrixgenerics",
    "EnhancedVolcano":        "bioconductor-enhancedvolcano",
    "ashr":                   "bioconductor-ashr",
    "vsn":                    "bioconductor-vsn",
    "geneplotter":            "bioconductor-geneplotter",
    "fdrtool":                "bioconductor-fdrtool",
    "biomaRt":                "bioconductor-biomart",
    "org.Hs.eg.db":           "bioconductor-org.hs.eg.db",
    "org.Mm.eg.db":           "bioconductor-org.mm.eg.db",
    "AnnotationDbi":          "bioconductor-annotationdbi",
    "clusterProfiler":        "bioconductor-clusterprofiler",
    "enrichplot":             "bioconductor-enrichplot",
    "DOSE":                   "bioconductor-dose",
    "pathview":               "bioconductor-pathview",
    "GSVA":                   "bioconductor-gsva",
    "GSEABase":               "bioconductor-gseabase",
    "sva":                    "bioconductor-sva",
    "RUVSeq":                 "bioconductor-ruvseq",
    "PCAtools":               "bioconductor-pcatools",
    "ComplexHeatmap":         "bioconductor-complexheatmap",
    "SingleCellExperiment":   "bioconductor-singlecellexperiment",
    "scater":                 "bioconductor-scater",
    "scran":                  "bioconductor-scran",
    "DropletUtils":           "bioconductor-dropletutils",
    # ── CRAN packages ──
    "ggplot2":                "r-ggplot2",
    "dplyr":                  "r-dplyr",
    "tidyr":                  "r-tidyr",
    "readr":                  "r-readr",
    "pheatmap":               "r-pheatmap",
    "RColorBrewer":           "r-rcolorbrewer",
    "ggrepel":                "r-ggrepel",
    "hexbin":                 "r-hexbin",
    "cowplot":                "r-cowplot",
    "ggpubr":                 "r-ggpubr",
    "forcats":                "r-forcats",
    "stringr":                "r-stringr",
    "purrr":                  "r-purrr",
    "tibble":                 "r-tibble",
    "magrittr":               "r-magrittr",
    "circlize":               "r-circlize",
    "knitr":                  "r-knitr",
    "rmarkdown":              "r-rmarkdown",
    "tidyverse":              "r-tidyverse",
    "data.table":             "r-data.table",
    "reshape2":               "r-reshape2",
    "scales":                 "r-scales",
    "Matrix":                 "r-matrix",
    "MASS":                   "r-mass",
    "survival":               "r-survival",
    "glmnet":                 "r-glmnet",
    "lme4":                   "r-lme4",
    "caret":                  "r-caret",
    "randomForest":           "r-randomforest",
    "e1071":                  "r-e1071",
    "igraph":                 "r-igraph",
}


def _map_r_to_conda(pkg_name: str) -> Optional[str]:
    """将 R 包名映射为 conda 包名（小写不区分大小写尝试）"""
    # 精确匹配
    if pkg_name in R_PACKAGE_TO_CONDA:
        return R_PACKAGE_TO_CONDA[pkg_name]
    # 大小写不敏感回退
    for r_name, conda_name in R_PACKAGE_TO_CONDA.items():
        if r_name.lower() == pkg_name.lower():
            return conda_name
    return None


# ─── 允许的 CLI 工具白名单 ───

ALLOWED_CLI_TOOLS = {
    # 序列比对
    "blastn", "blastp", "blastx", "tblastn", "tblastx",
    "makeblastdb", "blastdbcmd", "blastdb_aliastool",
    # 比对工具
    "bwa", "bowtie2", "hisat2", "star", "minimap2",
    "samtools", "bcftools", "bedtools", "vcftools",
    # 变异调用
    "gatk", "freebayes", "varscan",
    # 质控
    "fastqc", "multiqc", "trimmomatic", "cutadapt", "fastp",
    # 峰调用
    "macs2", "macs3",
    # 分类
    "kraken2", "kraken", "bracken", "metaphlan",
    # 基因预测
    "prodigal", "augustus", "glimmerhmm",
    # 多序列比对 & 进化树
    "muscle", "mafft", "clustalo", "fasttree", "iqtree", "raxml", "raxmlng",
    # 格式转换
    "seqtk", "emboss", "bedops",
    # 通用工具（受限）
    "python", "python3", "pip", "Rscript", "perl",
    # 网络工具（仅用于下载）
    "curl", "wget",
    # 跨平台 shell（[ResilientExecutor] 平台适配后可能使用）
    "bash", "wsl", "powershell", "pwsh", "cmd", "sh", "zsh",
}


# ─── API 错误检测模式 ───

API_ERROR_PATTERNS = [
    # HTTP 状态码
    r"\b401\b", r"\b403\b", r"\b404\b", r"\b429\b", r"\b500\b", r"\b502\b", r"\b503\b",
    # 认证错误
    r"API\s*[Kk]ey", r"[Aa]uthentication\s+failed", r"[Uu]nauthorized",
    r"[Aa]ccess\s+[Dd]enied", r"[Ii]nvalid\s+.*(api|key|token|credential)",
    # 限流
    r"[Rr]ate\s*[Ll]imit", r"[Tt]oo\s+[Mm]any\s+[Rr]equests",
    r"[Qq]uota\s+[Ee]xceeded",
    # 网络连接
    r"ConnectionError", r"ConnectionRefusedError", r"TimeoutError",
    r"urlopen\s+error", r"[Nn]ame.*[Rr]esolution", r"DNS",
    r"Max\s+retries\s+exceeded",
    # NCBI/Biopython 特有（注意：不能用 r"ncbi"/r"entrez" 这种过于宽泛的模式，
    # 因为合法的 PubMed URL 如 pubmed.ncbi.nlm.nih.gov 也包含 "ncbi"，会导致误报）
    r"HTTP\s+Error\s+\d+", r"NCBI\s+Error", r"Entrez\s+Error",
    r"E-Utility\s+error", r"UID\s+is\s+invalid",
    # 通用 API 错误
    r"API\s+(call|request)\s+failed", r"[Ss]ervice\s+[Uu]navailable",
    r"[Ee]xternal\s+[Aa]PI",
]

# 编译正则
_API_ERROR_COMPILED = [re.compile(p, re.IGNORECASE) for p in API_ERROR_PATTERNS]


def _detect_api_errors(stdout: str, stderr: str) -> Optional[str]:
    """检测执行输出中的 API 调用错误，返回具体错误类型或 None

    对于 HTTP 状态码（401/403/404/429/500/502/503）使用正则边界匹配（\\b），
    避免染色体位置号（如 160540105 中的 "401"）等数字串误触。
    对于文本类错误（unauthorized 等）保持简单子串匹配。
    """
    combined = f"{stdout}\n{stderr}"
    combined_lc = combined.lower()

    # ── HTTP 状态码：使用正则边界匹配，避免数字串中误触 ──
    # 匹配模式：HTTP 401, status=401, status code 401, 返回 401 等上下文
    def _match_http_code(code: str) -> bool:
        """检测 HTTP 状态码是否出现在合理的 API 错误上下文中"""
        patterns = [
            rf"\bHTTP\s*{code}\b",
            rf"\bstatus[=:\s]+{code}\b",
            rf"\b{code}\s*Unauthorized\b" if code == "401" else rf"\b{code}\b",
            rf"\b{code}\s*Forbidden\b" if code == "403" else "",
            rf"\b{code}\s*Not\s+Found\b" if code == "404" else "",
            rf"\b{code}\s*Too\s+Many\s+Requests\b" if code == "429" else "",
            rf"\b{code}\s*Internal\s+Server" if code == "500" else "",
        ]
        for p in patterns:
            if p and re.search(p, combined, re.IGNORECASE):
                return True
        return False

    # 认证错误检测
    if _match_http_code("401") or _match_http_code("403"):
        code = "401" if _match_http_code("401") else "403"
        return f"API认证失败: 检测到 'HTTP {code}'，请检查API Key配置"

    # 文本类认证错误（不需要边界匹配）
    text_auth_errors = ["unauthorized", "authentication failed",
                        "access denied", "invalid api", "invalid key",
                        "invalid token", "api key", "credential"]
    for err in text_auth_errors:
        if err in combined_lc:
            return f"API认证失败: 检测到 '{err}'，请检查API Key配置"

    # 限流检测
    if _match_http_code("429"):
        return f"API限流: 检测到 'HTTP 429'，请求过于频繁"
    text_rate_errors = ["rate limit", "too many requests", "quota exceeded"]
    for err in text_rate_errors:
        if err in combined_lc:
            return f"API限流: 检测到 '{err}'，请求过于频繁"

    # 网络错误
    text_network_errors = ["connectionerror", "connectionrefused", "timeouterror",
                           "urlopen error", "name resolution", "dns",
                           "max retries exceeded", "connection refused", "10060"]
    for err in text_network_errors:
        if err in combined_lc:
            return f"API网络错误: 检测到 '{err}'，无法连接远程服务"

    # 资源不存在
    if _match_http_code("404"):
        return f"API资源不存在: 检测到 'HTTP 404'"
    text_not_found = ["not found", "no results", "no data"]
    for err in text_not_found:
        if err in combined_lc:
            return f"API资源不存在: 检测到 '{err}'"

    # 服务端错误
    for code in ["500", "502", "503"]:
        if _match_http_code(code):
            return f"API服务端错误: 检测到 'HTTP {code}'"
    text_server_errors = ["internal server error", "bad gateway", "service unavailable"]
    for err in text_server_errors:
        if err in combined_lc:
            return f"API服务端错误: 检测到 '{err}'"

    # 用编译好的正则做更广泛的匹配
    for pattern in _API_ERROR_COMPILED:
        match = pattern.search(combined)
        if match:
            return f"API调用异常: 匹配到 '{match.group()}'"

    return None


# ─── AST 静态分析 ───

def validate_file_access(code: str, workspace: Path) -> tuple[bool, str]:
    """检查 Python 代码中的文件路径是否在 workspace 白名单内

    仅检测字面量路径（如 open("C:\\xxx")），动态路径无法静态分析。
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True, ""  # 语法错误交给执行时再报

    workspace_str = str(workspace).lower()

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr

            if func_name in ("open",) and node.args:
                if isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    file_path = node.args[0].value
                    if _is_path_outside_workspace(file_path, workspace_str):
                        return False, f"文件路径越权: {file_path}（仅允许在 workspace 目录内操作）"

    return True, ""


def validate_subprocess_calls(code: str) -> tuple[bool, str]:
    """检查 subprocess 调用是否只使用白名单工具"""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True, ""

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = ""
            if isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            elif isinstance(node.func, ast.Name):
                func_name = node.func.id

            if func_name in ("run", "Popen", "call", "check_output", "check_call"):
                if node.args:
                    first_arg = node.args[0]
                    if isinstance(first_arg, (ast.List, ast.Tuple)):
                        if first_arg.elts and isinstance(first_arg.elts[0], ast.Constant):
                            cmd = str(first_arg.elts[0].value)
                            if cmd not in ALLOWED_CLI_TOOLS:
                                return False, f"禁止执行的命令: {cmd}（不在白名单中）"
                    elif isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                        cmd = first_arg.value.split()[0] if first_arg.value else ""
                        if cmd and cmd not in ALLOWED_CLI_TOOLS:
                            return False, f"禁止执行的命令: {cmd}（不在白名单中）"

    return True, ""


def validate_dangerous_calls(code: str) -> tuple[bool, str]:
    """检查危险的 os/shutil 系统调用
    
    拦截 os.system(), os.popen(), shutil.rmtree() 等可能导致系统损害的调用。
    """
    DANGEROUS_PATTERNS = {
        # os 模块危险函数
        "system": "os.system() 可执行任意 shell 命令，禁止使用",
        "popen": "os.popen() 可执行任意 shell 命令，禁止使用",
        "exec": "os.exec*() 系列函数可替换当前进程，禁止使用",
        "execl": "os.exec*() 系列函数可替换当前进程，禁止使用",
        "execle": "os.exec*() 系列函数可替换当前进程，禁止使用",
        "execlp": "os.exec*() 系列函数可替换当前进程，禁止使用",
        "execlpe": "os.exec*() 系列函数可替换当前进程，禁止使用",
        "execv": "os.exec*() 系列函数可替换当前进程，禁止使用",
        "execve": "os.exec*() 系列函数可替换当前进程，禁止使用",
        "execvp": "os.exec*() 系列函数可替换当前进程，禁止使用",
        "execvpe": "os.exec*() 系列函数可替换当前进程，禁止使用",
        "spawn": "os.spawn*() 系列函数可创建新进程，禁止使用",
        "spawnl": "os.spawn*() 系列函数可创建新进程，禁止使用",
        "spawnle": "os.spawn*() 系列函数可创建新进程，禁止使用",
        "spawnlp": "os.spawn*() 系列函数可创建新进程，禁止使用",
        "spawnlpe": "os.spawn*() 系列函数可创建新进程，禁止使用",
        "spawnv": "os.spawn*() 系列函数可创建新进程，禁止使用",
        "spawnve": "os.spawn*() 系列函数可创建新进程，禁止使用",
        "spawnvp": "os.spawn*() 系列函数可创建新进程，禁止使用",
        "spawnvpe": "os.spawn*() 系列函数可创建新进程，禁止使用",
        # shutil 危险函数
        "rmtree": "shutil.rmtree() 可递归删除目录，禁止使用",
    }
    
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True, ""
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = ""
            if isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            elif isinstance(node.func, ast.Name):
                func_name = node.func.id
            
            if func_name in DANGEROUS_PATTERNS:
                return False, DANGEROUS_PATTERNS[func_name]
    
    return True, ""


def _is_path_outside_workspace(path_str: str, workspace_str: str) -> bool:
    """判断路径是否在 workspace 之外"""
    p = Path(path_str)
    if not p.is_absolute():
        return False
    return not str(p).lower().startswith(workspace_str)


# ─── pip install 追踪 ───

def _extract_pip_installs(code: str) -> list[str]:
    """从代码中提取所有 pip install 的包名

    支持格式：
    - # pip install pkg1 pkg2 pkg3   （注释行，多包空格分隔）
    - # pip install --quiet pkg1     （带 --quiet/-q 标志）
    - subprocess pip install pkg1    （传统写法）
    """
    packages = []
    # 先匹配 pip install 行，再提取行内所有包名
    for line in code.split('\n'):
        # 找到包含 pip install 的行
        if 'pip' not in line or 'install' not in line:
            continue
        install_match = re.search(r'pip\s+install\s+', line)
        if not install_match:
            continue
        # 取 install 之后的部分
        rest = line[install_match.end():]
        # 跳过标志位（--quiet, -q, --no-deps 等）
        tokens = rest.split()
        for token in tokens:
            # 跳过以 - 开头的标志位
            if token.startswith('-'):
                continue
            # 跳过看起来像非包名的 token（路径、URL 等）
            if '/' in token or '\\' in token or token.startswith('http'):
                continue
            # 验证看起来像包名（字母/数字/下划线/连字符/点号）
            if re.match(r'^[a-zA-Z0-9_\-\.]+$', token):
                if token not in packages:
                    packages.append(token)
    return packages


def _pre_install_packages(packages: list[str], python_path: str, timeout: int = 120) -> dict:
    """预先安装 Python 包，返回安装结果。

    增强：超时后使用进程树杀灭（Windows taskkill /F /T），
    避免 C 编译挂起的子进程残留。
    """
    if not packages:
        return {"success": True, "installed": [], "failed": [], "output": ""}

    results = {"success": True, "installed": [], "failed": [], "output": ""}
    for pkg in packages:
        try:
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            proc = subprocess.Popen(
                [python_path, "-m", "pip", "install", "--quiet", pkg],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
                creationflags=creationflags,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Windows: 杀整个进程树（/F 强制 /T 子进程）
                if os.name == 'nt':
                    subprocess.run(
                        ['taskkill', '/F', '/T', '/PID', str(proc.pid)],
                        capture_output=True,
                    )
                else:
                    proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                results["failed"].append({
                    "package": pkg,
                    "error": f"安装超时 ({timeout}s)，进程已强制终止",
                })
                results["success"] = False
                continue

            if proc.returncode == 0:
                results["installed"].append(pkg)
            else:
                results["failed"].append({
                    "package": pkg,
                    "error": stderr.strip() or stdout.strip(),
                })
                results["success"] = False
        except Exception as e:
            results["failed"].append({
                "package": pkg,
                "error": str(e),
            })
            results["success"] = False

    return results


# ─── npm 包预安装（JavaScript 依赖管理）───

def _extract_npm_installs(code: str) -> list[str]:
    """从 JavaScript 代码中提取所有外部 npm 包名"""
    packages = []
    # 匹配 require('xxx')
    for m in re.finditer(r"require\(['\"]([^'\"]+)['\"]\)", code):
        pkg = m.group(1)
        if not pkg.startswith(('.', '/', 'node:')):
            packages.append(pkg)
    # 匹配 import ... from 'xxx'
    for m in re.finditer(r"""from\s+['\"]([^'\"]+)['\"]""", code):
        pkg = m.group(1)
        if not pkg.startswith(('.', '/', 'node:')):
            packages.append(pkg)
    return list(set(packages))


def _pre_install_npm_packages(packages: list[str], cwd: str, timeout: int = 120) -> dict:
    """预安装 npm 包到 workspace/node_modules
    使用 --prefix 强制安装到 cwd，避免 npm 上溯到父级 package.json
    """
    if not packages:
        return {"success": True, "installed": [], "failed": []}

    results = {"success": True, "installed": [], "failed": []}
    npm_exe = str(which("npm") or "npm")
    # 创建 cwd 确保 --prefix 目标目录存在
    os.makedirs(cwd, exist_ok=True)
    cmd = [npm_exe, "install", "--prefix", cwd, "--no-save", "--no-audit", "--no-fund", "--package-lock=false"] + packages

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
        if result.returncode == 0:
            results["installed"] = packages
        else:
            # npm install 可能部分成功，仍记录为已安装
            results["installed"] = packages
            _logger.warning("npm install 可能有部分失败: %s", result.stderr[:200])
    except subprocess.TimeoutExpired:
        results["failed"] = [{"package": p, "error": f"安装超时 ({timeout}s)"} for p in packages]
        results["success"] = False
    except Exception as e:
        results["failed"] = [{"package": p, "error": str(e)} for p in packages]
        results["success"] = False

    return results


# ─── CodeExecutor ───

class CodeExecutor:
    def __init__(self, timeout: int = 300, workspace: str = "./workspace",
                 use_venv: bool = False):
        self.timeout = timeout
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.use_venv = use_venv
        # 启动时清理上次运行残留的孤儿 venv（正常退出已清理，这里只兜底 crash 场景）
        for orphan in self.workspace.glob(".venv_*"):
            if orphan.is_dir():
                shutil.rmtree(str(orphan), ignore_errors=True)
                _logger.info("清理孤儿venv: %s", orphan.name)
        self._venv_cache: dict[str, Path] = {}  # session_id -> venv_path
        # conda 管理器：单例，C 扩展包走 conda per-session 隔离
        self._conda_manager = CondaEnvManager()
        self._conda_session_cache: dict[str, str] = {}  # session_id -> conda_env_name
        self._conda_r_session_cache: dict[str, str] = {}  # session_id -> r_conda_env_name

    def scan(self, code: str) -> tuple[bool, str]:
        """原有黑名单扫描"""
        for pattern in FORBIDDEN:
            if re.search(pattern, code):
                return False, f"禁止操作: {pattern}"
        return True, ""

    def validate(self, code: str, language: str) -> tuple[bool, str]:
        """增强安全验证：黑名单 + 文件路径白名单 + subprocess 白名单"""
        safe, reason = self.scan(code)
        if not safe:
            return False, reason

        if language == "python":
            safe, reason = validate_file_access(code, self.workspace)
            if not safe:
                return False, reason

            safe, reason = validate_subprocess_calls(code)
            if not safe:
                return False, reason

            safe, reason = validate_dangerous_calls(code)
            if not safe:
                return False, reason

        return True, ""

    def _get_installed_packages(self, python_path: str) -> set[str]:
        """获取 venv 中已安装的包名（用于增量安装）"""
        try:
            result = subprocess.run(
                [python_path, "-m", "pip", "list", "--format=freeze"],
                capture_output=True, text=True, timeout=30,
                encoding="utf-8", errors="replace",
            )
            installed = set()
            for line in result.stdout.strip().split("\n"):
                if "==" in line:
                    name = line.split("==")[0].lower().replace("-", "_")
                    installed.add(name)
            return installed
        except Exception:
            return set()

    def _get_venv_python(self, session_id: str, packages: list[str]) -> tuple[Optional[str], str]:
        """为 session 创建虚拟环境并安装依赖，返回 (python路径, 状态信息)

        Codex 策略：venv 按 session_id 隔离，同一会话内所有步骤共享，
        支持增量安装（已安装的包不重复安装），会话结束即销毁。
        """
        if not self.use_venv or not session_id:
            return None, ""

        venv_path = self._venv_cache.get(session_id)
        if venv_path and venv_path.exists():
            python_path = str(venv_path / "Scripts" / "python.exe" if platform.system() == "Windows"
                             else venv_path / "bin" / "python")
            if Path(python_path).exists():
                # venv 已存在，只需安装增量依赖
                if packages:
                    installed = self._get_installed_packages(python_path)
                    missing = [p for p in packages
                               if p.lower().replace("-", "_") not in installed]
                    if missing:
                        install_result = _pre_install_packages(missing, python_path, timeout=180)
                        if install_result["failed"]:
                            failed_pkgs = [f["package"] for f in install_result["failed"]]
                            return python_path, f"[venv增量安装部分失败: {failed_pkgs}]"
                        return python_path, f"[venv增量安装: {missing}]"
                return python_path, ""

        # 首次创建 venv
        venv_dir = self.workspace / f".venv_{session_id}"
        try:
            venv.create(str(venv_dir), with_pip=True, clear=True)
            python_path = str(venv_dir / "Scripts" / "python.exe" if platform.system() == "Windows"
                              else venv_dir / "bin" / "python")

            if packages:
                install_result = _pre_install_packages(packages, python_path, timeout=180)
                if install_result["failed"]:
                    failed_pkgs = [f["package"] for f in install_result["failed"]]
                    return python_path, f"[venv安装部分依赖失败: {failed_pkgs}]"

            self._venv_cache[session_id] = venv_dir
            return python_path, f"[venv已创建: session={session_id[:8]}]"
        except Exception as e:
            return None, f"[venv创建失败: {e}]"

    def _get_conda_env(self, session_id: str, conda_pkgs: list[str],
                       pip_pkgs: list[str]) -> dict:
        """为 session 创建/复用 conda 隔离环境并安装包

        模板克隆优化：首次创建 _openclaw_bio_template，后续 session 从模板
        克隆（硬链接，~10秒），再 conda install C 扩展包 + pip install 纯 Python 包。

        Returns:
            {"python_cmd": ["conda", "run", "-n", "env", "python"], "status": "..."}
            或 {"python_cmd": None, "status": "失败原因"}
        """
        if not self._conda_manager.is_available():
            return {"python_cmd": None, "status": "[conda 不可用]"}

        # 检查该 session 是否已有 conda env（复用）
        if session_id in self._conda_session_cache:
            env_name = self._conda_session_cache[session_id]
            # 增量安装新的纯 Python 包
            if pip_pkgs:
                self._conda_manager.install_pip_packages(env_name, pip_pkgs)
            return {
                "python_cmd": ["conda", "run", "-n", env_name, "python"],
                "status": f"[conda 环境复用: {env_name}]",
            }

        # 创建新的 session conda env
        env_name, status = self._conda_manager.create_session_env(session_id, conda_pkgs)
        if not env_name:
            return {"python_cmd": None, "status": status}

        self._conda_session_cache[session_id] = env_name

        # 在 conda env 中用 pip 安装纯 Python 包
        if pip_pkgs:
            pip_ok, pip_status = self._conda_manager.install_pip_packages(env_name, pip_pkgs)
            if not pip_ok:
                status += f" {pip_status}"

        return {
            "python_cmd": ["conda", "run", "-n", env_name, "python"],
            "status": status,
        }

    def _get_r_conda_env(self, session_id: str, conda_r_pkgs: list[str]) -> dict:
        """为 session 创建/复用 R conda 隔离环境并安装 R 包

        R 模板克隆优化：首次创建 _openclaw_r_template（含 r-base=4.3），
        后续 session 从模板克隆，再 conda install 所需的 Bioconductor/CRAN 包。

        Returns:
            {"rscript_cmd": ["conda", "run", "-n", "env", "Rscript"], "status": "..."}
            或 {"rscript_cmd": None, "status": "失败原因"}
        """
        if not self._conda_manager.is_available():
            return {"rscript_cmd": None, "status": "[conda 不可用]"}

        # 检查该 session 是否已有 R conda env（复用）
        if session_id in self._conda_r_session_cache:
            env_name = self._conda_r_session_cache[session_id]
            return {
                "rscript_cmd": self._conda_manager.get_rscript_cmd(session_id),
                "status": f"[R conda 环境复用: {env_name}]",
            }

        # 创建新的 session R conda env
        env_name, status = self._conda_manager.create_r_session_env(session_id, conda_r_pkgs)
        if not env_name:
            return {"rscript_cmd": None, "status": status}

        self._conda_r_session_cache[session_id] = env_name
        return {
            "rscript_cmd": self._conda_manager.get_rscript_cmd(session_id),
            "status": status,
        }

    def install_skill_dependencies(self, session_id: str, skill_id: str) -> dict:
        """读取技能目录下的 requirements.txt 并批量安装到隔离 venv

        第1层防御（主动安装）：在 execute_code 执行代码前调用，
        一次性安装技能所需的所有依赖，避免逐个安装的低效循环。

        Args:
            session_id: 会话 ID
            skill_id: 技能 ID（目录名），如 "scrna-qc"

        Returns:
            {
                "installed": [pkg1, pkg2, ...],   # 成功安装的包
                "failed": [{"package": pkg, "error": msg}, ...],  # 安装失败的包
                "skipped": [pkg1, ...],            # 已安装跳过的包
            }
        """
        skill_dir = self.workspace.parent / "skills" / skill_id
        req_file = skill_dir / "requirements.txt"

        # ── tooluniverse 系列技能：自动补充 tooluniverse 依赖（无需修改 SKILL.md）──
        extra_packages = []
        if skill_id.startswith("tooluniverse-"):
            extra_packages = ["tooluniverse"]

        if not req_file.exists() and not extra_packages:
            return {"installed": [], "failed": [], "skipped": []}

        # 解析 requirements.txt 中的包名
        packages = list(extra_packages)
        if req_file.exists():
            try:
                content = req_file.read_text(encoding="utf-8")
                for line in content.split("\n"):
                    line = line.strip()
                    # 跳过空行和注释
                    if not line or line.startswith("#"):
                        continue
                    # 提取包名（去掉版本约束符：==, >=, <=, ~=, !=, >, <, ; 等）
                    # 例如: "scanpy>=1.9.0" → "scanpy"
                    pkg_name = re.split(r"[=<>!~;\[\s]", line)[0].strip()
                    if pkg_name:
                        packages.append(pkg_name)
            except Exception:
                pass

        if not packages:
            return {"installed": [], "failed": [], "skipped": []}

        # 获取 venv python 路径
        venv_path = self._venv_cache.get(session_id)
        if not venv_path or not venv_path.exists():
            # venv 不存在 → 先创建
            python_path, _ = self._get_venv_python(session_id, [])
            if not python_path:
                return {"installed": [], "failed": [
                    {"package": ", ".join(packages), "error": "venv 创建失败"}
                ], "skipped": []}
            venv_path = self._venv_cache.get(session_id)

        if not venv_path or not venv_path.exists():
            return {"installed": [], "failed": [
                {"package": ", ".join(packages), "error": "venv 创建失败"}
            ], "skipped": []}

        python_path = str(
            venv_path / "Scripts" / "python.exe"
            if platform.system() == "Windows"
            else venv_path / "bin" / "python"
        )

        # 检查已安装的包，做增量安装
        installed_set = self._get_installed_packages(python_path)
        missing = []
        skipped = []
        for pkg in packages:
            normalized = pkg.lower().replace("-", "_")
            if normalized in installed_set:
                skipped.append(pkg)
            else:
                missing.append(pkg)

        if not missing:
            _logger.info("技能 %s 的所有依赖已安装，跳过: %s", skill_id, packages)
            return {"installed": [], "failed": [], "skipped": skipped}

        # 批量安装缺失的包
        _logger.info("批量安装技能 %s 的依赖: %s", skill_id, missing)
        install_result = _pre_install_packages(missing, python_path, timeout=300)

        result = {
            "installed": install_result["installed"],
            "failed": install_result["failed"],
            "skipped": skipped,
        }
        if install_result["failed"]:
            failed_names = [f["package"] for f in install_result["failed"]]
            _logger.warning("技能 %s 部分依赖安装失败: %s", skill_id, failed_names)
        if install_result["installed"]:
            _logger.info("技能 %s 依赖安装成功: %s", skill_id, install_result["installed"])
        return result

    def install_batch_packages(self, session_id: str, packages: list[str]) -> dict:
        """批量安装一组包到隔离 venv（兜底方案）

        Args:
            session_id: 会话 ID
            packages: 要安装的包名列表

        Returns:
            {"installed": [...], "failed": [...]}
        """
        if not packages:
            return {"installed": [], "failed": []}

        venv_path = self._venv_cache.get(session_id)
        if not venv_path or not venv_path.exists():
            python_path, _ = self._get_venv_python(session_id, [])
            if not python_path:
                return {"installed": [], "failed": [
                    {"package": ", ".join(packages), "error": "venv 创建失败"}
                ]}
            venv_path = self._venv_cache.get(session_id)

        if not venv_path or not venv_path.exists():
            return {"installed": [], "failed": [
                {"package": ", ".join(packages), "error": "venv 创建失败"}
            ]}

        python_path = str(
            venv_path / "Scripts" / "python.exe"
            if platform.system() == "Windows"
            else venv_path / "bin" / "python"
        )

        # 增量安装
        installed_set = self._get_installed_packages(python_path)
        missing = [p for p in packages
                   if p.lower().replace("-", "_") not in installed_set]

        if not missing:
            return {"installed": [], "failed": []}

        return _pre_install_packages(missing, python_path, timeout=300)

    def install_package(self, session_id: str, pkg_name: str) -> bool:
        """向指定 session 的 venv 安装单个包（用于 ModuleNotFoundError 后自动补装）

        Codex 策略：post_tools_node 检测到 ModuleNotFoundError 后调用此方法，
        自动在隔离 venv 中安装缺失的包，LLM 无需手动写 pip install。

        Args:
            session_id: 会话 ID
            pkg_name: 要安装的包名（如 "anndata"）

        Returns:
            安装成功返回 True
        """
        venv_path = self._venv_cache.get(session_id)
        if not venv_path or not venv_path.exists():
            # venv 不存在 → 先创建并安装
            python_path, _ = self._get_venv_python(session_id, [pkg_name])
            return python_path is not None

        python_path = str(
            venv_path / "Scripts" / "python.exe"
            if platform.system() == "Windows"
            else venv_path / "bin" / "python"
        )
        result = _pre_install_packages([pkg_name], python_path, timeout=180)
        if result["failed"]:
            _logger.warning("自动安装失败: %s → %s", pkg_name, result["failed"])
            return False
        _logger.info("自动安装成功: %s", pkg_name)
        return True

    def cleanup_session(self, session_id: str):
        """会话结束时销毁 venv 和 conda env（Codex 核心策略：用完即弃）

        Args:
            session_id: 要清理的会话 ID
        """
        # 清理 venv
        venv_dir = self._venv_cache.pop(session_id, None)
        if venv_dir and venv_dir.exists():
            shutil.rmtree(str(venv_dir), ignore_errors=True)
        # 清理 Python conda env
        if session_id in self._conda_session_cache:
            self._conda_manager.destroy_session_env(session_id)
            self._conda_session_cache.pop(session_id, None)
        # 清理 R conda env
        if session_id in self._conda_r_session_cache:
            self._conda_manager.destroy_session_env(session_id)
            self._conda_r_session_cache.pop(session_id, None)

    def cleanup_all_sessions(self):
        """清理所有 venv 和 conda env（程序退出时调用）"""
        for session_id in list(self._venv_cache.keys()):
            self.cleanup_session(session_id)
        # 清理所有 conda session envs（Python + R，可能在 cache 外）
        self._conda_manager.destroy_all()
        self._conda_session_cache.clear()
        self._conda_r_session_cache.clear()

    def _normalize_paths(self, code: str) -> str:
        """自动修正代码中的 Linux 绝对路径为当前 workspace 路径（兜底方案）

        Claude Code 做法的主方案是动态环境注入到系统提示词，让 LLM 知道自己在哪。
        但 LLM 偶尔还是会犯错使用 /workspace/ 路径，这里做最后一道兜底修正。
        Windows 上 Python 的 open() 也接受正斜杠路径，所以统一用正斜杠即可。

        关键：只替换绝对路径形式的 /workspace/（如 open("/workspace/...")），
        不碰相对路径（如 "./workspace/..."、"../workspace/..."），否则 Windows 上
        Path("./E:/path/...") 会被解析为 ".E:\\path\\..." 导致 WinError 123。
        """
        workspace_posix = str(self.workspace).replace("\\", "/")

        # 负向前瞻：确保 /workspace/ 前面不是 . 或单词字符（避免误触 ./ ../ x/workspace/）
        # 捕获组保留前缀字符（如引号、括号等）
        code = re.sub(
            r'(?<![.\w])/workspace/',
            f'{workspace_posix}/',
            code
        )
        # 末尾无斜杠的带引号形式："/workspace"  '/workspace'
        code = re.sub(
            r'(?<![.\w])"/workspace"',
            f'"{workspace_posix}"',
            code
        )
        code = re.sub(
            r"(?<![.\w])'/workspace'",
            f"'{workspace_posix}'",
            code
        )
        return code

    def execute_streaming(self, language: str, code: str, skill_id: str = "",
                          session_id: str = "") -> dict:
        """执行代码并实时将输出记录到日志，最终返回完整结果。

        与 execute() 相同的安全验证和 pip 预追踪逻辑，
        但使用 subprocess.Popen 逐行读取 stdout，实现实时日志输出。
        用户可通过日志实时观察长时间运行的任务（如 BLAST 搜索、模型训练）。
        """
        if language not in LANG_CONFIG:
            return {"success": False, "error": f"不支持的语言: {language}",
                    "stdout": "", "stderr": "", "api_error": None, "pip_result": None}

        code = self._normalize_paths(code)

        # ── ResilientExecutor 入口 hook：解决 P3/P4/P5 ──
        # 平台适配 / 路径注入 / 网络韧性 三层保护，
        # 与具体 skill / API 解耦。
        try:
            from src.middleware.resilient_executor import prepare_code_for_execution
            code = prepare_code_for_execution(code, language, self.workspace)
        except Exception as _re_err:
            _logger.warning("[ResilientExecutor] 入口注入失败（不影响执行）: %s", _re_err)

        safe, reason = self.validate(code, language)
        if not safe:
            return {"success": False, "error": reason, "stdout": "", "stderr": "",
                    "api_error": None, "pip_result": None}

        config = LANG_CONFIG[language]

        # ── Node.js / R 可用性检查 ──
        if language == "javascript":
            node_cmd = _resolve_node_cmd()
            if node_cmd is None:
                return {"success": False, "error": "Node.js 未安装或不在 PATH 中",
                        "stdout": "", "stderr": "", "api_error": None, "pip_result": None}
            config = {**config, "cmd": node_cmd}
        if language == "r":
            # ── 静态拦截 R 包安装命令，防止执行卡死 ──
            # install.packages() / BiocManager::install() 在 Windows 上需要源码编译，
            # 极慢（10-30 分钟）且无输出，容易导致终端假死。
            # R 包应由 conda 统一管理，LLM 不应自行安装。
            if re.search(r'\b(?:install\.packages|BiocManager::install)\s*\(', code, re.IGNORECASE):
                return {
                    "success": False,
                    "error": (
                        "代码中包含 R 包安装命令（install.packages / BiocManager::install），"
                        "这会导致执行卡死。系统已自动通过 conda 管理 R 包，"
                        "请移除安装代码后直接重试原逻辑。"
                        "如果重试仍缺包，请尝试使用 Python 替代方案（如 pydeseq2 替代 DESeq2）。"
                    ),
                    "stdout": "", "stderr": "", "api_error": None, "pip_result": None,
                }

            # ── 解析代码中的 R 包依赖（与 Python 提取 pip install 对称）──
            r_packages = _extract_r_packages(code)
            conda_r_pkgs = [c for p in r_packages if (c := _map_r_to_conda(p))]
            unknown_r_pkgs = [p for p in r_packages if not _map_r_to_conda(p)]

            if self._conda_manager.is_available():
                # ── 优先策略：通过 conda 创建隔离 R 环境（与 Python venv 对称）──
                # 无论系统是否有 R，统一走 conda 隔离环境，确保包依赖可控
                if unknown_r_pkgs:
                    _logger.warning("R packages without conda mapping: %s", unknown_r_pkgs)
                r_result = self._get_r_conda_env(session_id, conda_r_pkgs)
                if r_result["rscript_cmd"]:
                    config = {**config, "cmd": r_result["rscript_cmd"]}
                    if r_packages:
                        pip_result = {
                            "method": "conda_r_env",
                            "r_packages": r_packages,
                            "conda_packages": conda_r_pkgs,
                            "unknown": unknown_r_pkgs,
                            "status": r_result["status"],
                        }
                else:
                    # conda R 环境创建失败，回退到系统 R（如果有）
                    if which("Rscript"):
                        _logger.warning(
                            "conda R 环境创建失败 (%s)，回退到系统 R（可能缺少包: %s）",
                            r_result['status'], r_packages,
                        )
                    else:
                        return {
                            "success": False,
                            "error": (
                                "conda R 环境创建失败且系统无 Rscript。\n"
                                f"conda 状态: {r_result['status']}\n"
                                "建议：1) 检查 conda 是否可用 2) 使用 Python + PyDESeq2 替代方案"
                            ),
                            "stdout": "", "stderr": "", "api_error": None, "pip_result": None,
                        }

            elif which("Rscript"):
                # ── 兜底策略：conda 不可用但系统有 R ──
                if r_packages:
                    _logger.warning(
                        "conda 不可用，使用系统 R 执行（可能缺少包: %s）。"
                        "建议安装 miniconda 以获得自动 R 包管理能力。",
                        r_packages,
                    )
                # 使用系统 R 的默认命令（LANG_CONFIG 中的 ["Rscript"]）
            else:
                return {
                    "success": False,
                    "error": (
                        "Rscript 未安装，且 conda 也不可用。\n"
                        "建议：1) 安装 miniconda 以自动管理 R 环境 "
                        "2) 使用 Python + PyDESeq2 替代方案"
                    ),
                    "stdout": "", "stderr": "", "api_error": None, "pip_result": None,
                }

        # ── JavaScript npm 预安装 ──
        if language == "javascript" and _resolve_node_cmd():
            npm_packages = _extract_npm_installs(code)
            if npm_packages:
                npm_result = _pre_install_npm_packages(npm_packages, str(self.workspace))
                if npm_result["installed"]:
                    _logger.info("[npm预安装] %s", ", ".join(npm_result["installed"]))
                if npm_result["failed"]:
                    failed_names = [f["package"] for f in npm_result["failed"]]
                    _logger.warning("[npm预安装失败] %s", failed_names)

        # ── 第1层防御：主动安装技能依赖（Codex 策略）──
        deps_result = None
        if skill_id and self.use_venv and session_id:
            deps_result = self.install_skill_dependencies(session_id, skill_id)
        elif not skill_id and language == "python" and "tu.tools" in code and self.use_venv and session_id:
            # ── 兆底：LLM 未填 related_skill 但代码中使用了 tu.tools → 自动安装 tooluniverse ──
            _logger.info("[ToolUniverse自动检测] LLM 未填 related_skill，但代码含 tu.tools，自动安装 tooluniverse")
            python_path, _ = self._get_venv_python(session_id, ["tooluniverse"])
            if python_path:
                deps_result = {"installed": ["tooluniverse"], "failed": [], "skipped": []}

        # ── 环境隔离 + 包预安装（双层架构: venv 纯Python包 + conda C扩展包）──
        pip_result = None
        python_cmd = config["cmd"]
        if language == "python":
            packages = _extract_pip_installs(code)

            # ── 包分类：纯 Python vs C 扩展 ──
            pip_pkgs, conda_pkgs = _classify_packages(packages)

            if conda_pkgs and self._conda_manager.is_available():
                # ── 路径 A：conda 优先 —— C 扩展包走 conda per-session 隔离 ──
                conda_result = self._get_conda_env(session_id, conda_pkgs, pip_pkgs)
                if conda_result["python_cmd"]:
                    python_cmd = conda_result["python_cmd"]
                    all_pkgs = conda_pkgs + pip_pkgs
                    if all_pkgs:
                        pip_result = {
                            "method": "conda_env",
                            "packages": all_pkgs,
                            "conda_packages": conda_pkgs,
                            "pip_packages": pip_pkgs,
                            "status": conda_result["status"],
                        }
                else:
                    # conda 创建/安装失败，返回错误信息
                    return {
                        "success": False,
                        "error": (
                            f"conda 环境创建失败，以下 C 扩展包无法安装:\n"
                            + ", ".join(conda_pkgs) +
                            f"\n\n状态: {conda_result['status']}\n"
                            "请尝试使用纯 Python 替代方案重试。"
                        ),
                        "stdout": "", "stderr": "", "api_error": None, "pip_result": None,
                    }

            elif conda_pkgs and not self._conda_manager.is_available():
                # ── 路径 B：conda 不可用 —— 降级，给出纯 Python 替代方案提示 ──
                blocked_msgs = []
                for p in conda_pkgs:
                    key = p.lower().replace("-", "_")
                    fallback = CONDA_FALLBACK_MESSAGES.get(
                        key,
                        f"需要编译 C 扩展，pip 安装极易挂起。建议使用纯 Python 替代方案。"
                    )
                    blocked_msgs.append(f"  - {p}: {fallback}")
                return {
                    "success": False,
                    "error": (
                        "以下包需要 C 编译环境（conda 未安装）：\n"
                        + "\n".join(blocked_msgs) +
                        "\n\n请移除这些依赖并使用推荐的纯 Python 替代方案，"
                        "或安装 conda/miniconda 后重试。"
                    ),
                    "stdout": "", "stderr": "", "api_error": None, "pip_result": None,
                }

            else:
                # ── 路径 C：纯 Python 包 —— 现有 venv 路径（不变）──
                if self.use_venv and session_id:
                    venv_python, venv_status = self._get_venv_python(session_id, packages)
                    if venv_python:
                        python_cmd = [venv_python]
                        if packages:
                            pip_result = {"method": "venv", "packages": packages,
                                          "status": venv_status or "在虚拟环境中安装"}
                        else:
                            pip_result = None
                    elif packages:
                        pip_result = {"method": "venv_failed", "packages": packages,
                                      "status": venv_status}
                elif packages:
                    # venv 未启用：在系统 Python 中安装
                    pip_install_result = _pre_install_packages(packages, sys.executable, timeout=120)
                    pip_result = {"method": "system", "packages": packages,
                                  "installed": pip_install_result["installed"],
                                  "failed": pip_install_result["failed"],
                                  "success": pip_install_result["success"]}
                    if pip_install_result["failed"] and not pip_install_result["installed"]:
                        return {"success": False,
                                "error": f"包安装失败: {[f['package'] for f in pip_install_result['failed']]}",
                                "stdout": "", "stderr": "", "api_error": None, "pip_result": pip_result}

        # 第1层防御结果：注入到 pip_result
        if deps_result:
            deps_info = f"[技能依赖预安装] skill={skill_id}"
            if deps_result["installed"]:
                deps_info += f", 已安装: {', '.join(deps_result['installed'])}"
            if deps_result["skipped"]:
                deps_info += f", 已存在跳过: {', '.join(deps_result['skipped'])}"
            if deps_result["failed"]:
                failed_names = [f["package"] for f in deps_result["failed"]]
                deps_info += f", 安装失败: {', '.join(failed_names)}"
            if pip_result:
                pip_result["deps_pre_install"] = deps_info
            else:
                pip_result = {"method": "skill_deps", "status": deps_info}

        # ── JavaScript 后缀自适应（require → .cjs, import → .mjs）──
        suffix = config['suffix']
        if language == "javascript":
            has_require = bool(re.search(r'\brequire\s*\(', code))
            has_import = bool(re.search(r'\bimport\s+', code))
            if has_require and not has_import:
                suffix = ".cjs"
            elif has_require and has_import:
                # 混合：用 .cjs + 注入 createRequire 包装
                suffix = ".cjs"
                if 'import.meta.url' not in code and 'createRequire' not in code:
                    code = (
                        "import { createRequire } from 'module';\n"
                        "const require = createRequire(import.meta.url);\n" + code
                    )

        # ── HTTP 超时保护 + API 响应追踪 ──
        # LLM 生成的代码常忘记设置 timeout，导致外部 API 调用无限等待卡死执行
        # 同时，LLM 代码经常不 print API 响应，导致空输出时无法诊断问题
        # 通过 patch requests.Session.request，自动记录所有 HTTP 请求的响应到 stderr
        if language == "python":
            # ── ToolUniverse 自动注入：检测到 tu.tools 但无初始化代码时自动注入 ──
            # Agent 看到 SKILL.md 中的 tu.tools.* 示例但常忘记初始化 tu 对象
            # 系统层兜底：自动注入 from tooluniverse import ToolUniverse + tu 初始化
            _tu_preamble = ""
            if "tu.tools" in code and "from tooluniverse" not in code and "ToolUniverse" not in code:
                _tu_preamble = (
                    "# [系统注入] ToolUniverse 自动初始化\n"
                    "from tooluniverse import ToolUniverse\n"
                    "tu = ToolUniverse()\n"
                    "tu.load_tools()\n"
                )
                _logger.info("[ToolUniverse自动注入] 检测到 tu.tools 调用但未初始化，已自动注入")

            _http_preamble = (
                "# [系统注入] HTTP 超时保护（30秒）+ API 响应追踪\n"
                "import sys as _sys_http\n"
                "try:\n"
                "    import requests as _req\n"
                "    _orig_request = _req.Session.request\n"
                "    def _patched_request(self, method, url, **kw):\n"
                "        if 'timeout' not in kw or kw['timeout'] is None:\n"
                "            kw['timeout'] = 30\n"
                "        _resp = _orig_request(self, method, url, **kw)\n"
                "        # 记录 API 响应摘要到 stderr（方便日志追踪和问题定位）\n"
                "        _ctype = _resp.headers.get('content-type', '?')\n"
                "        _body_len = len(_resp.text) if hasattr(_resp, 'text') else 0\n"
                "        import sys as _sys_local\n"
                "        print(f'[API] {method.upper()} {url} → {_resp.status_code} | {_ctype} | {_body_len} chars', file=_sys_local.stderr)\n"
                "        # 非 200 或短/空响应 → 记录响应体（帮助诊断 API 返回空数据）\n"
                "        if _resp.status_code != 200 or _body_len < 200:\n"
                "            _preview = _resp.text[:500] if _body_len > 500 else _resp.text\n"
                "            print(f'[API-Body] {_preview}', file=_sys_local.stderr)\n"
                "        return _resp\n"
                "    _req.Session.request = _patched_request\n"
                "except ImportError:\n"
                "    pass\n"
                "finally:\n"
                "    del _sys_http\n"
            )
            code = _tu_preamble + _http_preamble + code

        # ── 写入脚本文件（session_id 隔离：并发时各线程写不同文件） ──
        _suffix_tag = f"_{session_id[:8]}" if session_id else ""
        script_path = self.workspace / f"run{_suffix_tag}{suffix}"
        script_path.write_text(code, encoding="utf-8")

        # ── 使用 Popen 流式执行 ──
        cmd = (python_cmd + [str(script_path)] if language == "python" and python_cmd != config["cmd"]
               else config["cmd"] + [str(script_path)])
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}

        try:
            process = subprocess.Popen(
                cmd,
                cwd=self.workspace.parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )

            stdout_lines = []
            stderr_lines = []
            start_time = time.time()

            # 逐行读取 stdout 和 stderr，实时记录到日志
            # 使用线程读取 stderr，避免主线程阻塞在 stdout 上
            import threading as _threading
            def _read_stderr(proc, buf):
                try:
                    for line in proc.stderr:
                        buf.append(line)
                        _logger.info("[exec-stderr] %s", line.rstrip())
                except Exception:
                    pass
            _stderr_thread = _threading.Thread(target=_read_stderr, args=(process, stderr_lines), daemon=True)
            _stderr_thread.start()

            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    stdout_lines.append(line)
                    _logger.info("[exec] %s", line.rstrip())

                # 超时检查
                if time.time() - start_time > self.timeout:
                    process.kill()
                    process.wait(timeout=5)
                    _stderr_thread.join(timeout=3)
                    return {"success": False, "error": f"超时 ({self.timeout}s)",
                            "stdout": "".join(stdout_lines)[-15000:],
                            "stderr": "".join(stderr_lines)[-5000:],
                            "api_error": None, "pip_result": pip_result}

            # 等待 stderr 线程完成
            _stderr_thread.join(timeout=5)

            stdout = "".join(stdout_lines)[-15000:]
            stderr = "".join(stderr_lines)[-5000:]

            # API 调用细粒度追踪
            api_error = None
            if process.returncode != 0:
                api_error = _detect_api_errors(stdout, stderr)
            if api_error is None and process.returncode == 0:
                api_warning = _detect_api_errors(stdout, stderr)
                if api_warning:
                    api_error = f"[警告] {api_warning}"

            # ── 空输出兜底：确保系统永远有输出 ──
            # LLM 生成的代码经常缺少空结果分支的 print 语句，
            # 导致 API 返回空数据时代码无输出，LLM 误认为"成功"而疯狂重试。
            # 当代码成功退出但 stdout/stderr 均无有效内容时，注入语义化提示。
            if process.returncode == 0 and not stdout.strip() and not stderr.strip():
                stdout = (
                    "[代码执行成功但无任何输出]\n"
                    "代码正常退出（returncode=0）但未产生任何标准输出。\n"
                    "这通常说明：代码逻辑中的条件分支未处理空结果情况（如 API 返回空数据时缺少 print）。\n"
                    "请检查代码逻辑并添加输出语句，或换用其他方法。"
                )
                _logger.warning("execute_code 返回空输出，已注入兜底提示")

            return {
                "success": process.returncode == 0,
                "stdout": stdout,
                "stderr": stderr,
                "returncode": process.returncode,
                "api_error": api_error,
                "pip_result": pip_result,
            }

        except FileNotFoundError:
            return {"success": False,
                    "error": f"{cmd[0]} 未安装或不在 PATH 中，请用 Python 替代",
                    "stdout": "", "stderr": "", "api_error": None, "pip_result": None}
        except Exception as e:
            return {"success": False, "error": str(e),
                    "stdout": "", "stderr": "", "api_error": None, "pip_result": pip_result}

