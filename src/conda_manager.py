"""Conda 环境管理器：per-session 隔离 + 模板克隆优化

核心策略：
- 轻包（纯 Python）→ 走现有 venv + pip 路径（快，2-5秒）
- 重包（C 扩展/CLI 工具）→ 走 conda env per-session（隔离，用完即删）
- 模板克隆：首次创建 _openclaw_bio_template，后续 session 从模板克隆（~10秒 vs 60秒）
- 没有 conda → 降级到纯 Python 替代方案
"""
import json
import logging
import subprocess
import uuid
from pathlib import Path
from shutil import which
from typing import Optional

_logger = logging.getLogger("agent.conda_manager")


class CondaEnvManager:
    """管理 per-session conda 环境，通过模板克隆实现快速创建。

    模板环境（_openclaw_bio_template）只创建一次，包含基础 Python。
    每个 session 从模板克隆（硬链接复制），然后 conda install 所需 C 扩展包。
    session 结束后 conda env remove 销毁，零残留。
    """

    TEMPLATE_NAME = "_openclaw_bio_template"
    R_TEMPLATE_NAME = "_openclaw_r_template"
    _SINGLETON = None  # type: Optional["CondaEnvManager"]

    def __new__(cls):
        if cls._SINGLETON is None:
            cls._SINGLETON = super().__new__(cls)
        return cls._SINGLETON

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self._conda_available = which("conda") is not None
        self._template_ready = False
        self._r_template_ready = False
        self._python_paths: dict[str, str] = {}  # env_name -> python_exe_path
        self._session_envs: dict[str, str] = {}  # session_id -> python_env_name
        self._r_session_envs: dict[str, str] = {}  # session_id -> r_env_name

        if self._conda_available:
            _logger.info("检测到 conda，C 扩展包将走 conda 路径")
            self._enable_libmamba()
            self.cleanup_orphans()
        else:
            _logger.info("未检测到 conda，C 扩展包将使用纯 Python 替代方案")

    def _enable_libmamba(self):
        """尝试启用 conda libmamba solver（比经典 solver 快 5-10 倍）"""
        ret, _, _ = self._run_conda(
            ["config", "--set", "solver", "libmamba"],
            timeout=10,
        )
        if ret == 0:
            _logger.info("已启用 conda libmamba solver（加速依赖解析）")
        else:
            _logger.debug("libmamba solver 不可用，使用默认 solver")

    # ─── 公共接口 ───

    def is_available(self) -> bool:
        """conda 是否可用"""
        return self._conda_available

    def ensure_template(self) -> bool:
        """确保模板 conda 环境存在（一次性操作，幂等）"""
        if not self._conda_available:
            return False
        if self._template_ready:
            return True

        if self._template_exists():
            self._template_ready = True
            _logger.info("conda 模板环境已存在: %s", self.TEMPLATE_NAME)
            return True

        _logger.info("首次创建 conda 模板环境: %s ...", self.TEMPLATE_NAME)
        ret, _, stderr = self._run_conda(
            ["create", "-n", self.TEMPLATE_NAME, "python=3.11", "-y"],
            timeout=300,
        )
        if ret != 0:
            _logger.error("创建 conda 模板环境失败: %s", stderr[:500])
            return False

        _logger.info("conda 模板环境创建成功: %s", self.TEMPLATE_NAME)
        self._template_ready = True
        return True

    def create_session_env(self, session_id: str, conda_packages: list[str]) -> tuple[Optional[str], str]:
        """为 session 创建隔离 conda 环境（模板克隆 + 包安装）

        Args:
            session_id: 会话 ID
            conda_packages: 需要通过 conda 安装的包名列表

        Returns:
            (env_name, status_message) — env_name 为 None 表示失败
        """
        if not self._conda_available:
            return None, "[conda 不可用]"

        if not self.ensure_template():
            return None, "[conda 模板环境创建失败]"

        # 检查该 session 是否已有 conda env（复用）
        existing = self._session_envs.get(session_id)
        if existing:
            _logger.info("复用 session conda 环境: %s", existing)
            return existing, "[conda 环境复用]"

        # 生成唯一环境名
        env_name = f"_openclaw_sess_{session_id[:8]}_{uuid.uuid4().hex[:4]}"

        # 容错：清理可能残留的同名环境
        self._run_conda(["env", "remove", "-n", env_name, "-y"], timeout=30)

        # 从模板克隆（硬链接，~10秒）
        _logger.info("从模板克隆 conda 环境: %s -> %s", self.TEMPLATE_NAME, env_name)
        ret, _, stderr = self._run_conda(
            ["create", "--clone", self.TEMPLATE_NAME, "-n", env_name, "-y"],
            timeout=120,
        )
        if ret != 0:
            _logger.error("克隆 conda 环境失败: %s", stderr[:500])
            return None, f"[conda 克隆失败: {stderr[:200]}]"

        _logger.info("conda 环境克隆成功: %s", env_name)

        # 安装 C 扩展包
        if conda_packages:
            _logger.info("在 conda 环境 %s 中安装包: %s", env_name, conda_packages)
            ret, stdout, stderr = self._run_conda(
                ["install", "-n", env_name, "-c", "bioconda", "-c", "conda-forge", "-y"] + conda_packages,
                timeout=300,
            )
            if ret != 0:
                _logger.warning("conda 安装部分包可能失败: %s", stderr[:500])
                # 不阻断——部分包可能已成功安装

        self._session_envs[session_id] = env_name
        return env_name, f"[conda 环境已创建: {env_name}]"

    def install_pip_packages(self, env_name: str, packages: list[str], timeout: int = 180) -> tuple[bool, str]:
        """在已有 conda 环境中用 pip 安装纯 Python 包

        Returns:
            (success, status_message)
        """
        if not packages:
            return True, ""

        python_exe = self._get_env_python_exe(env_name)
        if not python_exe:
            return False, "[conda python 路径获取失败]"

        _logger.info("conda 环境 %s 中 pip 安装: %s", env_name, packages)
        ret, stdout, stderr = self._run_raw(
            [python_exe, "-m", "pip", "install", "--quiet"] + packages,
            timeout=timeout,
        )
        if ret != 0:
            return False, f"[conda pip 安装失败: {stderr[:200]}]"
        return True, f"[conda pip 安装成功: {packages}]"

    def get_python_cmd(self, session_id: str) -> Optional[list[str]]:
        """获取 session 对应 conda 环境的 python 执行命令

        Returns:
            ["conda", "run", "-n", env_name, "python"] 或 None
        """
        env_name = self._session_envs.get(session_id)
        if not env_name:
            return None
        return ["conda", "run", "-n", env_name, "python"]

    # ─── R 语言 conda 环境管理 ───

    def ensure_r_template(self) -> bool:
        """确保 R 模板 conda 环境存在（含 r-base=4.3），一次性操作，幂等"""
        if not self._conda_available:
            return False
        if self._r_template_ready:
            return True

        if self._r_template_exists():
            self._r_template_ready = True
            _logger.info("R conda 模板环境已存在: %s", self.R_TEMPLATE_NAME)
            return True

        _logger.info("首次创建 R conda 模板环境: %s (r-base=4.3)...", self.R_TEMPLATE_NAME)
        ret, _, stderr = self._run_conda(
            ["create", "-n", self.R_TEMPLATE_NAME,
             "-c", "conda-forge", "r-base=4.3", "-y"],
            timeout=600,
        )
        if ret != 0:
            _logger.error("创建 R conda 模板环境失败: %s", stderr[:500])
            return False

        _logger.info("R conda 模板环境创建成功: %s", self.R_TEMPLATE_NAME)
        self._r_template_ready = True
        return True

    def create_r_session_env(self, session_id: str,
                              conda_r_packages: list[str]) -> tuple[Optional[str], str]:
        """为 session 创建隔离 R conda 环境（从 R 模板克隆 + 安装 R 包）

        Args:
            session_id: 会话 ID
            conda_r_packages: 需要通过 conda 安装的 R 包名列表（已映射为 conda 包名）

        Returns:
            (env_name, status_message) — env_name 为 None 表示失败
        """
        if not self._conda_available:
            return None, "[conda 不可用]"

        if not self.ensure_r_template():
            return None, "[R conda 模板环境创建失败]"

        # 检查该 session 是否已有 R conda env（复用）
        existing = self._r_session_envs.get(session_id)
        if existing:
            _logger.info("复用 session R conda 环境: %s", existing)
            return existing, "[R conda 环境复用]"

        # 生成唯一环境名（R 环境使用 _openclaw_r_sess_ 前缀）
        env_name = f"_openclaw_r_sess_{session_id[:8]}_{uuid.uuid4().hex[:4]}"

        # 容错：清理可能残留的同名环境
        self._run_conda(["env", "remove", "-n", env_name, "-y"], timeout=30)

        # 从 R 模板克隆
        _logger.info("从 R 模板克隆 conda 环境: %s -> %s", self.R_TEMPLATE_NAME, env_name)
        ret, _, stderr = self._run_conda(
            ["create", "--clone", self.R_TEMPLATE_NAME, "-n", env_name, "-y"],
            timeout=120,
        )
        if ret != 0:
            _logger.error("克隆 R conda 环境失败: %s", stderr[:500])
            return None, f"[R conda 克隆失败: {stderr[:200]}]"

        _logger.info("R conda 环境克隆成功: %s", env_name)

        # 安装 R 包（DESeq2, ggplot2 等）
        if conda_r_packages:
            _logger.info("在 R conda 环境 %s 中安装包: %s", env_name, conda_r_packages)
            ret, stdout, stderr = self._run_conda(
                ["install", "-n", env_name,
                 "-c", "bioconda", "-c", "conda-forge", "-y"] + conda_r_packages,
                timeout=600,
            )
            if ret != 0:
                _logger.error("R conda 安装失败: %s", stderr[:500])
                # 清理半成品环境，避免 LLM 拿到缺少包的环境后自行 BiocManager::install 卡死
                self._run_conda(["env", "remove", "-n", env_name, "-y"], timeout=30)
                return None, f"[R conda 包安装失败: {stderr[:200]}]"

        self._r_session_envs[session_id] = env_name
        return env_name, f"[R conda 环境已创建: {env_name}]"

    def get_rscript_cmd(self, session_id: str) -> Optional[list[str]]:
        """获取 session 对应 R conda 环境的 Rscript 执行命令

        Returns:
            ["conda", "run", "-n", env_name, "Rscript"] 或 None
        """
        env_name = self._r_session_envs.get(session_id)
        if not env_name:
            return None
        return ["conda", "run", "-n", env_name, "Rscript"]

    def destroy_session_env(self, session_id: str):
        """销毁 session 的 conda 环境（Python + R）"""
        # 清理 Python conda 环境
        env_name = self._session_envs.pop(session_id, None)
        if env_name:
            _logger.info("销毁 Python conda 环境: %s", env_name)
            self._run_conda(["env", "remove", "-n", env_name, "-y"], timeout=60)
            self._python_paths.pop(env_name, None)

        # 清理 R conda 环境
        r_env_name = self._r_session_envs.pop(session_id, None)
        if r_env_name:
            _logger.info("销毁 R conda 环境: %s", r_env_name)
            self._run_conda(["env", "remove", "-n", r_env_name, "-y"], timeout=60)

    def destroy_all(self):
        """销毁所有 session conda 环境（Python + R）"""
        all_sids = set(list(self._session_envs.keys()) + list(self._r_session_envs.keys()))
        for sid in all_sids:
            self.destroy_session_env(sid)

    def cleanup_orphans(self):
        """清理上次 crash 残留的孤儿 conda 环境（Python + R）"""
        if not self._conda_available:
            return
        ret, stdout, _ = self._run_conda(["env", "list"], timeout=30)
        if ret != 0:
            return
        for line in stdout.split("\n"):
            line_lower = line.strip().lower()
            # 匹配模式: _openclaw_sess_ 或 _openclaw_r_sess_ 开头的环境
            if "_openclaw_sess_" in line_lower or "_openclaw_r_sess_" in line_lower:
                parts = line.split()
                if parts:
                    env_name = parts[0]
                    known = (env_name in self._session_envs.values()
                             or env_name in self._r_session_envs.values())
                    if not known:
                        _logger.info("清理孤儿 conda 环境: %s", env_name)
                        self._run_conda(["env", "remove", "-n", env_name, "-y"], timeout=30)

    def reset_template(self):
        """重建模板环境（用于依赖更新）"""
        if self._template_exists():
            _logger.info("重建 conda 模板环境")
            self._run_conda(["env", "remove", "-n", self.TEMPLATE_NAME, "-y"], timeout=60)
        self._template_ready = False
        return self.ensure_template()

    def reset_r_template(self):
        """重建 R 模板环境（用于依赖更新）"""
        if self._r_template_exists():
            _logger.info("重建 R conda 模板环境")
            self._run_conda(["env", "remove", "-n", self.R_TEMPLATE_NAME, "-y"], timeout=60)
        self._r_template_ready = False
        return self.ensure_r_template()

    # ─── 内部方法 ───

    def _template_exists(self) -> bool:
        """检查模板环境是否已存在"""
        ret, stdout, _ = self._run_conda(["env", "list", "--json"], timeout=30)
        if ret != 0:
            return False
        try:
            data = json.loads(stdout)
            for env_path in data.get("envs", []):
                if Path(env_path).name == self.TEMPLATE_NAME:
                    return True
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        # JSON 解析失败时的文本兜底
        ret2, stdout2, _ = self._run_conda(["env", "list"], timeout=30)
        return self.TEMPLATE_NAME in stdout2

    def _r_template_exists(self) -> bool:
        """检查 R 模板环境是否已存在"""
        ret, stdout, _ = self._run_conda(["env", "list", "--json"], timeout=30)
        if ret != 0:
            return False
        try:
            data = json.loads(stdout)
            for env_path in data.get("envs", []):
                if Path(env_path).name == self.R_TEMPLATE_NAME:
                    return True
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        ret2, stdout2, _ = self._run_conda(["env", "list"], timeout=30)
        return self.R_TEMPLATE_NAME in stdout2

    def _get_env_python_exe(self, env_name: str) -> Optional[str]:
        """获取 conda 环境中 python 可执行文件的路径（缓存）"""
        if env_name in self._python_paths:
            cached = self._python_paths[env_name]
            if Path(cached).exists():
                return cached

        # 通过 conda run 获取 sys.executable
        ret, stdout, _ = self._run_conda(
            ["run", "-n", env_name, "python", "-c", "import sys; print(sys.executable)"],
            timeout=10,
        )
        if ret == 0:
            path = stdout.strip()
            if path:
                self._python_paths[env_name] = path
                return path
        return None

    def _run_conda(self, args: list[str], timeout: int = 180) -> tuple[int, str, str]:
        """执行 conda 命令，返回 (returncode, stdout, stderr)"""
        return self._run_raw(["conda"] + args, timeout=timeout)

    @staticmethod
    def _run_raw(cmd: list[str], timeout: int = 180) -> tuple[int, str, str]:
        """执行任意命令，长时间运行的命令会输出心跳日志"""
        _logger.debug("执行: %s", " ".join(cmd))
        try:
            import time
            # 对于短时间命令（timeout <= 60s），用简单模式
            if timeout <= 60:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    encoding="utf-8",
                    errors="replace",
                )
                return proc.returncode, proc.stdout, proc.stderr

            # 对于长时间命令，用 Popen 实现心跳日志
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            start = time.time()
            last_heartbeat = start
            while proc.poll() is None:
                elapsed = time.time() - start
                if elapsed > timeout:
                    proc.kill()
                    proc.wait()
                    return -1, "", f"命令超时 ({timeout}s): {' '.join(cmd)}"
                # 每 30 秒输出一次心跳
                if time.time() - last_heartbeat > 30:
                    # 提取命令的关键信息（前 3 个参数）
                    cmd_preview = " ".join(cmd[:4]) if len(cmd) > 4 else " ".join(cmd)
                    _logger.info("⏳ 安装中... (%.0f 秒) %s", elapsed, cmd_preview)
                    last_heartbeat = time.time()
                time.sleep(1)
            stdout, stderr = proc.communicate()
            return proc.returncode, stdout, stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"命令超时 ({timeout}s): {' '.join(cmd)}"
        except Exception as e:
            return -1, "", str(e)
