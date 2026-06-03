"""技能上下文：首次注入骨架，按需读取具体文件"""
import re
from pathlib import Path

class SkillContext:
    def __init__(self, skills_root: str = "./skills"):
        self.skills_root = Path(skills_root)

    def load_skeleton(self, skill_id: str) -> str:
        """注入 SKILL.md 全文 + 文件清单（含摘要），不含 examples/references 实际内容"""
        skill_dir = self.skills_root / skill_id
        parts = []

        # SKILL.md 全文（去掉 HTML 注释）
        skill_md = skill_dir / "SKILL.md"
        if skill_md.exists():
            text = skill_md.read_text(encoding="utf-8")
            text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
            parts.append(text)

        # 文件清单（含摘要）
        manifest = self._build_manifest(skill_dir)
        if manifest:
            parts.append("## 可用文件清单\n" + manifest)

        return "\n\n---\n\n".join(parts)

    def load_summary(self, skill_id: str, description: str = "") -> str:
        """分层注入策略：只返回技能描述 + 文件清单（含 SKILL.md），不注入 SKILL.md 全文。

        LLM 需要调用 read_file(skill_id, "SKILL.md") 获取完整技能文档。
        这避免了 top_k=8 个 SKILL.md 全文一次性注入导致 token 爆炸。
        """
        skill_dir = self.skills_root / skill_id
        parts = []

        # 技能描述
        if description:
            parts.append(f"**描述**: {description}")

        # 文件清单（含 SKILL.md）
        file_parts = []
        skill_md = skill_dir / "SKILL.md"
        if skill_md.exists():
            size_kb = skill_md.stat().st_size / 1024
            file_parts.append(f"- `SKILL.md` — 技能核心文档，包含完整API说明、代码模式和使用指导 ({size_kb:.1f} KB)")

        manifest = self._build_manifest(skill_dir)
        if manifest:
            file_parts.append(manifest)

        if file_parts:
            parts.append("## 可用文件清单\n" + "\n".join(file_parts))

        return "\n\n".join(parts)

    def _build_manifest(self, skill_dir: Path) -> str:
        """构建文件清单，每个文件附带一句话摘要"""
        lines = []

        # examples/
        ex_dir = skill_dir / "examples"
        if ex_dir.is_dir():
            lines.append("### examples/")
            for f in sorted(ex_dir.iterdir()):
                if f.is_file():
                    summary = self._summarize(f)
                    lines.append(f"- `examples/{f.name}` — {summary}")

        # references/
        ref_dir = skill_dir / "references"
        if ref_dir.is_dir():
            lines.append("### references/")
            for f in sorted(ref_dir.iterdir()):
                if f.is_file():
                    summary = self._summarize(f)
                    lines.append(f"- `references/{f.name}` — {summary}")

        # 根目录脚本
        root_scripts = []
        for pattern in ("*.py", "*.sh", "*.ps1", "*.R"):
            for f in skill_dir.glob(pattern):
                root_scripts.append(f)
        if root_scripts:
            lines.append("### 根目录脚本")
            for f in sorted(root_scripts):
                summary = self._summarize(f)
                f_path = str(skill_dir / f.name).replace('\\', '/')
                lines.append(f"- `{f.name}` — {summary} (完整路径: {f_path})")

        # scripts/ 目录
        scr_dir = skill_dir / "scripts"
        if scr_dir.is_dir():
            lines.append("### scripts/")
            for f in sorted(scr_dir.iterdir()):
                if f.is_file():
                    summary = self._summarize(f)
                    f_path = str(scr_dir / f.name).replace('\\', '/')
                    lines.append(f"- `scripts/{f.name}` — {summary} (完整路径: {f_path})")

        return "\n".join(lines) if lines else ""

    def _summarize(self, filepath: Path) -> str:
        """提取文件的一句话摘要：.py 取首行注释，.md 取首个标题，其他取文件名"""
        try:
            first_lines = filepath.read_text(encoding="utf-8")[:500]
            if filepath.suffix == ".py":
                # 取首个非空非注释行作为摘要
                for line in first_lines.split("\n"):
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        return stripped.lstrip("#").strip()[:120]
                    if stripped and not stripped.startswith(("import", "from", "__")):
                        return stripped[:120]
            elif filepath.suffix == ".md":
                for line in first_lines.split("\n"):
                    if line.strip().startswith("#"):
                        return line.strip().lstrip("#").strip()[:120]
            return f"{filepath.suffix} 文件, {filepath.stat().st_size} 字节"
        except Exception:
            return "无法读取摘要"
