import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml


SKILLS_ROOT = Path("./skills")


@dataclass
class SkillAuditResult:
    skill_id: str
    path: str
    ok: bool
    has_skill_md: bool
    skill_md_bytes: int | None
    utf8_ok: bool
    has_frontmatter: bool
    description: str | None
    description_empty: bool
    has_examples_dir: bool
    has_references_dir: bool
    has_scripts_dir: bool
    has_any_scripts: bool
    root_script_files: list[str]
    scripts_dir_files: list[str]
    examples_files: list[str]
    references_files: list[str]
    errors: list[str]


def _parse_frontmatter(text: str) -> tuple[bool, dict[str, Any]]:
    m = re.search(r"---\s*\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return False, {}
    try:
        data = yaml.safe_load(m.group(1)) or {}
        if not isinstance(data, dict):
            return True, {}
        return True, data
    except Exception:
        return True, {}


def audit_skill(skill_dir: Path) -> SkillAuditResult:
    skill_id = skill_dir.name
    errors: list[str] = []

    skill_md = skill_dir / "SKILL.md"
    has_skill_md = skill_md.exists()
    skill_md_bytes: int | None = None
    utf8_ok = False
    has_frontmatter = False
    description: str | None = None

    if has_skill_md:
        try:
            raw = skill_md.read_bytes()
            skill_md_bytes = len(raw)
            text = raw.decode("utf-8")
            utf8_ok = True
            has_frontmatter, fm = _parse_frontmatter(text)
            desc = fm.get("description") if isinstance(fm, dict) else None
            if isinstance(desc, str):
                description = desc.strip()
            elif desc is not None:
                description = str(desc).strip()
        except UnicodeDecodeError:
            errors.append("SKILL.md is not valid UTF-8")
        except Exception as e:
            errors.append(f"Failed to read SKILL.md: {e}")
    else:
        errors.append("Missing SKILL.md")

    description_empty = (description is None) or (description.strip() == "")

    ex_dir = skill_dir / "examples"
    ref_dir = skill_dir / "references"
    scr_dir = skill_dir / "scripts"

    root_script_files: list[str] = []
    for pattern in ("*.py", "*.sh", "*.ps1", "*.R"):
        for p in sorted(skill_dir.glob(pattern)):
            if p.is_file():
                root_script_files.append(p.name)

    scripts_dir_files = [p.name for p in sorted(scr_dir.iterdir()) if p.is_file()] if scr_dir.is_dir() else []
    examples_files = [p.name for p in sorted(ex_dir.iterdir()) if p.is_file()] if ex_dir.is_dir() else []
    references_files = [p.name for p in sorted(ref_dir.iterdir()) if p.is_file()] if ref_dir.is_dir() else []
    has_any_scripts = bool(root_script_files or scripts_dir_files)

    ok = has_skill_md and utf8_ok and not description_empty

    return SkillAuditResult(
        skill_id=skill_id,
        path=str(skill_dir).replace("\\", "/"),
        ok=ok,
        has_skill_md=has_skill_md,
        skill_md_bytes=skill_md_bytes,
        utf8_ok=utf8_ok,
        has_frontmatter=has_frontmatter,
        description=description,
        description_empty=description_empty,
        has_examples_dir=ex_dir.is_dir(),
        has_references_dir=ref_dir.is_dir(),
        has_scripts_dir=scr_dir.is_dir(),
        has_any_scripts=has_any_scripts,
        root_script_files=root_script_files,
        scripts_dir_files=scripts_dir_files,
        examples_files=examples_files,
        references_files=references_files,
        errors=errors,
    )


def main() -> int:
    if not SKILLS_ROOT.is_dir():
        raise SystemExit("Missing ./skills directory")

    skill_dirs = [p for p in sorted(SKILLS_ROOT.iterdir()) if p.is_dir()]
    results: list[SkillAuditResult] = [audit_skill(d) for d in skill_dirs]

    summary = {
        "skills_total": len(results),
        "ok_total": sum(1 for r in results if r.ok),
        "missing_skill_md": sum(1 for r in results if not r.has_skill_md),
        "utf8_errors": sum(1 for r in results if r.has_skill_md and not r.utf8_ok),
        "empty_description": sum(1 for r in results if r.description_empty),
        "has_examples_dir": sum(1 for r in results if r.has_examples_dir),
        "has_references_dir": sum(1 for r in results if r.has_references_dir),
        "has_scripts_dir": sum(1 for r in results if r.has_scripts_dir),
        "has_any_scripts": sum(1 for r in results if r.has_any_scripts),
    }

    output = {
        "summary": summary,
        "results": [asdict(r) for r in results],
    }

    out_path = Path("skill_audit_report.json")
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

