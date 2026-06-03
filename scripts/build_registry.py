"""扫描 skills/ 目录，生成 skill_registry.json"""
import json, yaml, re
from pathlib import Path

SKILLS_ROOT = Path("./skills")

def parse_frontmatter(text: str) -> dict:
    m = re.search(r'---\s*\n(.*?)\n---', text, re.DOTALL)
    return yaml.safe_load(m.group(1)) if m else {}

def scan_skill(skill_dir: Path) -> dict:
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return None

    content = skill_md.read_text(encoding="utf-8")
    fm = parse_frontmatter(content)

    # 代码语言检测
    languages = [lang for lang in ["python", "bash", "r"]
                 if re.search(rf'```{lang}', content, re.IGNORECASE)]

    # examples/
    examples = []
    ex_dir = skill_dir / "examples"
    if ex_dir.is_dir():
        for f in ex_dir.iterdir():
            if f.is_file():
                examples.append({"name": f"examples/{f.name}",
                                 "size": f.stat().st_size})

    # references/
    references = []
    ref_dir = skill_dir / "references"
    if ref_dir.is_dir():
        for f in ref_dir.iterdir():
            if f.is_file():
                references.append({"name": f"references/{f.name}",
                                   "size": f.stat().st_size})

    # scripts/ (可执行脚本)
    scripts = []
    scr_dir = skill_dir / "scripts"
    if scr_dir.is_dir():
        for f in scr_dir.iterdir():
            if f.is_file():
                scripts.append({"name": f"scripts/{f.name}",
                                "path": str(skill_dir / "scripts" / f.name).replace('\\', '/')})

    # 根目录脚本
    for pattern in ("*.py", "*.sh", "*.ps1", "*.R"):
        for f in skill_dir.glob(pattern):
            scripts.append({"name": f.name,
                            "path": str(skill_dir / f.name).replace('\\', '/')})

    # 其他子目录
    extra_dirs = [d.name for d in skill_dir.iterdir()
                  if d.is_dir() and d.name not in
                  ("examples", "scripts", "references")]

    return {
        "id": skill_dir.name,
        "description": fm.get("description", ""),
        "path": f"./skills/{skill_dir.name}",
        "has_examples": len(examples) > 0,
        "has_references": len(references) > 0,
        "has_scripts": len(scripts) > 0,
        "examples": examples,
        "references": references,
        "scripts": scripts,
        "code_languages": languages,
        "extra_dirs": extra_dirs,
    }

def main():
    registry = []
    for skill_dir in sorted(SKILLS_ROOT.iterdir()):
        if skill_dir.is_dir():
            skill = scan_skill(skill_dir)
            if skill:
                registry.append(skill)
            else:
                print(f"SKIP {skill_dir.name}: no SKILL.md")

    with open("skill_registry.json", "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    print(f"Registered {len(registry)} skills.")

if __name__ == "__main__":
    main()
