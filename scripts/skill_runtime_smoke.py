import ast
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path


REPORT_IN = Path("skill_audit_report.json")
SKILLS_ROOT = Path("./skills")


DEFAULT_SEED = 42
DEFAULT_SCRIPTS_SKILLS = 30
DEFAULT_EXAMPLES_SKILLS = 30
DEFAULT_MAX_PY_FILES_PER_SKILL = 3
DEFAULT_MAX_IMPORTS_PER_FILE = 25
DEFAULT_IMPORT_TIMEOUT_S = 20


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return int(v.strip())
    except Exception:
        return default


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _pick_sample(rng: random.Random, items: list[str], n: int) -> list[str]:
    if n <= 0:
        return []
    if len(items) <= n:
        return list(items)
    return rng.sample(items, n)


def _resolve_skill_dir(skill_id: str) -> Path:
    return (SKILLS_ROOT / skill_id).resolve()


def _iter_candidate_files(skill_id: str, kind: str, names: list[str]) -> list[Path]:
    base = _resolve_skill_dir(skill_id)
    results: list[Path] = []
    for n in names:
        if kind == "root":
            p = base / n
        else:
            p = base / kind / n
        if p.is_file():
            results.append(p)
    return results


def _find_py_files(skill_id: str, audit_row: dict) -> list[Path]:
    files: list[Path] = []
    root_names = audit_row.get("root_script_files") or []
    script_names = audit_row.get("scripts_dir_files") or []
    example_names = audit_row.get("examples_files") or []

    files.extend([p for p in _iter_candidate_files(skill_id, "root", root_names) if p.suffix.lower() == ".py"])
    files.extend([p for p in _iter_candidate_files(skill_id, "scripts", script_names) if p.suffix.lower() == ".py"])
    files.extend([p for p in _iter_candidate_files(skill_id, "examples", example_names) if p.suffix.lower() == ".py"])

    seen: set[str] = set()
    unique: list[Path] = []
    for p in files:
        key = str(p.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


def _find_sh_files(skill_id: str, audit_row: dict) -> list[Path]:
    files: list[Path] = []
    root_names = audit_row.get("root_script_files") or []
    script_names = audit_row.get("scripts_dir_files") or []
    example_names = audit_row.get("examples_files") or []

    files.extend([p for p in _iter_candidate_files(skill_id, "root", root_names) if p.suffix.lower() in {".sh", ".ps1"}])
    files.extend([p for p in _iter_candidate_files(skill_id, "scripts", script_names) if p.suffix.lower() in {".sh", ".ps1"}])
    files.extend([p for p in _iter_candidate_files(skill_id, "examples", example_names) if p.suffix.lower() in {".sh", ".ps1"}])
    return files


def _find_r_files(skill_id: str, audit_row: dict) -> list[Path]:
    files: list[Path] = []
    root_names = audit_row.get("root_script_files") or []
    script_names = audit_row.get("scripts_dir_files") or []
    example_names = audit_row.get("examples_files") or []

    files.extend([p for p in _iter_candidate_files(skill_id, "root", root_names) if p.suffix.lower() == ".r"])
    files.extend([p for p in _iter_candidate_files(skill_id, "scripts", script_names) if p.suffix.lower() == ".r"])
    files.extend([p for p in _iter_candidate_files(skill_id, "examples", example_names) if p.suffix.lower() == ".r"])
    return files


def _extract_python_imports(py_path: Path, max_imports: int) -> list[str]:
    src = py_path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(src, filename=str(py_path))
    except Exception:
        return []
    mods: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name.split(".")[0].strip()
                if name and name not in mods:
                    mods.append(name)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue
            if not node.module:
                continue
            name = node.module.split(".")[0].strip()
            if name and name not in mods:
                mods.append(name)
        if len(mods) >= max_imports:
            break

    return mods[:max_imports]


def _try_import(module_name: str, timeout_s: int) -> tuple[bool, str | None]:
    code = (
        "import importlib\n"
        f"importlib.import_module({module_name!r})\n"
        "print('ok')\n"
    )
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if r.returncode == 0:
            return True, None
        err = (r.stderr or r.stdout or "").strip()
        return False, err[:400] if err else "import failed"
    except subprocess.TimeoutExpired:
        return False, "import timeout"
    except Exception as e:
        return False, f"import error: {e}"


_BASH_BUILTINS = {
    "cd", "echo", "export", "set", "unset", "test", "[", "]", "if", "then", "fi", "for", "do", "done", "while",
    "case", "esac", "function", "return", "exit", "read", "printf", "pwd", "true", "false",
}


def _extract_shell_commands(path: Path, limit: int = 100) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    cmds: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith(("<<", "{", "}", "(")) or stripped.endswith(":"):
            continue
        first = stripped.split("#", 1)[0].strip()
        if not first:
            continue
        try:
            parts = shlex.split(first, posix=True)
        except Exception:
            parts = re.split(r"\s+", first)
        if not parts:
            continue
        cmd = parts[0]
        cmd = cmd.strip()
        if not cmd or cmd in _BASH_BUILTINS:
            continue
        cmd = cmd.split("/")[-1]
        if cmd and cmd not in cmds:
            cmds.append(cmd)
        if len(cmds) >= limit:
            break
    return cmds


def _extract_r_packages(path: Path, limit: int = 50) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    pkgs: list[str] = []
    for m in re.finditer(r"\b(?:library|require)\s*\(\s*([A-Za-z0-9_.]+)\s*\)", text):
        pkg = m.group(1)
        if pkg and pkg not in pkgs:
            pkgs.append(pkg)
        if len(pkgs) >= limit:
            break
    return pkgs


@dataclass
class SkillSmoke:
    skill_id: str
    python_files_checked: list[str]
    python_missing_modules: dict[str, str]
    shell_files_checked: list[str]
    shell_missing_commands: list[str]
    r_files_checked: list[str]
    rscript_available: bool
    r_packages: list[str]


def main() -> int:
    if not REPORT_IN.exists():
        raise SystemExit(f"Missing {REPORT_IN}. Run: python .\\scripts\\skill_audit.py")
    if not SKILLS_ROOT.is_dir():
        raise SystemExit("Missing ./skills directory")

    seed = _env_int("SMOKE_SEED", DEFAULT_SEED)
    n_scripts = _env_int("SMOKE_SCRIPTS_SKILLS", DEFAULT_SCRIPTS_SKILLS)
    n_examples = _env_int("SMOKE_EXAMPLES_SKILLS", DEFAULT_EXAMPLES_SKILLS)
    max_py_files = _env_int("SMOKE_MAX_PY_FILES_PER_SKILL", DEFAULT_MAX_PY_FILES_PER_SKILL)
    max_imports = _env_int("SMOKE_MAX_IMPORTS_PER_FILE", DEFAULT_MAX_IMPORTS_PER_FILE)
    import_timeout = _env_int("SMOKE_IMPORT_TIMEOUT_S", DEFAULT_IMPORT_TIMEOUT_S)

    rng = random.Random(seed)
    data = _read_json(REPORT_IN)
    rows: list[dict] = data.get("results") or []
    by_id = {r.get("skill_id") or r.get("skill_id".upper()) or r.get("skill_id".lower()): r for r in rows}
    # audit uses "skill_id"
    by_id = {r["skill_id"]: r for r in rows if "skill_id" in r}

    script_skills: list[str] = []
    example_skills: list[str] = []
    for r in rows:
        sid = r.get("skill_id")
        if not sid:
            continue
        if r.get("has_any_scripts"):
            script_skills.append(sid)
        if r.get("has_examples_dir"):
            example_skills.append(sid)

    sampled_scripts = _pick_sample(rng, sorted(set(script_skills)), n_scripts)
    sampled_examples = _pick_sample(rng, sorted(set(example_skills)), n_examples)
    sampled = list(dict.fromkeys(sampled_scripts + sampled_examples))

    rscript_available = shutil.which("Rscript") is not None
    bash_available = shutil.which("bash") is not None

    results: list[SkillSmoke] = []
    for sid in sampled:
        row = by_id.get(sid, {})
        py_files = _find_py_files(sid, row)[:max_py_files]
        sh_files = _find_sh_files(sid, row)[:2]
        r_files = _find_r_files(sid, row)[:2]

        missing_mods: dict[str, str] = {}
        py_checked: list[str] = []
        for p in py_files:
            py_checked.append(str(p).replace("\\", "/"))
            for mod in _extract_python_imports(p, max_imports=max_imports):
                if mod in missing_mods:
                    continue
                ok, err = _try_import(mod, timeout_s=import_timeout)
                if not ok:
                    missing_mods[mod] = err or "import failed"

        missing_cmds: list[str] = []
        sh_checked: list[str] = []
        if sh_files:
            for p in sh_files:
                sh_checked.append(str(p).replace("\\", "/"))
                for cmd in _extract_shell_commands(p):
                    if cmd in missing_cmds:
                        continue
                    if shutil.which(cmd) is None:
                        missing_cmds.append(cmd)
        if (not bash_available) and any(p.suffix.lower() == ".sh" for p in sh_files):
            if "bash" not in missing_cmds:
                missing_cmds.insert(0, "bash")

        r_checked: list[str] = [str(p).replace("\\", "/") for p in r_files]
        pkgs: list[str] = []
        for p in r_files:
            for pkg in _extract_r_packages(p):
                if pkg not in pkgs:
                    pkgs.append(pkg)

        results.append(
            SkillSmoke(
                skill_id=sid,
                python_files_checked=py_checked,
                python_missing_modules=missing_mods,
                shell_files_checked=sh_checked,
                shell_missing_commands=missing_cmds,
                r_files_checked=r_checked,
                rscript_available=rscript_available,
                r_packages=pkgs,
            )
        )

    summary = {
        "seed": seed,
        "sampled_total": len(results),
        "sampled_scripts_skills": len(sampled_scripts),
        "sampled_examples_skills": len(sampled_examples),
        "bash_available": bash_available,
        "rscript_available": rscript_available,
        "skills_with_missing_python_modules": sum(1 for r in results if r.python_missing_modules),
        "skills_with_missing_shell_commands": sum(1 for r in results if r.shell_missing_commands),
    }

    output = {
        "summary": summary,
        "sampled_skill_ids": sampled,
        "results": [asdict(r) for r in results],
    }
    out_path = Path("skill_runtime_smoke_report.json")
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

