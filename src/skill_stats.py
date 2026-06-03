"""Skill-level execution statistics tracker.

Tracks per-skill: times retrieved, selected, execution success/fail/timeout,
and LLM self-correction attempts after failures.
"""
import json
import os
from collections import defaultdict

STATS_PATH = "skill_stats.json"


def _default_entry() -> dict:
    return {
        "times_retrieved": 0,
        "times_selected": 0,
        "exec_success": 0,
        "exec_fail": 0,
        "exec_timeout": 0,
        "exec_api_fail": 0,
        "exec_api_auth_fail": 0,
        "exec_api_rate_limit": 0,
        "exec_api_network_error": 0,
        "exec_pkg_install_fail": 0,
        "correction_attempts": 0,
    }


class SkillStatsTracker:
    def __init__(self, path: str = STATS_PATH):
        self.path = path
        self.stats: dict[str, dict] = defaultdict(_default_entry)
        self._load()

    # ── recording ──

    def record_retrieval(self, skill_ids: list[str]):
        for sid in skill_ids:
            self.stats[sid]["times_retrieved"] += 1

    def record_selection(self, skill_id: str):
        if skill_id:
            self.stats[skill_id]["times_selected"] += 1

    def record_execution(self, skill_id: str, tool_status: str, is_retry: bool = False):
        if not skill_id:
            return
        entry = self.stats[skill_id]
        if tool_status == "成功":
            entry["exec_success"] += 1
        elif tool_status == "超时":
            entry["exec_timeout"] += 1
        elif tool_status in ("外部 API 认证失败", "外部 API 认证失败"):
            entry["exec_api_fail"] += 1
            entry["exec_api_auth_fail"] += 1
        elif tool_status == "外部 API 限流":
            entry["exec_api_fail"] += 1
            entry["exec_api_rate_limit"] += 1
        elif tool_status in ("外部 API 网络错误", "外部 API 网络错误"):
            entry["exec_api_fail"] += 1
            entry["exec_api_network_error"] += 1
        elif tool_status == "包安装失败":
            entry["exec_fail"] += 1
            entry["exec_pkg_install_fail"] += 1
        elif "外部 API" in tool_status:
            entry["exec_api_fail"] += 1
        elif tool_status == "失败":
            entry["exec_fail"] += 1
        if is_retry:
            entry["correction_attempts"] += 1

    # ── persistence ──

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._to_serializable(), f, ensure_ascii=False, indent=2)

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for sid, entry in data.items():
                self.stats[sid] = {**_default_entry(), **entry}
        except Exception:
            pass

    def _to_serializable(self) -> dict:
        return dict(self.stats)

    # ── reporting ──

    def never_retrieved(self, all_skill_ids: set) -> list:
        return sorted(all_skill_ids - set(self.stats.keys()))

    def never_selected(self) -> list:
        return sorted(
            sid for sid, e in self.stats.items()
            if e["times_retrieved"] > 0 and e["times_selected"] == 0
        )

    def top_failures(self, n: int = 10) -> list[tuple[str, int]]:
        ranked = [(sid, e["exec_fail"] + e["exec_timeout"] + e["exec_api_fail"])
                  for sid, e in self.stats.items()]
        ranked.sort(key=lambda x: -x[1])
        return ranked[:n]

    def report(self, all_skill_ids: set | None = None) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("Skill Stats Report")
        lines.append("=" * 60)

        total_retrieved = sum(e["times_retrieved"] for e in self.stats.values())
        total_selected = sum(e["times_selected"] for e in self.stats.values())
        total_success = sum(e["exec_success"] for e in self.stats.values())
        total_fail = sum(e["exec_fail"] for e in self.stats.values())
        total_timeout = sum(e["exec_timeout"] for e in self.stats.values())
        total_api_fail = sum(e["exec_api_fail"] for e in self.stats.values())
        total_api_auth = sum(e["exec_api_auth_fail"] for e in self.stats.values())
        total_api_rate = sum(e["exec_api_rate_limit"] for e in self.stats.values())
        total_api_net = sum(e["exec_api_network_error"] for e in self.stats.values())
        total_pkg_fail = sum(e["exec_pkg_install_fail"] for e in self.stats.values())
        total_corrections = sum(e["correction_attempts"] for e in self.stats.values())

        lines.append(f"Skills tracked: {len(self.stats)}")
        lines.append(f"Total retrievals: {total_retrieved}")
        lines.append(f"Total selections: {total_selected}")
        lines.append(f"Exec success: {total_success}")
        lines.append(f"Exec fail: {total_fail}")
        lines.append(f"Exec timeout: {total_timeout}")
        lines.append(f"External API fail (total): {total_api_fail}")
        lines.append(f"  - Auth fail: {total_api_auth}")
        lines.append(f"  - Rate limit: {total_api_rate}")
        lines.append(f"  - Network error: {total_api_net}")
        lines.append(f"Package install fail: {total_pkg_fail}")
        lines.append(f"Correction attempts: {total_corrections}")
        lines.append("")

        never_sel = self.never_selected()
        if never_sel:
            lines.append(f"Skills retrieved but never selected ({len(never_sel)}):")
            for sid in never_sel[:15]:
                lines.append(f"  - {sid}")
            if len(never_sel) > 15:
                lines.append(f"  ... and {len(never_sel) - 15} more")
            lines.append("")

        top_fail = self.top_failures(10)
        if top_fail:
            lines.append("Top failures:")
            for sid, count in top_fail:
                if count > 0:
                    lines.append(f"  - {sid}: {count} failures")
            lines.append("")

        if all_skill_ids:
            never_ret = self.never_retrieved(all_skill_ids)
            lines.append(f"Skills never retrieved ({len(never_ret)}):")
            for sid in never_ret[:15]:
                lines.append(f"  - {sid}")
            if len(never_ret) > 15:
                lines.append(f"  ... and {len(never_ret) - 15} more")

        lines.append("=" * 60)
        return "\n".join(lines)

    def export_csv(self) -> str:
        header = "skill_id,retrieved,selected,success,fail,timeout,api_fail,api_auth,api_rate,api_net,pkg_fail,corrections"
        rows = [header]
        for sid in sorted(self.stats):
            e = self.stats[sid]
            rows.append(
                f"{sid},{e['times_retrieved']},{e['times_selected']},"
                f"{e['exec_success']},{e['exec_fail']},{e['exec_timeout']},"
                f"{e['exec_api_fail']},{e['exec_api_auth_fail']},{e['exec_api_rate_limit']},"
                f"{e['exec_api_network_error']},{e['exec_pkg_install_fail']},{e['correction_attempts']}"
            )
        return "\n".join(rows)
