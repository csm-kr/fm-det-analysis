"""
Experiment budget — phases/index.json 의 experiment_budget 을 검사.

plan.md §3-14 참고. phase 진입 직전에 execute.py 가 호출한다.

experiment_budget 스키마 (top-level phases/index.json):
    {
      "experiment_budget": {
        "datasets": N, "algorithms": M,
        "ablation_dims": [{"name", "values": [...]}],
        "seeds": K,
        "per_run_hours": h, "per_run_disk_gb": GB,
        "total_gpu_hours_estimate": ...,    # 자동 계산 (또는 수기 명시)
        "gpu_hours_threshold": H,           # 초과 시 사용자 confirm prompt
        "disk_gb_threshold": D
      }
    }
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class BudgetReport:
    total_runs: int
    total_gpu_hours: float
    total_disk_gb: float
    gpu_threshold: Optional[float]
    disk_threshold: Optional[float]
    over_gpu: bool
    over_disk: bool

    @property
    def any_over(self) -> bool:
        return self.over_gpu or self.over_disk


def compute_total_runs(budget: dict) -> int:
    """총 학습 횟수 = datasets × algorithms × ∏(values 길이) × seeds."""
    datasets = int(budget.get("datasets", 1))
    algorithms = int(budget.get("algorithms", 1))
    seeds = int(budget.get("seeds", 1))
    dims = budget.get("ablation_dims") or []
    factor = 1
    for d in dims:
        values = d.get("values") or []
        if values:
            factor *= max(1, len(values))
    return datasets * algorithms * factor * seeds


def compute_budget(budget: dict) -> BudgetReport:
    runs = compute_total_runs(budget)
    per_h = float(budget.get("per_run_hours", 0))
    per_d = float(budget.get("per_run_disk_gb", 0))
    total_h = runs * per_h
    total_d = runs * per_d

    gpu_th = budget.get("gpu_hours_threshold")
    disk_th = budget.get("disk_gb_threshold")
    over_gpu = gpu_th is not None and total_h > float(gpu_th)
    over_disk = disk_th is not None and total_d > float(disk_th)
    return BudgetReport(
        total_runs=runs,
        total_gpu_hours=total_h,
        total_disk_gb=total_d,
        gpu_threshold=float(gpu_th) if gpu_th is not None else None,
        disk_threshold=float(disk_th) if disk_th is not None else None,
        over_gpu=over_gpu,
        over_disk=over_disk,
    )


def load_top_budget(top_index_file: Path) -> Optional[dict]:
    """phases/index.json 에서 experiment_budget dict 를 반환 (없으면 None)."""
    if not top_index_file.exists():
        return None
    try:
        data = json.loads(top_index_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data.get("experiment_budget")


def format_report(report: BudgetReport) -> str:
    lines = [
        f"  총 학습 횟수: {report.total_runs}",
        f"  총 GPU-시간 예상: {report.total_gpu_hours:.1f}h" + (
            f"  (임계 {report.gpu_threshold:.1f}h {'❌ 초과' if report.over_gpu else '✓'})"
            if report.gpu_threshold is not None else ""
        ),
        f"  총 디스크 예상: {report.total_disk_gb:.1f}GB" + (
            f"  (임계 {report.disk_threshold:.1f}GB {'❌ 초과' if report.over_disk else '✓'})"
            if report.disk_threshold is not None else ""
        ),
    ]
    return "\n".join(lines)
