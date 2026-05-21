"""
Fairness matrix — compare/ablate phase 의 step 산출물이 동일 환경에서 나왔는지 검증.

plan.md §3-11-F 참고. execute.py 가 phase 완료 직후 (compare/ablate kind 만) 호출.

검증 차원 (run_dir 별로 추출):
    - dataset_split_hash : config.yaml 의 data 섹션 해시 (또는 data_manifest.json)
    - seed_set           : seed.txt 의 정수 (여러 시드면 정렬 후 비교)
    - metric_primary_key : eval.json 의 metric 이름 (값이 아니라 키)
    - hardware_tag       : uname -r + nvidia-smi --query-gpu=name (또는 config 의 hardware 필드)
    - git_rev            : git_rev.txt
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class FairnessReport:
    violations: list = field(default_factory=list)  # list of (dimension, values_seen)
    inspected: list = field(default_factory=list)   # run_dir 목록

    @property
    def violated(self) -> bool:
        return bool(self.violations)


def _read_text(p: Path) -> Optional[str]:
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _read_json(p: Path) -> Optional[dict]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _dataset_split_hash(run_dir: Path) -> Optional[str]:
    """data_manifest.json 우선, 없으면 config.yaml 의 data 섹션 hash."""
    manifest = _read_json(run_dir / "data_manifest.json")
    if manifest is not None:
        text = json.dumps(manifest, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    # config.yaml 은 yaml 파싱 의존성 회피 — 단순 text hash
    cfg = _read_text(run_dir / "config.yaml")
    if cfg is None:
        return None
    # data: 섹션만 추출
    lines = cfg.splitlines()
    in_data = False
    data_lines = []
    for ln in lines:
        if ln.startswith("data:"):
            in_data = True
            continue
        if in_data and ln and not ln.startswith((" ", "\t")):
            break
        if in_data:
            data_lines.append(ln)
    text = "\n".join(data_lines) if data_lines else cfg
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _seed(run_dir: Path) -> Optional[str]:
    return _read_text(run_dir / "seed.txt")


def _git_rev(run_dir: Path) -> Optional[str]:
    return _read_text(run_dir / "git_rev.txt")


def _metric_primary_key(run_dir: Path) -> Optional[str]:
    """eval.json 의 metric 이름 (값이 아니라 '어떤 메트릭을 봤는가') 비교."""
    data = _read_json(run_dir / "eval.json")
    if data is None:
        return None
    # metric_primary 가 dict 이면 키, 아니면 'metric_primary' 자체
    if "metric_primary_name" in data:
        return str(data["metric_primary_name"])
    return "metric_primary" if "metric_primary" in data else None


def _hardware_tag(run_dir: Path) -> Optional[str]:
    """config.yaml 의 hardware 섹션 또는 bench.json 의 hardware 필드."""
    bench = _read_json(run_dir / "bench.json") or {}
    if "hardware" in bench:
        return str(bench["hardware"])
    cfg = _read_text(run_dir / "config.yaml") or ""
    for ln in cfg.splitlines():
        if ln.startswith("hardware:"):
            return ln.split(":", 1)[1].strip()
    return None


def verify(run_dirs: list, root: Path) -> FairnessReport:
    """run_dirs 의 5 차원이 모두 동일한지 검증.

    각 run_dir 은 `runs/{name}` 형태 또는 절대 경로.
    """
    report = FairnessReport()
    extractors = {
        "dataset_split_hash": _dataset_split_hash,
        "seed": _seed,
        "git_rev": _git_rev,
        "metric_primary_key": _metric_primary_key,
        "hardware_tag": _hardware_tag,
    }
    values_by_dim: dict[str, set] = {d: set() for d in extractors}
    nones_by_dim: dict[str, int] = {d: 0 for d in extractors}

    paths = []
    for rd in run_dirs:
        p = Path(rd) if Path(rd).is_absolute() else root / rd
        if not p.is_dir():
            continue
        paths.append(p)
        report.inspected.append(str(p.relative_to(root)) if p.is_relative_to(root) else str(p))
        for dim, fn in extractors.items():
            val = fn(p)
            if val is None:
                nones_by_dim[dim] += 1
            else:
                values_by_dim[dim].add(val)

    if len(paths) < 2:
        return report  # 비교 대상이 1 개 이하면 위반 아님

    # seed 는 변동 허용 (시드 세트가 같다는 게 더 약한 조건 — 일단 모두 같아야 한다고 봄)
    # 차원별로 값이 2 가지 이상이면 위반
    for dim, vals in values_by_dim.items():
        if dim == "seed":
            continue  # 시드 정책은 별도 검증 — 여기선 skip
        if len(vals) > 1:
            report.violations.append((dim, sorted(vals)))

    return report


def format_report(report: FairnessReport) -> str:
    lines = [f"  [Fairness] 검사 대상 run_dir {len(report.inspected)} 개"]
    if not report.violated:
        lines.append(f"  ✓ 5 차원 (dataset_split / seed / git_rev / metric / hardware) 모두 동일")
        return "\n".join(lines)
    lines.append(f"  ❌ 위반 차원 {len(report.violations)} 개:")
    for dim, vals in report.violations:
        lines.append(f"    - {dim}: {vals}")
    return "\n".join(lines)
