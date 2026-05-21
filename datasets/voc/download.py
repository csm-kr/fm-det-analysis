"""PASCAL VOC 2007 + 2012 다운로드 진입점.

공식 tar (VOCdevkit 구조) — VOC2007 trainval/test + VOC2012 trainval.
무결성 검증 후 runs/data-sanity-voc-download-{ts}/manifest.json 작성.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

VOC_URLS = [
    ("VOCtrainval_06-Nov-2007.tar", "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"),
    ("VOCtest_06-Nov-2007.tar", "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar"),
    ("VOCtrainval_11-May-2012.tar", "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"),
]

EXPECTED = {
    "voc07-trainval": 5011,
    "voc07-test": 4952,
    "voc12-trainval": 11540,
}


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M")


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _wget(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["wget", "-c", "--tries=3", "--timeout=120",
           "--progress=dot:giga", "-O", str(dest), url]
    print(f"  wget → {dest.name}", flush=True)
    subprocess.run(cmd, check=True)


def _untar(tar_path: Path, dest_dir: Path) -> bool:
    """tarfile 로 압축 해제. members 검증 통과 시 True."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r") as tf:
        try:
            tf.extractall(dest_dir)
        except Exception as e:
            print(f"  tar extract FAIL: {e}", file=sys.stderr)
            return False
    return True


def _count_split(voc_root: Path, year: str, split: str) -> int:
    f = voc_root / f"VOC{year}" / "ImageSets" / "Main" / f"{split}.txt"
    if not f.exists():
        return 0
    return sum(1 for line in f.read_text(encoding="utf-8").splitlines() if line.strip())


def download(data_root: Path, run_dir: Path, seed: int) -> dict:
    voc_root = data_root / "voc"
    tars_dir = voc_root / "_tars"
    voc_root.mkdir(parents=True, exist_ok=True)
    tars_dir.mkdir(parents=True, exist_ok=True)

    file_records = []
    error_reasons: list[str] = []

    for fname, url in VOC_URLS:
        tar_path = tars_dir / fname
        _wget(url, tar_path)
        sha = _sha256(tar_path)
        size_b = tar_path.stat().st_size
        untar_ok = _untar(tar_path, voc_root)
        if not untar_ok:
            error_reasons.append(f"{fname}: tar extract fail")
        file_records.append({
            "name": fname, "url": url, "size_bytes": size_b,
            "sha256": sha, "untar_ok": untar_ok,
        })

    # 압축 해제 위치: voc_root / "VOCdevkit" / "VOC{2007,2012}"
    devkit = voc_root / "VOCdevkit"
    counts = {
        "voc07-trainval": _count_split(devkit, "2007", "trainval"),
        "voc07-test": _count_split(devkit, "2007", "test"),
        "voc12-trainval": _count_split(devkit, "2012", "trainval"),
    }
    counts["voc-trainval"] = counts["voc07-trainval"] + counts["voc12-trainval"]

    integrity_ok = all(r["untar_ok"] for r in file_records)
    for k, expect in EXPECTED.items():
        if counts[k] != expect:
            integrity_ok = False
            error_reasons.append(f"{k}: got {counts[k]} expected {expect}")

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(ROOT))
        except ValueError:
            return str(p)

    manifest = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "target": "voc",
        "data_root": _rel(data_root),
        "seed": seed,
        "files": file_records,
        "extracted": {"devkit": _rel(devkit)},
        "image_counts": counts,
        "image_counts_expected": EXPECTED | {"voc-trainval": EXPECTED["voc07-trainval"] + EXPECTED["voc12-trainval"]},
        "integrity_ok": integrity_ok,
    }
    if error_reasons:
        manifest["error_reasons"] = error_reasons

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="Download PASCAL VOC 2007 + 2012.")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    run_dir = ROOT / "runs" / f"data-sanity-voc-download-{_now_tag()}"
    manifest = download(args.data_root, run_dir, args.seed)
    print(f"\nmanifest → {run_dir / 'manifest.json'}")
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))
    return 0 if manifest["integrity_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
