"""COCO 2017 다운로드 진입점.

scripts/ 는 하네스 정본 전용 (CLAUDE.md) — 데이터 다운로드 진입점은 루트 평탄.
val + annotations 만 받는다. train2017 (18GB) 은 별도 phase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

COCO_URLS = {
    "val": [
        ("val2017.zip", "http://images.cocodataset.org/zips/val2017.zip"),
        (
            "annotations_trainval2017.zip",
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        ),
    ],
    "train": [
        ("train2017.zip", "http://images.cocodataset.org/zips/train2017.zip"),
        (
            "annotations_trainval2017.zip",
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        ),
    ],
}

EXPECTED_IMG_COUNT = {"val": 5000, "train": 118287}


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
    cmd = [
        "wget", "-c", "--tries=3", "--timeout=60",
        "--progress=dot:giga",
        "-O", str(dest), url,
    ]
    print(f"  wget → {dest.name}", flush=True)
    subprocess.run(cmd, check=True)


def _unzip(zip_path: Path, dest_dir: Path) -> bool:
    """zipfile (stdlib) 으로 무결성 검증 + 압축 해제. 통과 시 True."""
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad is not None:
            print(f"  CRC FAIL: {bad}", file=sys.stderr)
            return False
        dest_dir.mkdir(parents=True, exist_ok=True)
        zf.extractall(dest_dir)
    return True


def download(target: str, split: str, data_root: Path, run_dir: Path, seed: int) -> dict:
    assert target == "coco", f"unsupported target: {target}"
    assert split in COCO_URLS, f"unknown split: {split}"

    coco_root = data_root / "coco"
    zips_dir = coco_root / "_zips"
    images_dir = coco_root / f"{split}2017"
    annotations_dir = coco_root / "annotations"

    coco_root.mkdir(parents=True, exist_ok=True)
    zips_dir.mkdir(parents=True, exist_ok=True)

    file_records = []
    error_reasons: list[str] = []

    for fname, url in COCO_URLS[split]:
        zip_path = zips_dir / fname
        _wget(url, zip_path)
        sha = _sha256(zip_path)
        size_b = zip_path.stat().st_size
        unzip_ok = _unzip(zip_path, coco_root)
        if not unzip_ok:
            error_reasons.append(f"{fname}: CRC fail")
        file_records.append({
            "name": fname,
            "url": url,
            "size_bytes": size_b,
            "sha256": sha,
            "unzip_ok": unzip_ok,
        })

    n_imgs = len(list(images_dir.glob("*.jpg"))) if images_dir.exists() else 0
    n_ann = len(list(annotations_dir.glob("*.json"))) if annotations_dir.exists() else 0
    instances_target = annotations_dir / f"instances_{split}2017.json"
    n_imgs_expected = EXPECTED_IMG_COUNT[split]

    integrity_ok = (
        n_imgs == n_imgs_expected
        and n_ann >= 4
        and instances_target.exists()
        and all(r["unzip_ok"] for r in file_records)
    )
    if n_imgs != n_imgs_expected:
        error_reasons.append(f"image_count != {n_imgs_expected} (got {n_imgs})")
    if n_ann < 4:
        error_reasons.append(f"ann_files < 4 (got {n_ann})")
    if not instances_target.exists():
        error_reasons.append(f"instances_{split}2017.json missing")

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(ROOT))
        except ValueError:
            return str(p)

    manifest = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "target": target,
        "split": split,
        "data_root": _rel(data_root),
        "seed": seed,
        "files": file_records,
        "extracted": {
            "images_dir": _rel(images_dir),
            "annotations_dir": _rel(annotations_dir),
        },
        "image_count": n_imgs,
        "image_count_expected": n_imgs_expected,
        f"{split}_images": n_imgs,
        "ann_files": n_ann,
        "target_annotation": f"instances_{split}2017.json",
        "target_annotation_exists": instances_target.exists(),
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
    ap = argparse.ArgumentParser(description="Download COCO 2017 (val/train + annotations).")
    ap.add_argument("--target", default="coco", choices=["coco"])
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    run_dir = ROOT / "runs" / f"data-sanity-download-{_now_tag()}"
    manifest = download(args.target, args.split, args.data_root, run_dir, args.seed)
    print(f"\nmanifest → {run_dir / 'manifest.json'}")
    summary = {k: v for k, v in manifest.items() if k != "files"}
    print(json.dumps(summary, indent=2))
    return 0 if manifest["integrity_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
