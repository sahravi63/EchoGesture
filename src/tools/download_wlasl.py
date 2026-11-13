"""Get WLASL videos.

Option A (recommended, ~12k pre-trimmed clips named <video_id>.mp4):
    pip install kaggle   # put kaggle.json in ~/.kaggle (Windows: C:\\Users\\<you>\\.kaggle)
    kaggle datasets download -d risangbaskoro/wlasl-processed -p data/raw --unzip
    -> data/raw/videos/<video_id>.mp4   (then extract with --trimmed)

Option B (fallback, original sources; many links are dead, YouTube needs yt-dlp):
    pip install yt-dlp requests
    python src/tools/download_wlasl.py --subset 300 --out data/raw_orig
    -> data/raw_orig/<video_id>.mp4 (full source video; extract WITHOUT --trimmed so the
       WLASL frame_start/frame_end range is applied)

Either way, coverage is reported against src/data/wlasl/wlasl_manifest.csv.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MANIFEST = Path(__file__).resolve().parent.parent / "data" / "wlasl" / "wlasl_manifest.csv"


def rows(subset: int):
    with open(MANIFEST, newline="") as f:
        return [r for r in csv.DictReader(f) if int(r["gloss_rank"]) < subset]


def fetch(r, out: Path) -> str:
    dst = out / f"{r['video_id']}.mp4"
    if dst.exists() and dst.stat().st_size > 1000:
        return "have"
    url = r["url"]
    try:
        if "youtube" in url or "youtu.be" in url:
            cmd = ["yt-dlp", "-q", "-f", "mp4/bestvideo[ext=mp4]/best", "-o", str(dst), url]
            subprocess.run(cmd, check=True, timeout=120, capture_output=True)
        else:
            import requests
            h = {"User-Agent": "Mozilla/5.0", "Referer": "http://www.aslpro.com/cgi-bin/aslpro/aslpro.cgi"}
            resp = requests.get(url, headers=h, timeout=30)
            resp.raise_for_status()
            if "swf" in url:
                return "skip-swf"
            dst.write_bytes(resp.content)
        return "ok" if dst.exists() and dst.stat().st_size > 1000 else "empty"
    except Exception as e:
        return f"fail:{type(e).__name__}"


def coverage(video_dir: Path, subset: int):
    rs = rows(subset)
    have = {p.stem for p in video_dir.glob("*.mp4")}
    by_split = {}
    for r in rs:
        s = by_split.setdefault(r["split"], [0, 0]); s[1] += 1; s[0] += r["video_id"] in have
    tot = sum(v[0] for v in by_split.values())
    print(f"WLASL{subset}: {tot}/{len(rs)} clips present in {video_dir}")
    for k, (h, n) in sorted(by_split.items()):
        print(f"  {k:5s} {h}/{n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=300, help="100/300/1000/2000")
    ap.add_argument("--out", default="data/raw_orig")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--coverage-only", help="just report coverage for this video folder")
    a = ap.parse_args()
    if a.coverage_only:
        coverage(Path(a.coverage_only), a.subset); return
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rs = rows(a.subset)
    with ThreadPoolExecutor(a.threads) as ex:
        res = list(ex.map(lambda r: fetch(r, out), rs))
    from collections import Counter
    print(Counter(x.split(":")[0] for x in res))
    coverage(out, a.subset)


if __name__ == "__main__":
    sys.exit(main())
