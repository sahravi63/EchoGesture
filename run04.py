"""EchoGesture run 04: robustness of the shipped ensemble, end to end on real videos.

Needs: the run03 models in src/models, the WLASL videos somewhere under --input-root,
MediaPipe + its .task bundles (src/tools/fetch_models.py). CPU only.
    !python run04.py
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--input-root", default="/kaggle/input")
ap.add_argument("--subset", type=int, default=300)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--device", default="Kaggle CPU")
ap.add_argument("--zip", default="/kaggle/working/echogesture_run04.zip")
a = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
PY = sys.executable


def sh(cmd):
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    if r.returncode:
        print(f"!! command failed with exit code {r.returncode}", flush=True)
    return r.returncode


cfg = json.load(open("src/models/config.json"))
missing = [m for m in cfg.get("models", []) if not os.path.exists(f"src/models/{m}")]
if not cfg.get("models") or missing:
    sys.exit(f"!! run03 models not in src/models (config models: {cfg.get('models')}, missing: {missing})")
print("config:", cfg, flush=True)

# find the folder that holds the WLASL test videos
rows = [r for r in csv.DictReader(open("src/data/wlasl/wlasl_manifest.csv", newline=""))
        if int(r["gloss_rank"]) < a.subset and r["split"] == "test"]
VID = None
for r in rows[:20]:
    hits = sorted(glob.glob(os.path.join(a.input_root, "**", f"{r['video_id']}.mp4"), recursive=True))
    if hits:
        VID = os.path.dirname(hits[0])
        break
if not VID:
    sys.exit(f"!! no WLASL test videos under {a.input_root} - is the wlasl-processed dataset attached?")
have = sum(os.path.exists(os.path.join(VID, f"{r['video_id']}.mp4")) for r in rows)
print(f"videos: {VID}  ({have}/{len(rows)} test clips of the {a.subset}-gloss subset present)", flush=True)

rc = sh(f"{PY} src/tools/robustness.py --videos '{VID}' --subset {a.subset} --limit {a.limit} "
        f"--workers {a.workers} --device '{a.device}'")
sh(f"{PY} src/tools/make_results_csv.py")
sh("cat results/results.csv")
sh(f"zip -qr {a.zip} src/models/*.tflite src/models/*.json results && ls -lh {a.zip}")
sys.exit(rc)
