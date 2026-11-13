"""EchoGesture run 06: final deliverables from the shipped v9 models (CPU, no training).

  1. predict() on every official WLASL300 test video -> results/predictions.jsonl + .json
     (exactly what the organisers' evaluator does), end-to-end top-1 / top-5
  2. latency including landmarks on a real clip, size, results.csv
  3. the demo video (random test clips + robustness segment, caption HUD)
  4. one zip with the complete repository contents for GitHub

Inputs on Kaggle: the run05 output (code + v9 models), risangbaskoro/wlasl-processed.
    !python run06.py
"""
import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--input-root", default="/kaggle/input")
ap.add_argument("--subset", type=int, default=300)
ap.add_argument("--device", default="Kaggle CPU")
ap.add_argument("--zip", default="/kaggle/working/echogesture_final.zip")
ap.add_argument("--skip-demo", action="store_true")
a = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
PY = os.environ.get("EG_PY", sys.executable)        # EG_PY: test hook only
T0 = time.time()


def sh(cmd):
    print(f"$ {cmd}   [{(time.time() - T0) / 60:.0f} min]", flush=True)
    r = subprocess.run(cmd, shell=True)
    if r.returncode:
        print(f"!! command failed with exit code {r.returncode}", flush=True)
    return r.returncode


print("code version:", open("src/VERSION").read().strip(), flush=True)
cfg = json.load(open("src/models/config.json"))
print("config:", cfg, flush=True)
if not (cfg.get("gap_fill") and cfg.get("low_light") and len(cfg.get("models", [])) == 3):
    sys.exit("!! these are not the v9 models (config lacks gap_fill/low_light or 3 models) - wrong input version?")
for m in cfg["models"]:
    assert os.path.exists(f"src/models/{m}"), m

# ---- test videos + manifest the evaluator-style batch run reads
rows = [r for r in csv.DictReader(open("src/data/wlasl/wlasl_manifest.csv", newline=""))
        if int(r["gloss_rank"]) < a.subset and r["split"] == "test"]
VID = None
for r in rows[:20]:
    hits = sorted(glob.glob(os.path.join(a.input_root, "**", f"{r['video_id']}.mp4"), recursive=True))
    if hits:
        VID = os.path.dirname(hits[0])
        break
if not VID:
    sys.exit("!! attach risangbaskoro/wlasl-processed")
rows = [r for r in rows if os.path.exists(os.path.join(VID, f"{r['video_id']}.mp4"))]
os.makedirs("results", exist_ok=True)
with open("results/test_manifest.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["clip_id", "path", "gloss"])
    for r in rows:
        w.writerow([r["video_id"], f"{r['video_id']}.mp4", r["gloss"].upper()])
print(f"videos: {VID}  {len(rows)} test clips", flush=True)

sh(f"{PY} -m pip install -q mediapipe==1.2.2 ai-edge-litert==2.3.0 2>&1 | tail -2")
sh(f"{PY} src/tools/fetch_models.py")

# ---- 1. predictions exactly as the evaluator produces them
sh(f"{PY} src/main.py --videos '{VID}' --manifest results/test_manifest.csv --out results")
truth = {r["video_id"]: r["gloss"].upper() for r in rows}
preds = [json.loads(l) for l in open("results/predictions.jsonl")]
n = len(preds)
top1 = sum(p["prediction"] == truth[p["clip_id"]] for p in preds)
top5 = sum(truth[p["clip_id"]] in p["top5"] for p in preds)
missing = set(truth) - {p["clip_id"] for p in preds}
print(f"\n== predict() on {n} official test videos: top-1 {100 * top1 / n:.2f}%  top-5 {100 * top5 / n:.2f}%"
      f"  missing ids: {len(missing)} ==", flush=True)
timings = json.load(open("results/timings.json"))
lm = sorted(t["landmark_ms"] / max(t["frames"], 1) for t in timings)
mm = sorted(t["model_ms"] for t in timings)
pct = lambda v, q: v[min(len(v) - 1, int(q * len(v)))]
print(f"per clip: landmarks p50 {pct(lm, .5):.1f} ms/frame, model p50 {pct(mm, .5):.1f} ms p95 {pct(mm, .95):.1f} ms",
      flush=True)
sys.path[:0] = ["src", "src/tools"]
from evaluate import update_measured  # noqa: E402
packed = json.load(open("results/measured.json")).get("top1_accuracy", {})
setup = f"{len(cfg['models'])} model(s) x TTA {cfg.get('tta', 1)}"
update_measured(top1_accuracy={
    "value": round(100 * top1 / n, 2), "device": a.device, "runtime": f"LiteRT INT8, {setup}, MediaPipe Holistic",
    "note": f"predict() end to end on {n} official WLASL300 test videos, top-5 {100 * top5 / n:.2f}%; "
            f"on pre-extracted landmarks {packed.get('value')}%"})

# ---- 2. latency with landmarks on a real clip, then results.csv
clip = os.path.join(VID, f"{rows[0]['video_id']}.mp4")
sh(f"{PY} src/tools/bench_latency.py --device '{a.device}' --video '{clip}'")
sh(f"{PY} src/tools/make_results_csv.py")
sh("cat results/results.csv")

# ---- 3. demo video
if not a.skip_demo:
    sh(f"{PY} src/demo/render_demo.py --videos '{VID}' --out results/demo.mp4")
    if os.path.exists("results/demo.mp4"):
        shutil.copy("results/demo.mp4", os.path.join(os.path.dirname(a.zip), "echogesture_demo.mp4"))

# ---- 4. the repository as it should go to GitHub (MediaPipe .task bundles included so it runs offline)
R = os.path.basename(HERE)
sh(f"cd .. && rm -f {a.zip} && zip -qr {a.zip} {R} -x '{R}/runs/*' "
   f"'{R}/init_from_zip/*' '*/__pycache__/*' '{R}/results/demo.mp4' && ls -lh {a.zip}")
print(f"done in {(time.time() - T0) / 60:.0f} min", flush=True)
