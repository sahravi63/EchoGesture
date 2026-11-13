"""EchoGesture run 07 (v11): make every clip independent and re-measure.

v10 kept MediaPipe's tracking state between clips (only a timestamp jump separated them),
so a clip's landmarks - and sometimes its prediction - depended on the clip before it:
the batch run and the robustness run disagreed on 13 of 317 clips. v11 switches on
reset_per_clip (a fresh graph per clip) and re-runs, with the same models:
  1. predict() on the 317 official test videos -> predictions.jsonl/.json, top-1/top-5
  2. a second, reversed-order pass that must agree 100% (determinism check)
  3. latency (with landmarks), robustness (5 perturbations), results.csv
CPU only.  Inputs: run06 output (or run05 output) + risangbaskoro/wlasl-processed.
    !python run07.py
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--input-root", default="/kaggle/input")
ap.add_argument("--subset", type=int, default=300)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--device", default="Kaggle CPU")
ap.add_argument("--zip", default="/kaggle/working/echogesture_run07.zip")
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
cfg_path = "src/models/config.json"
cfg = json.load(open(cfg_path))
if not (cfg.get("gap_fill") and cfg.get("low_light") and len(cfg.get("models", [])) == 3):
    sys.exit("!! not the v9 models")
cfg["reset_per_clip"] = True
json.dump(cfg, open(cfg_path, "w"), indent=2)
print("config:", cfg, flush=True)

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
for name, order in (("test_manifest.csv", rows), ("test_manifest_reversed.csv", rows[::-1])):
    with open(f"results/{name}", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "path", "gloss"])
        for r in order:
            w.writerow([r["video_id"], f"{r['video_id']}.mp4", r["gloss"].upper()])
print(f"videos: {VID}  {len(rows)} test clips", flush=True)

sh(f"{PY} -m pip install -q mediapipe==1.2.2 ai-edge-litert==2.3.0 2>&1 | tail -2")
sh(f"{PY} src/tools/fetch_models.py")

# ---- 1 + 2. predictions, then the same clips in reverse order (must match exactly)
sh(f"{PY} src/main.py --videos '{VID}' --manifest results/test_manifest.csv --out results")
sh(f"{PY} src/main.py --videos '{VID}' --manifest results/test_manifest_reversed.csv --out results/reversed")
truth = {r["video_id"]: r["gloss"].upper() for r in rows}
preds = {json.loads(l)["clip_id"]: json.loads(l) for l in open("results/predictions.jsonl")}
rev = {json.loads(l)["clip_id"]: json.loads(l) for l in open("results/reversed/predictions.jsonl")}
n = len(preds)
top1 = sum(p["prediction"] == truth[k] for k, p in preds.items())
top5 = sum(truth[k] in p["top5"] for k, p in preds.items())
same = sum(preds[k]["prediction"] == rev[k]["prediction"] for k in preds)
print(f"\n== predict() on {n} official test videos: top-1 {100 * top1 / n:.2f}%  top-5 {100 * top5 / n:.2f}%  "
      f"missing ids: {len(set(truth) - set(preds))} ==", flush=True)
print(f"== determinism: forward vs reversed order agree on {same}/{n} clips ==", flush=True)
sys.path[:0] = ["src", "src/tools"]
from evaluate import update_measured  # noqa: E402
setup = f"{len(cfg['models'])} model(s) x TTA {cfg.get('tta', 1)}"
update_measured(top1_accuracy={
    "value": round(100 * top1 / n, 2), "device": a.device, "runtime": f"LiteRT INT8, {setup}, MediaPipe Holistic",
    "note": f"predict() end to end on {n} official WLASL300 test videos, top-5 {100 * top5 / n:.2f}%; "
            f"each clip independent (fresh MediaPipe graph); forward vs reversed order agree {same}/{n}"})

# ---- 3. latency, robustness, results.csv
clip = os.path.join(VID, f"{rows[0]['video_id']}.mp4")
sh(f"{PY} src/tools/bench_latency.py --device '{a.device}' --video '{clip}'")
sh(f"{PY} src/tools/robustness.py --videos '{VID}' --subset {a.subset} --split test "
   f"--workers {a.workers} --device '{a.device}'")
rob = json.load(open("results/robustness.json"))
agree = sum(preds[r["video_id"]]["prediction"] == r.get("clean") for r in rob["per_clip"])
print(f"== robustness clean pass agrees with the batch run on {agree}/{n} clips ==", flush=True)
sh(f"{PY} src/tools/make_results_csv.py")
sh("cat results/results.csv")
sh(f"rm -f {a.zip} && zip -qr {a.zip} results src/models/config.json -x 'results/*.mp4' && ls -lh {a.zip}")
print(f"done in {(time.time() - T0) / 60:.0f} min", flush=True)
