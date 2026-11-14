"""EchoGesture run 09 (v13): train on landmarks from REALLY occluded, blurred and darkened videos.

v12 (run08) cut the occlusion drop from 22.4 to 13.9 pt by running MediaPipe on covered
train/val videos; the mean drop is 5.87 pt (target 5). Blur (7.9 pt) and low light (4.4 pt)
are now the other big losses, so v13 adds a second degraded copy of every train/val clip:
motion blur (5-15 px, random angle, 40-100% of the clip) or low light (x0.25-0.6, gamma
1.3-2.2, noise), and swaps a training clip for its occluded copy with p = 0.25 and for its
blurred/dark copy with p = 0.20. Everything else is the v9/v12 recipe.
Test videos are only used at the end, for the same measurements as run07:
  packed clean top-1, predict() top-1/top-5 + determinism, latency, robustness, results.csv.

Inputs on Kaggle: echogesture code (v9 output), 01_extract output (wlasl300.npz),
wl2000 init (Version-05 output or echogesture_trained.zip), risangbaskoro/wlasl-processed.
GPU T4 x2, Internet ON.
    !python run08.py
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
ap.add_argument("--seeds", default="1,2,3")
ap.add_argument("--gpus", default="0,1,0")
ap.add_argument("--epochs", type=int, default=120)
ap.add_argument("--p-alt", default="0.25,0.20", help="occluded, blurred/dark")
ap.add_argument("--tta", type=int, default=2)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--device", default="Kaggle CPU")
ap.add_argument("--limit", type=int, default=0, help="test hook: only N clips per stage")
ap.add_argument("--zip", default="/kaggle/working/echogesture_run09.zip")
a = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
PY = os.environ.get("EG_PY", sys.executable)        # EG_PY: test hook only
MPENV = os.environ.get("EG_MPENV", "/tmp/eg_mpenv")   # MediaPipe kept away from TensorFlow (not saved as output)
T0 = time.time()


def sh(cmd):
    print(f"$ {cmd}   [{(time.time() - T0) / 60:.0f} min]", flush=True)
    r = subprocess.run(cmd, shell=True)
    if r.returncode:
        print(f"!! command failed with exit code {r.returncode}", flush=True)
    return r.returncode


def find(pattern):
    hits = sorted(glob.glob(os.path.join(a.input_root, "**", pattern), recursive=True))
    if not hits:
        sys.exit(f"!! could not find {pattern} under {a.input_root} - is that input attached?")
    return hits[0]


def bundle():
    sh(f"rm -f {a.zip} && zip -qr {a.zip} src/models/*.tflite src/models/*.json results runs/*.log "
       f"runs/ft300tv_s*/history.csv runs/occ_lm/_covers.json runs/deg_lm/_covers.json -x 'results/*.mp4' && ls -lh {a.zip}")


print("code version:", open("src/VERSION").read().strip(), flush=True)
cfg_path = "src/models/config.json"
cfg = json.load(open(cfg_path))
if not (cfg.get("gap_fill") and cfg.get("low_light")):
    sys.exit("!! expected the v9 config (gap_fill + low_light)")
cfg["reset_per_clip"] = True                        # v11 behaviour
json.dump(cfg, open(cfg_path, "w"), indent=2)

D = os.path.dirname(find("wlasl300.npz"))
_init = sorted(glob.glob(os.path.join(a.input_root, "**", "runs/wl2000/best.keras"), recursive=True))
if _init:
    INIT = _init[0]
else:
    import zipfile
    zipfile.ZipFile(find("echogesture_trained.zip")).extract("runs/wl2000/best.keras", "init_from_zip")
    INIT = "init_from_zip/runs/wl2000/best.keras"
rows_all = [r for r in csv.DictReader(open("src/data/wlasl/wlasl_manifest.csv", newline=""))
            if int(r["gloss_rank"]) < 300]
VID = None
for r in rows_all[:50]:
    hits = sorted(glob.glob(os.path.join(a.input_root, "**", f"{r['video_id']}.mp4"), recursive=True))
    if hits:
        VID = os.path.dirname(hits[0])
        break
if not VID:
    sys.exit("!! attach risangbaskoro/wlasl-processed")
print(f"data: {D}\ninit: {INIT}\nvideos: {VID}", flush=True)
os.makedirs("runs", exist_ok=True)
lim = f" --limit {a.limit}" if a.limit else ""

# ---- 1. occluded TRAIN/VAL videos -> landmarks (MediaPipe in its own directory)
if not os.path.exists(f"{MPENV}/mediapipe"):
    sh(f"{PY} -m pip install -q --target {MPENV} mediapipe==1.2.2 2>&1 | tail -2")
MP = f"PYTHONPATH={MPENV} {PY}"
sh(f"{MP} src/tools/fetch_models.py")
if sh(f"{MP} src/tools/extract_occluded.py --videos '{VID}' --modes occ,deg --workers {a.workers}{lim}") \
        or not all(os.path.exists(f"runs/wlasl300_{m}.npz") for m in ("occ", "deg")):
    sys.exit("!! degraded landmark extraction failed")
import numpy as np  # noqa: E402
_b = np.load(f"{D}/wlasl300.npz")
_ids = set(map(str, _b["ids"][np.isin(_b["split"], ["train", "val"])]))
for _m in ("occ", "deg"):
    _occ = set(map(str, np.load(f"runs/wlasl300_{_m}.npz")["ids"]))
    print(f"{_m} copies match {len(_ids & _occ)}/{len(_ids)} train+val clips", flush=True)
    if not a.limit and len(_ids & _occ) < 0.5 * len(_ids):
        sys.exit(f"!! fewer than half of the training clips have a {_m} copy - ids do not match?")

# ---- 2. three fine-tunes in parallel: the v9 recipe + occluded copies
seeds = [int(s) for s in a.seeds.split(",")]
procs = []
for s, g in zip(seeds, a.gpus.split(",")):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, TF_FORCE_GPU_ALLOW_GROWTH="true")
    cmd = [PY, "src/training/train.py", "--data", f"{D}/wlasl300.npz",
           "--vocab", "src/data/wlasl/vocab_wlasl300.json", "--init", INIT,
           "--out", f"runs/ft300tv_s{s}", "--epochs", str(a.epochs), "--bs", "64",
           "--workers", "2", "--seed", str(s), "--train-splits", "train,val", "--val-split", "none",
           "--alt-data", "runs/wlasl300_occ.npz,runs/wlasl300_deg.npz", "--p-alt", a.p_alt]
    procs.append((s, subprocess.Popen(cmd, env=env, stdout=open(f"runs/s{s}.log", "w"),
                                      stderr=subprocess.STDOUT)))
    print(f"started seed {s} on GPU {g}", flush=True)
t0 = time.time()
while any(p.poll() is None for _, p in procs):
    time.sleep(60)
    status = []
    for s, _ in procs:
        log = open(f"runs/s{s}.log", errors="ignore").read().splitlines()
        ep = [l for l in log if l.startswith("Epoch ")]
        status.append(f"s{s} {ep[-1].split()[1] if ep else '-'}")
    print(f"[{(time.time() - t0) / 60:.0f} min] " + "  ".join(status), flush=True)
for s, p in procs:
    log = open(f"runs/s{s}.log", errors="ignore").read().splitlines()
    print(f"seed {s}: exit {p.returncode}\n  " + "\n  ".join([l for l in log if "degraded copies" in l]
                                                         + log[-3:]), flush=True)
ok = [s for s, p in procs if p.returncode == 0]
if not ok:
    sys.exit("!! every seed failed; see runs/s*.log")

# ---- 3. INT8 export + config (v13 = v9 settings + reset_per_clip, new weights)
for f in glob.glob("src/models/echogesture_int8_s*.tflite"):
    os.remove(f)
models = []
for s in ok:
    out = f"src/models/echogesture_int8_s{s}.tflite"
    if sh(f"{PY} src/training/convert.py --ckpt runs/ft300tv_s{s}/last.keras "
          f"--calib runs/ft300tv_s{s}/calib_windows.npy --out {out}") == 0:
        models.append(os.path.basename(out))
shutil.copy(f"runs/ft300tv_s{ok[0]}/vocab.json", "src/models/vocab.json")
cfg.update(models=models, tta=a.tta, gap_fill=True, low_light=True, reset_per_clip=True)
json.dump(cfg, open(cfg_path, "w"), indent=2)
print("config:", cfg, flush=True)
with open("src/VERSION", "w") as f:
    f.write("v13 (2026-11-14): trained with landmarks of really occluded, blurred and darkened train/val videos (run09)\n")

# ---- 4. clean accuracy on the official test split (packed landmarks; v9 was 71.92)
sh(f"{PY} src/tools/evaluate.py --packed {D}/wlasl300.npz --split test --device '{a.device}'")
bundle()                                            # keep the models even if a later step fails

# ---- 5. end to end on the test VIDEOS (MediaPipe into the main env only now, after training)
sh(f"{PY} -m pip install -q mediapipe==1.2.2 ai-edge-litert==2.3.0 2>&1 | tail -2")
rows = [r for r in rows_all if r["split"] == "test" and os.path.exists(os.path.join(VID, f"{r['video_id']}.mp4"))]
rows = rows[: a.limit or None]
os.makedirs("results", exist_ok=True)
for name, order in (("test_manifest.csv", rows), ("test_manifest_reversed.csv", rows[::-1])):
    with open(f"results/{name}", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "path", "gloss"])
        for r in order:
            w.writerow([r["video_id"], f"{r['video_id']}.mp4", r["gloss"].upper()])
sh(f"{PY} src/main.py --videos '{VID}' --manifest results/test_manifest.csv --out results")
sh(f"{PY} src/main.py --videos '{VID}' --manifest results/test_manifest_reversed.csv --out results/reversed")
truth = {r["video_id"]: r["gloss"].upper() for r in rows}
preds = {json.loads(l)["clip_id"]: json.loads(l) for l in open("results/predictions.jsonl")}
rev = {json.loads(l)["clip_id"]: json.loads(l) for l in open("results/reversed/predictions.jsonl")}
n = len(preds)
top1 = sum(p["prediction"] == truth[k] for k, p in preds.items())
top5 = sum(truth[k] in p["top5"] for k, p in preds.items())
same = sum(preds[k]["prediction"] == rev[k]["prediction"] for k in preds)
print(f"\n== v13 predict() on {n} official test videos: top-1 {100 * top1 / n:.2f}%  top-5 {100 * top5 / n:.2f}%  "
      f"missing ids: {len(set(truth) - set(preds))} ==", flush=True)
print(f"== determinism: forward vs reversed order agree on {same}/{n} clips ==", flush=True)
sys.path[:0] = ["src", "src/tools"]
from evaluate import update_measured  # noqa: E402
update_measured(top1_accuracy={
    "value": round(100 * top1 / n, 2), "device": a.device,
    "runtime": f"LiteRT INT8, {len(models)} model(s) x TTA {a.tta}, MediaPipe Holistic",
    "note": f"predict() end to end on {n} official WLASL300 test videos, top-5 {100 * top5 / n:.2f}%; "
            f"each clip independent (fresh MediaPipe graph); forward vs reversed order agree {same}/{n}"})
sh(f"{PY} src/tools/bench_latency.py --device '{a.device}' --video '{os.path.join(VID, rows[0]['video_id'] + '.mp4')}'")
sh(f"{PY} src/tools/robustness.py --videos '{VID}' --subset 300 --split test "
   f"--workers {a.workers} --device '{a.device}'{lim}")
rob = json.load(open("results/robustness.json"))
agree = sum(preds.get(r["video_id"], {}).get("prediction") == r.get("clean") for r in rob["per_clip"])
print(f"== robustness clean pass agrees with the batch run on {agree}/{len(rob['per_clip'])} clips ==", flush=True)
sh(f"{PY} src/tools/make_results_csv.py")
sh("cat results/results.csv")
bundle()
print(f"done in {(time.time() - T0) / 60:.0f} min", flush=True)
