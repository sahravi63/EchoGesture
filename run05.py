"""EchoGesture run 05 (v9 robustness): retrain the 3-seed ensemble with gap filling and
detector-failure augmentation, ship it with low-light correction, then measure
clean accuracy (packed landmarks) and end-to-end robustness on the official TEST videos.

Nothing here is tuned on test: the v9 settings are fixed in code before this run.
Inputs on Kaggle: echogesture code, 01_extract output (packed npz), Version-05 output
(wl2000 init), risangbaskoro/wlasl-processed (videos). GPU T4 x2, Internet ON.
    !python run05.py
"""
import argparse
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
ap.add_argument("--tta", type=int, default=2)
ap.add_argument("--test-split", default="test")
ap.add_argument("--device", default="Kaggle CPU")
ap.add_argument("--workers", type=int, default=4, help="robustness worker processes")
ap.add_argument("--skip-robustness", action="store_true")
ap.add_argument("--zip", default="/kaggle/working/echogesture_run05.zip")
a = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
PY = sys.executable
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
    sh(f"zip -qr {a.zip} src/models/*.tflite src/models/*.json results runs/*.log "
       f"runs/ft300tv_s*/history.csv runs/ft300tv_s*/vocab.json && ls -lh {a.zip}")


print("code version:", open("src/VERSION").read().strip(), flush=True)
D = os.path.dirname(find("wlasl300.npz"))
_init = sorted(glob.glob(os.path.join(a.input_root, "**", "runs/wl2000/best.keras"), recursive=True))
if _init:
    INIT = _init[0]
else:
    import zipfile
    zipfile.ZipFile(find("echogesture_trained.zip")).extract("runs/wl2000/best.keras", "init_from_zip")
    INIT = "init_from_zip/runs/wl2000/best.keras"
print("data:", D, "\ninit:", INIT, flush=True)
seeds = [int(s) for s in a.seeds.split(",")]
gpus = a.gpus.split(",")
os.makedirs("runs", exist_ok=True)

# ---- 1. three fine-tunes in parallel (train+val; gap filling + v9 augmentation are the defaults)
procs = []
for s, g in zip(seeds, gpus):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, TF_FORCE_GPU_ALLOW_GROWTH="true")
    cmd = [PY, "src/training/train.py", "--data", f"{D}/wlasl300.npz",
           "--vocab", "src/data/wlasl/vocab_wlasl300.json", "--init", INIT,
           "--out", f"runs/ft300tv_s{s}", "--epochs", str(a.epochs), "--bs", "64",
           "--workers", "2", "--seed", str(s), "--train-splits", "train,val", "--val-split", "none"]
    procs.append((s, subprocess.Popen(cmd, env=env, stdout=open(f"runs/s{s}.log", "w"),
                                      stderr=subprocess.STDOUT)))
    print(f"started seed {s} on GPU {g}", flush=True)
t0 = time.time()
while any(p.poll() is None for _, p in procs):
    time.sleep(60)
    status = []
    for s, _ in procs:
        ep = [l for l in open(f"runs/s{s}.log", errors="ignore").read().splitlines() if l.startswith("Epoch ")]
        status.append(f"s{s} {ep[-1].split()[1] if ep else '-'}")
    print(f"[{(time.time() - t0) / 60:.0f} min] " + "  ".join(status), flush=True)
failed = [s for s, p in procs if p.returncode]
for s, p in procs:
    tail = open(f"runs/s{s}.log", errors="ignore").read().splitlines()[-3:]
    print(f"seed {s}: exit {p.returncode}\n  " + "\n  ".join(tail), flush=True)
ok = [s for s in seeds if s not in failed]
if not ok:
    sys.exit("!! every seed failed; see runs/s*.log")

# ---- 2. INT8 export + config (v9: gap_fill and low_light switched on)
for f in glob.glob("src/models/echogesture_int8_s*.tflite"):
    os.remove(f)                                    # no v7 models left behind by mistake
models = []
for s in ok:
    out = f"src/models/echogesture_int8_s{s}.tflite"
    if sh(f"{PY} src/training/convert.py --ckpt runs/ft300tv_s{s}/last.keras "
          f"--calib runs/ft300tv_s{s}/calib_windows.npy --out {out}") == 0:
        models.append(os.path.basename(out))
shutil.copy(f"runs/ft300tv_s{ok[0]}/vocab.json", "src/models/vocab.json")
cfg_path = "src/models/config.json"
cfg = json.load(open(cfg_path))
cfg.update(models=models, tta=a.tta, gap_fill=True, low_light=True)
json.dump(cfg, open(cfg_path, "w"), indent=2)
print("config:", cfg, flush=True)

# ---- 3. clean accuracy on the untouched official test split (pre-extracted landmarks)
ev = f"{PY} src/tools/evaluate.py --packed {D}/wlasl300.npz --split {a.test_split} --device '{a.device}'"
print("\n== components (for the record, not used to choose anything) ==", flush=True)
sh(f"{ev} --models {models[0]} --tta 1 --no-record")
sh(f"{ev} --tta 1 --no-record")
print("\n== FINAL shipped configuration (packed landmarks) ==", flush=True)
sh(ev)
sh(f"{PY} src/tools/bench_latency.py --device '{a.device}'")
sh(f"{PY} src/tools/make_results_csv.py")
bundle()                                            # keep the models even if step 4 fails

# ---- 4. end-to-end robustness on the test VIDEOS (MediaPipe installed only now, after
#         training, so it cannot disturb TensorFlow's packages)
if not a.skip_robustness:
    import csv
    rows = [r for r in csv.DictReader(open("src/data/wlasl/wlasl_manifest.csv", newline=""))
            if int(r["gloss_rank"]) < 300 and r["split"] == a.test_split]
    VID = None
    for r in rows[:20]:
        hits = sorted(glob.glob(os.path.join(a.input_root, "**", f"{r['video_id']}.mp4"), recursive=True))
        if hits:
            VID = os.path.dirname(hits[0])
            break
    if not VID:
        print("!! no WLASL test videos found - attach risangbaskoro/wlasl-processed; skipping robustness")
    else:
        print(f"\n== end-to-end robustness on {VID} ==", flush=True)
        sh(f"{PY} -m pip install -q mediapipe==1.2.2 ai-edge-litert==2.3.0 2>&1 | tail -2")
        sh(f"{PY} src/tools/fetch_models.py")
        sh(f"{PY} src/tools/robustness.py --videos '{VID}' --subset 300 --split {a.test_split} "
           f"--workers {a.workers} --device '{a.device}'")
        sh(f"{PY} src/tools/make_results_csv.py")
        bundle()
sh("cat results/results.csv")
print(f"done in {(time.time() - T0) / 60:.0f} min", flush=True)
