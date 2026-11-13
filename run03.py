"""EchoGesture run 03: fine-tune on WLASL300 train+val with 3 seeds (in parallel),
export each to INT8, ship them as a 3-model ensemble with 2-window TTA, and
measure everything on the untouched official TEST split.

On Kaggle (inputs: echogesture code, 01_extract output, Version-05 output):
    !python run03.py
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


def find(pattern):
    hits = sorted(glob.glob(os.path.join(a.input_root, "**", pattern), recursive=True))
    if not hits:
        sys.exit(f"!! could not find {pattern} under {a.input_root} - is that input attached?")
    return hits[0]


D = os.path.dirname(find("wlasl300.npz"))
_init = sorted(glob.glob(os.path.join(a.input_root, "**", "runs/wl2000/best.keras"), recursive=True))
if _init:
    INIT = _init[0]
else:                                    # fall back to the zip the training run bundled
    import zipfile
    z = find("echogesture_trained.zip")
    zipfile.ZipFile(z).extract("runs/wl2000/best.keras", "init_from_zip")
    INIT = "init_from_zip/runs/wl2000/best.keras"
print("data:", D, "\ninit:", INIT, flush=True)
seeds = [int(s) for s in a.seeds.split(",")]
gpus = a.gpus.split(",")
os.makedirs("runs", exist_ok=True)

# ---- 1. three fine-tunes in parallel (train+val, no model selection on any split)
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

# ---- 2. INT8 export of each model (last epoch; nothing was selected on test)
models = []
for s in ok:
    out = f"src/models/echogesture_int8_s{s}.tflite"
    if sh(f"{PY} src/training/convert.py --ckpt runs/ft300tv_s{s}/last.keras "
          f"--calib runs/ft300tv_s{s}/calib_windows.npy --out {out}") == 0:
        models.append(os.path.basename(out))
shutil.copy(f"runs/ft300tv_s{ok[0]}/vocab.json", "src/models/vocab.json")
cfg_path = "src/models/config.json"
cfg = json.load(open(cfg_path))
cfg.update(models=models, tta=a.tta)
json.dump(cfg, open(cfg_path, "w"), indent=2)
print("config:", cfg, flush=True)

# ---- 3. accuracy on the untouched official test split
ev = f"{PY} src/tools/evaluate.py --packed {D}/wlasl300.npz --split {a.test_split} --device '{a.device}'"
print("\n== components (for the record, not used to choose anything) ==", flush=True)
sh(f"{ev} --models {models[0]} --tta 1 --no-record")
sh(f"{ev} --models {models[0]} --tta {a.tta} --no-record")
sh(f"{ev} --tta 1 --no-record")
print("\n== FINAL shipped configuration ==", flush=True)
sh(ev)

# ---- 4. latency, size, results.csv, bundle
sh(f"{PY} src/tools/bench_latency.py --device '{a.device}'")
sh(f"{PY} src/tools/make_results_csv.py")
sh("cat results/results.csv")
sh("zip -qr /kaggle/working/echogesture_run03.zip src/models results runs/*.log "
   "runs/ft300tv_s*/history.csv runs/ft300tv_s*/vocab.json && ls -lh /kaggle/working/echogesture_run03.zip")
