"""Sample phone battery power + temperature over ADB while the EchoGesture app runs.

Unplug USB power first and connect over Wi-Fi ADB:
    adb tcpip 5555 ; adb connect <phone-ip>:5555 ; (unplug cable)
Run three phases of 10 minutes each and compute inference cost = app - camera_only:
    python src/tools/android_power_log.py --label idle        --minutes 10
    python src/tools/android_power_log.py --label camera_only --minutes 10
    python src/tools/android_power_log.py --label app         --minutes 10 --finalize

Thermal drift comes from the app's own per-second latency log (see android/ README):
    python src/tools/android_power_log.py --latency-log app_latency.csv
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC / "tools"))
from evaluate import update_measured  # noqa: E402
OUT = SRC.parent / "results" / "power"


def adb(cmd):
    return subprocess.run(["adb", "shell", cmd], capture_output=True, text=True, timeout=10).stdout.strip()


def sample():
    i = float(adb("cat /sys/class/power_supply/battery/current_now") or 0)   # usually uA
    v = float(adb("cat /sys/class/power_supply/battery/voltage_now") or 0)   # usually uV
    t = float(adb("cat /sys/class/power_supply/battery/temp") or 0) / 10      # deci-degC
    return abs(i) * v / 1e9, t                                               # mW (uA*uV=pW -> /1e9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label"); ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--latency-log", help="CSV from the app: second,model_ms,landmark_ms,fps")
    ap.add_argument("--device", default="Galaxy phone")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.label:
        model = adb("getprop ro.product.model")
        rows, t_end = [], time.time() + 60 * a.minutes
        while time.time() < t_end:
            p, temp = sample(); rows.append((time.time(), p, temp))
            print(f"{a.label}: {p:7.0f} mW  {temp:.1f} C", end="\r"); time.sleep(1)
        arr = np.array(rows)
        np.savetxt(OUT / f"{a.label}.csv", arr, delimiter=",", header="t,mW,tempC")
        print(f"\n{a.label}: mean {arr[:,1].mean():.0f} mW, temp {arr[0,2]:.1f}->{arr[-1,2]:.1f} C on {model}")
    if a.finalize:
        app = np.loadtxt(OUT / "app.csv", delimiter=",")[:, 1].mean()
        cam = np.loadtxt(OUT / "camera_only.csv", delimiter=",")[:, 1].mean()
        update_measured(avg_power={"value": round(float(app - cam)), "device": a.device,
                                   "runtime": "app vs camera-preview baseline",
                                   "note": f"app total {app:.0f} mW, camera-only {cam:.0f} mW"})
        print(f"avg_power (inference cost) = {app - cam:.0f} mW  (app total {app:.0f} mW)")
    if a.latency_log:
        d = np.genfromtxt(a.latency_log, delimiter=",", names=True)
        first = np.median(d["model_ms"][d["second"] < 60] + d["landmark_ms"][d["second"] < 60])
        last = np.median(d["model_ms"][d["second"] >= 540] + d["landmark_ms"][d["second"] >= 540])
        drift = 100 * (last - first) / first
        update_measured(thermal_drift_10min={"value": round(float(drift), 1), "device": a.device,
                                             "runtime": "per-frame latency, minute 10 vs minute 1",
                                             "note": f"{first:.1f} ms -> {last:.1f} ms"})
        print(f"thermal drift {drift:.1f}% ({first:.1f} -> {last:.1f} ms per frame)")


if __name__ == "__main__":
    main()
