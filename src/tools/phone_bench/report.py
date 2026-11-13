"""Turn the phone_bench.sh output (adb pull /data/local/tmp/eg/out) into results rows.

    python src/tools/phone_bench/report.py phone_results
avg_power  = battery power during the 10-min pipeline load minus the idle baseline (same screen
             state), from the fuel gauge's coulomb counter (dumpsys battery "Charge counter")
thermal_drift_10min = classifier latency after 10 min of load vs on the cool phone (%)
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC / "tools"))
from evaluate import update_measured  # noqa: E402

out = Path(sys.argv[1])
rows = list(csv.DictReader(open(out / "power.csv")))


def phase(name):
    r = [x for x in rows if x["phase"] == name and x["charge_uAh"]]
    t = np.array([float(x["t"]) for x in r]); q = np.array([float(x["charge_uAh"]) for x in r])
    v = np.array([float(x["voltage_mV"]) for x in r])
    plugged = sum(int(x["plugged"] or 0) for x in r)
    ma = -np.polyfit(t - t[0], q, 1)[0] * 3.6          # uAh/s -> mA (least-squares slope)
    cpu = [float(x["cpu_temp_C"]) for x in r if x["cpu_temp_C"]]
    bat = [float(x["batt_temp_dC"]) / 10 for x in r if x["batt_temp_dC"]]
    return dict(mw=ma * v.mean() / 1000, ma=ma, v=v.mean(), secs=t[-1] - t[0], dq=q[0] - q[-1],
                plugged=plugged, cpu=(cpu[0], cpu[-1]) if cpu else None, bat=(bat[0], bat[-1]) if bat else None)


idle, load = phase("idle"), phase("load")
if idle["plugged"] or load["plugged"]:
    print("!! the phone reported external power during the run - power numbers are not valid")


def lat(name):
    m = re.search(r"Inference \(avg\):\s*([\d.e+]+)", (out / f"{name}.txt").read_text())
    return float(m.group(1)) / 1000 if m else float("nan")      # us -> ms


l0, l1 = lat("lat_start"), lat("lat_end")
dev = " ".join((out / "device.txt").read_text().split())
drift = 100 * (l1 - l0) / l0
cost = load["mw"] - idle["mw"]
for n, p in (("idle", idle), ("load", load)):
    print(f"{n}: {p['mw']:.0f} mW ({p['ma']:.0f} mA at {p['v'] / 1000:.2f} V; {p['dq'] / 1000:.0f} mAh over "
          f"{p['secs']:.0f} s); CPU {p['cpu']} C, battery {p['bat']} C")
print(f"device {dev}\ninference cost (load - idle) {cost:.0f} mW\n"
      f"classifier latency {l0:.2f} -> {l1:.2f} ms ({drift:+.1f} %)")
for f in sorted(out.glob("load_*.txt")):
    print(f"  under load {f.stem[5:]:8s} {lat(f.stem):.2f} ms")
if not (idle["plugged"] or load["plugged"]):
    cpu = f"; CPU {load['cpu'][0]:.0f}->{load['cpu'][1]:.0f} C" if load["cpu"] else ""
    update_measured(
        avg_power={"value": round(cost), "device": dev,
                   "runtime": "LiteRT CPU: Holistic per-frame models at 15 fps + 3 INT8 classifiers x 2 views per second",
                   "note": f"battery coulomb counter: 10-min load {load['mw']:.0f} mW minus idle {idle['mw']:.0f} mW, "
                           f"same screen state{cpu}"})
update_measured(
    thermal_drift_10min={"value": round(drift, 1), "device": dev, "runtime": "INT8 classifier, 4 threads",
                         "note": f"latency {l0:.2f} ms on a cool phone -> {l1:.2f} ms after 10 min of pipeline load"})
