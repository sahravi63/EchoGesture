"""Build results/results.csv from results/measured.json (filled by evaluate.py,
bench_latency.py, robustness.py, android_power_log.py).

    python src/tools/make_results_csv.py
"""
import csv
import json
from pathlib import Path

RES = Path(__file__).resolve().parent.parent.parent / "results"
ROWS = [("top1_accuracy", 70, ">=", "%"), ("latency_p95", 50, "<=", "ms"),
        ("model_size", 25, "<=", "MB"), ("accuracy_drop_robustness", 5, "<=", "pt"),
        ("avg_power", 500, "<=", "mW"), ("thermal_drift_10min", 20, "<=", "%")]

RES.mkdir(exist_ok=True)
mf = RES / "measured.json"
m = json.loads(mf.read_text()) if mf.exists() else {}
with open(RES / "results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "target", "measured", "unit", "device", "runtime", "pass"])
    for name, tgt, op, unit in ROWS:
        r = m.get(name, {})
        v = r.get("value")
        ok = "" if v is None else ("PASS" if (v >= tgt if op == ">=" else v <= tgt) else "FAIL")
        w.writerow([name, f"{op}{tgt}", "" if v is None else v, unit, r.get("device", ""), r.get("runtime", ""), ok])
        print(f"{name:26s} {op}{tgt:<4} {str(v):>8} {unit:3s} {ok}")
print(f"wrote {RES / 'results.csv'}")
