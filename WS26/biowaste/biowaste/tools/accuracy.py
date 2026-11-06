#!/usr/bin/env python3
"""
Accuracy report: compares what the device logged with the truth from the reference scale.

ground_truth.csv (fill in while testing - one line per real tip):
    test_no,true_weight_g,true_type,record_no,notes
    1,212,preparation_waste,1,apple halves
    2,95,plate_leftovers,,MISSED - tipped too fast
record_no = the number shown on the dashboard for that tip; leave empty if no record appeared.

Usage: python3 accuracy.py ground_truth.csv ../server/data/events.csv
"""
import csv
import statistics
import sys

TYPES = ["preparation_waste", "spoiled_stock", "plate_leftovers"]


def main(gt_path, ev_path):
    gt = list(csv.DictReader(open(gt_path)))
    ev = {int(r["record_no"]): r for r in csv.DictReader(open(ev_path))}
    matched, missed, errs, pct, type_ok = [], [], [], [], 0
    confusion = {t: {u: 0 for u in TYPES} for t in TYPES}
    for g in gt:
        rn = g.get("record_no", "").strip()
        if not rn or int(rn) not in ev:
            missed.append(g)
            continue
        e = ev[int(rn)]
        tw, mw = float(g["true_weight_g"]), float(e["weight_g"])
        errs.append(mw - tw)
        pct.append(abs(mw - tw) / tw * 100 if tw else 0)
        type_ok += e["type"] == g["true_type"]
        confusion[g["true_type"]][e["type"]] += 1
        matched.append((g, e))
    used = {int(g["record_no"]) for g in gt if g.get("record_no", "").strip()}
    lo = min(used) if used else 0
    hi = max(used) if used else 0
    extra = [r for n, r in ev.items() if lo <= n <= hi and n not in used]

    n = len(gt)
    print(f"Test disposals (truth):   {n}")
    print(f"Detected:                 {len(matched)}  ({100*len(matched)/n:.0f}%)")
    print(f"Missed:                   {len(missed)}")
    print(f"Extra / false records:    {len(extra)}")
    if errs:
        print(f"Weight error mean:        {statistics.mean(errs):+.1f} g (bias)")
        print(f"Weight error |mean|:      {statistics.mean(abs(x) for x in errs):.1f} g")
        print(f"Weight error max:         {max(abs(x) for x in errs):.1f} g")
        print(f"Weight error % (mean):    {statistics.mean(pct):.1f} %")
        print(f"Type correct:             {type_ok}/{len(matched)} ({100*type_ok/len(matched):.0f}%)")
        print("\nConfusion (rows = true type, cols = logged type)")
        print(" " * 20 + "".join(f"{t[:12]:>14}" for t in TYPES))
        for t in TYPES:
            print(f"{t:20}" + "".join(f"{confusion[t][u]:>14}" for u in TYPES))
    if missed:
        print("\nMissed:")
        for g in missed:
            print(f"  test {g['test_no']}: {g['true_weight_g']} g {g['true_type']} - {g.get('notes', '')}")
    if extra:
        print("\nExtra records:")
        for r in extra:
            print(f"  record {r['record_no']}: {r['weight_g']} g {r['type']} at {r['timestamp']}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
