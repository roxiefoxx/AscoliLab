"""Check whether optimal_order_85.csv recovers the trisynaptic direction.

Trisynaptic path: EC -> DG -> CA3 -> CA1.
Recovery requires position(EC) < position(DG) < position(CA3) < position(CA1).

Run:  python tarjan_reorder/check_trisynaptic_claim.py
"""
import csv, collections
from pathlib import Path

CSV = Path(__file__).with_name("optimal_order_85.csv")
rows = list(csv.DictReader(CSV.open()))
pos = {r["cell_type"]: int(r["position"]) for r in rows}

def region(name):
    for r in ["CA1", "CA2", "CA3c", "CA3", "DG", "SUB", "MEC", "LEC", "EC"]:
        if name.startswith(r):
            return "CA3" if r == "CA3c" else r
    return "other"

by_region = collections.defaultdict(list)
for r in rows:
    by_region[region(r["cell_type"])].append(int(r["position"]))

print(f"artifact: {CSV}  ({len(rows)} rows)\n")
print("region mean positions (ascending):")
for k, v in sorted(by_region.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
    print(f"  {k:5s} n={len(v):3d}  mean={sum(v)/len(v):6.1f}  range={min(v)}-{max(v)}")

print("\nprincipal cells on the trisynaptic route:")
for c in ["EC LI II Pyramidal Fan", "MEC LII Stellate", "DG Granule",
          "CA3 Pyramidal", "CA1 Pyramidal"]:
    print(f"  {c:26s} {pos[c]:3d}")

ec, dg, ca3, ca1 = (pos["EC LI II Pyramidal Fan"], pos["DG Granule"],
                    pos["CA3 Pyramidal"], pos["CA1 Pyramidal"])
ok = ec < dg < ca3 < ca1
print(f"\nEC({ec}) < DG({dg}) < CA3({ca3}) < CA1({ca1})  ->  {ok}")
print("VERDICT:", "trisynaptic direction recovered" if ok else
      "trisynaptic direction NOT recovered — DG falls after CA3 and CA1")
print("What the head of the ordering does show: EC -> MEC -> CA3 -> CA2 -> CA1,")
print("i.e. the direct (temporoammonic) route, which bypasses DG.")
