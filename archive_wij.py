"""
Archive the standalone w_ij matrices.

Decision (Sept 2026): m_ij = w_ij * kappa_j is the operator for ALL analyses.
The raw w_ij matrices are the *structural* connectome; m_ij is the *functional*
one. Keeping both live invites accidental mixing, so w_ij is moved out of the
active tree and kept for a later, explicit structural-vs-functional comparison.

Nothing imports these files: a grep of non-normal_matrices/analyses/*.py and
mij_paper_replication.ipynb finds no reference to wij. This move is safe.

NOT archived, deliberately:
  - the `w_ij` column inside mij_netlist.csv  (provenance for m_ij; keep)
  - 85_kj.csv                                 (the kappa_j vector; keep)
  - ml_cij / ml_gij / ml_qi / ml_taud / ml_vij netlists (component factors of
    the m_ij product; left in place -- see note in the archive README)

Run from the repository root:  python archive_wij.py
Add --dry-run to preview.
"""
from __future__ import annotations
import argparse, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "archive" / "wij_structural"

TARGETS = [
    Path("non-normal_matrices/data/wij_matrix.csv"),
    Path("non-normal_matrices/data/ml_wij_netlist.csv"),
    Path("schur_decomp/outputs/moved_csv/matrices/wij_matrix.csv"),
]

README = """# archive/wij_structural

Archived Sept 2026. These are the raw `w_ij` matrices -- the **structural**
connectome, before the postsynaptic gain term `kappa_j` is applied.

All active analysis uses `m_ij = w_ij * kappa_j`, the **functional** operator.

## Why they were pulled out

Having both live in the tree makes it possible to load the wrong one and get a
plausible-looking result. There is no orientation oracle for this mistake the
way there is for a transpose: `w_ij` and `m_ij` have the same shape, the same
labels, the same sign pattern and the same E/I block structure. Only the
magnitudes differ, and they differ in a way that matters.

## The measurement that motivated the split

Block Frobenius norms, `[post, pre]` convention, E->I means E source onto I
receiver:

| Operator | E->I / I->E |
|---|---:|
| `m_ij = w_ij * kappa_j` | 912x |
| `w_ij` alone | 86x |

So roughly one order of magnitude (10.6x) of the feedforward asymmetry that the
non-normality argument rests on is contributed by postsynaptic gain, not by
wiring. `kappa_j` is indexed by the postsynaptic type, so the entire E->I block
is multiplied by interneuron kappa and the entire I->E block by principal-cell
kappa (median 0.0935 vs 0.0345 Hz/pA, ~9x at the tails).

That is a real result, not a confound -- but it belongs to a separate question:
how much of this circuit's feedforward geometry is wiring and how much is
excitability? That comparison is deferred, not abandoned.

## Caveat on these specific files

`wij_matrix.csv` was not verified to be exactly the `w_ij` column of
`mij_netlist.csv`. Before quoting the 86x figure anywhere, reconstruct `w_ij`
from the netlist and confirm. The netlist is the authoritative source.

## Component factors

`ml_cij`, `ml_gij`, `ml_qi`, `ml_taud` and `ml_vij` netlists were left in
`non-normal_matrices/data/`. They are the individual terms of
`m_ij = q_i * c_ij * g_ij * (V_ref - V_rev) * tau_ij * kappa_j` and are needed to
answer *which* factor drives the very small I->E block (0.8% of block-norm total
even before kappa). Move them here too only if that question is closed.
"""

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    moved, missing = [], []
    for rel in TARGETS:
        src = ROOT / rel
        if not src.exists():
            missing.append(rel)
            continue
        dst = DEST / rel                      # keep original tree under the archive
        if args.dry_run:
            print(f"would move  {rel}  ->  {dst.relative_to(ROOT)}")
            moved.append(rel)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            print(f"SKIP (already archived)  {rel}")
            continue
        shutil.move(str(src), str(dst))
        print(f"moved  {rel}  ->  {dst.relative_to(ROOT)}")
        moved.append(rel)

    if not args.dry_run and moved:
        DEST.mkdir(parents=True, exist_ok=True)
        (DEST / "README.md").write_text(README, encoding="utf-8")
        print(f"wrote  {(DEST / 'README.md').relative_to(ROOT)}")

    for rel in missing:
        print(f"not found (nothing to do)  {rel}")

    print(f"\n{len(moved)} file(s) {'would be ' if args.dry_run else ''}archived.")
    print("Reminder: docs/TRANSPOSE_WARNINGS.md lists both wij_matrix.csv paths "
          "in its file inventory -- update those two rows to point at archive/wij_structural/.")

if __name__ == "__main__":
    main()
