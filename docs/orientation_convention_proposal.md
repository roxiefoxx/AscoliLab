# Proposal: one orientation rule, checked not remembered

Follow-up to `docs/matrix_orientation_inventory.md`. Covers item 1
(`load_schur_blocks`) and item 3 (the `EI`/`IE` collision). EE-backbone is
deliberately out of scope pending the stochastic revamp.

---

## Correction to the inventory

I flagged the motif `.npz` orientation as "unverifiable". That was too strong.
`tarjan_reorder/ORIENTATION_AUDIT.md` already documents it, and notebooks `03`
and `04` already transpose via `TRANSPOSE_SOURCE_TARGET_BLOCKS=True`. The
orientation is handled and documented.

The real problem is *where* the contract lives: in a notebook boolean whose
correct value depends on a producer notebook (`motif_pathway_analysis.ipynb`)
that lives in a different project directory. Nothing in this repo fails if that
flag is wrong — the analysis just silently reports reversed causation. That is
the thing to fix, not the orientation itself.

---

## The Dale's-law check (this is the load-bearing idea)

E/I identity in this dataset comes from the sign of outgoing weights, which
means **the sign pattern of the E/I blocks is itself a orientation oracle**.
Measured on `tarjan_reorder/mij_matrix.csv` + `mij_netlist.csv` (32 E, 53 I,
0 unknown):

| Block (source → receiver) | `M[post, pre]` (correct) | `M[pre, post]` (flipped) |
|---|---|---|
| E → E | 100.0% positive (n=382) | 100.0% positive (n=382) |
| E → I | 100.0% positive (n=352) | 0.0% positive (n=171) |
| I → E | 0.0% positive (n=171) | 100.0% positive (n=352) |
| I → I | 0.0% positive (n=736) | 0.0% positive (n=736) |

The two off-diagonal blocks swap completely, and the separation is 100/0 — no
tolerance to tune, no netlist reconstruction, no residual threshold. If E→I is
not overwhelmingly positive and I→E overwhelmingly negative, the matrix is
transposed. Full stop.

This is better than the netlist-residual test in `load_inputs` for everyday use
because it needs only the matrix and the class labels, so it works on motif
blocks, on submatrices, and on anything downstream that still has E/I labels.

```python
def assert_post_pre(M, classes, *, name="matrix", min_purity=0.9):
    """Raise if M is not M[post, pre]. classes: array of 'e'/'i' per index."""
    import numpy as np
    M = np.asarray(M, dtype=float)
    cls = np.asarray(classes, dtype=str)
    E = np.flatnonzero(cls == "e")
    I = np.flatnonzero(cls == "i")
    if not len(E) or not len(I):
        return  # nothing to test against

    def purity(rows, cols, want_positive):
        nz = M[np.ix_(rows, cols)]
        nz = nz[nz != 0]
        if nz.size == 0:
            return None
        frac = (nz > 0).mean()
        return frac if want_positive else 1.0 - frac

    e_to_i = purity(I, E, True)    # rows=I receivers, cols=E sources
    i_to_e = purity(E, I, False)   # rows=E receivers, cols=I sources
    bad = [f"{k}={v:.3f}" for k, v in
           (("E_to_I_positive", e_to_i), ("I_to_E_negative", i_to_e))
           if v is not None and v < min_purity]
    if bad:
        raise ValueError(
            f"{name} does not look like M[post, pre]: {', '.join(bad)}. "
            "Excitatory sources should drive positive columns and inhibitory "
            "sources negative ones. The matrix is probably transposed."
        )
```

Call it once at every loader boundary. It costs microseconds and it converts
the convention from something to remember into something that fails loudly.

---

## Item 1 — `load_schur_blocks`

Three edits, none of which change any result.

**1. Move the transpose out of the notebook and into the loader, and make the
caller declare the stored orientation.**

```python
def load_motif_blocks_post_pre(csv_path, npz_path, *, stored_orientation):
    """Return (metadata, blocks) with blocks always oriented [post, pre].

    stored_orientation:
        "pre_rows_post_columns"  - blocks as exported by motif_pathway_analysis
        "post_rows_pre_columns"  - blocks already in analysis convention
    """
    metadata = pd.read_csv(csv_path)
    blocks = np.asarray(np.load(npz_path, allow_pickle=True)["analysis_blocks"], float)
    if blocks.shape != (len(metadata), 3, 3):
        raise ValueError(f"Expected {(len(metadata), 3, 3)} blocks, got {blocks.shape}.")
    if stored_orientation == "pre_rows_post_columns":
        blocks = blocks.transpose(0, 2, 1)
    elif stored_orientation != "post_rows_pre_columns":
        raise ValueError("stored_orientation must be 'pre_rows_post_columns' "
                         "or 'post_rows_pre_columns'.")
    return metadata, blocks
```

- The name now states what comes back, so no call site has to remember.
- `stored_orientation` is keyword-only and has **no default** — a wrong value is
  a typo you see, not a default you inherit.
- The vocabulary is the same two strings `load_inputs` already uses, so there is
  one set of words across the repo.
- `TRANSPOSE_SOURCE_TARGET_BLOCKS` disappears from notebooks `03` and `04`,
  replaced by `STORED_BLOCK_ORIENTATION = "pre_rows_post_columns"` passed in.

**2. Verify, don't trust.** After the transpose, run the Dale's-law check on the
stacked blocks using the `node_1/2/3` columns already in the metadata CSV:

```python
    stacked_check(metadata, blocks, ei_by_cell)   # raises if flipped
```

Because the producer notebook lives outside this repo, this is what makes the
orientation a property of the file rather than a property of your memory of how
the file was made.

**3. Delete the duplicate.** `motif_signed_overlap_analysis_script.py:13`
defines `load_blocks_and_metadata`, a byte-for-byte twin of `load_schur_blocks`.
That file already imports three helpers from
`motif_schur_decomposition_analysis_script`, so import the loader too. One
loader, one contract.

Going forward, have the producer write `orientation` as a string array into the
`.npz` alongside `analysis_blocks`; then `stored_orientation` can default to
reading it and fall back to the explicit argument for older files.

---

## Item 3 — the `EI` collision

The collision is wider than the inventory said. Two vocabularies are in play:

- **Arrow sense** (source-then-receiver): `EI` = E source → I receiver.
  Used by `block_view`, `zero_block`, `block_meaning`,
  `block_perturbation_table` in `inhibitory_schur_modulation_script.py`, and by
  `source_receiver_block_view`, `zero_source_receiver_block`, `block_meaning`,
  `block_perturbation_table` in `inhibitory_modulation.py`.
- **Partition sense** (row-block, column-block): `M_EI` = `M[E rows, I cols]` =
  I source → E receiver. Used by `schur_effective_E`, the Neumann functions and
  `disinhibition_sources` in `inhibitory_schur_modulation_script.py`, and by
  `BlockMatrices.ei` / `make_ei_blocks` / `summarize_blocks` /
  `scale_block` / `ablation_analysis` in `inhibitory_modulation.py`.

Both are internally consistent, so **no current output is wrong**. But
`inhibitory_modulation.py` can emit both in one notebook run: cell 10 of
notebook `02` prints a `block` column in the arrow sense, while
`summarize_blocks` prints a `name` column in the partition sense. Same four
strings, opposite meanings, adjacent tables.

### Fix: make the two senses visually impossible to confuse

Do not try to pick a winner — both are legitimate and both are needed. Instead
give each an unmistakable notation and never use bare `EI` again.

| Sense | New notation | Where it appears |
|---|---|---|
| Arrow / biological | `E_to_I`, `I_to_E`, `E_to_E`, `I_to_I` | anything a human reads: dict keys, `removed_block` and `block` column values, plot labels, CSV output |
| Partition / algebraic | `M_EI`, `M_IE` (unchanged) | local variables inside a formula only, never in output |

Concrete edits:

- `inhibitory_schur_modulation_script.py`: change `block_view` keys to the arrow
  names; `zero_block` parses on `_to_` instead of `block[0]`/`block[1]`;
  `block_meaning` keys and the loop at line 82 follow. `schur_effective_E`,
  `neumann_ii_model_selection` and `disinhibition_sources` keep their `M_EI` /
  `M_IE` locals — now unambiguous, because a bare two-letter pair only ever
  means the partition.
- `inhibitory_modulation.py`: same rename for `source_receiver_block_view`,
  `zero_source_receiver_block`, `block_meaning`, and the loop in
  `block_perturbation_table`. Separately, relabel the partition-sense strings in
  `summarize_blocks` from `"EE"/"EI"/"IE"/"II"` to `"M[E,E]"/"M[E,I]"/"M[I,E]"/"M[I,I]"`
  and make `scale_block`/`ablation_analysis` accept those. Bracket notation reads
  as row-then-column and cannot be mistaken for an arrow.
- Add to both modules' docstrings one line: *"Arrow names (`E_to_I`) are
  source→receiver. Bracket names (`M[E,I]`) are row-block, column-block. They
  are different blocks."*

### Why this is the low-confusion option

You never have to recall which convention a given function chose. `E_to_I`
reads as biology; `M[E,I]` reads as indexing. Anything that mixes them is a
visible mismatch in the same table rather than a silent one. And because both
notations survive, no formula has to be rewritten — only labels change, so
diffs are reviewable and numerical output is unchanged except for the strings.

---

## Suggested sequence

1. Add `assert_post_pre` (and the stacked-block wrapper) to a small shared
   module — `common/orientation.py` at the repo root is enough. This is the only
   new code.
2. Wire it into the six loaders that already return `M[post, pre]`. No behavior
   change, immediate safety net.
3. Item 1: rename the loader, add `stored_orientation`, drop the notebook flag,
   delete the duplicate in `motif_signed_overlap_analysis_script.py`.
4. Item 3: the two rename passes above.
5. Promote `tarjan_reorder/ORIENTATION_AUDIT.md` to `docs/` and extend its
   notebook/script tables to `schur_decomp/`, `inhib_modulation/`,
   `motif_analysis/` and `non-normal_matrices/`. It is already the right
   document; it is just scoped to one folder.

Steps 1-2 give most of the protection. Steps 3-4 are mechanical and reviewable.
