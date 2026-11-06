# Kickoff brief: operating point, steady state, and the stability buffer

Paste this into a new chat. First attach `AscoliLab/PROJECT_CONTEXT.md`, or point the chat at
it; that file has the data conventions, the numbers, and the "do not assume" list. This brief
adds only what is specific to this task. Its companion analysis is `kron_reduction/` (README
and notebook 01), which this project builds on.

## Task

Compare the 85-type operator at four operating points, **ρ(W) = 0.85, 0.95, 0.99, 1.0**. At
each, ask:

1. Does a steady state exist, and how long does it take to reach it?
2. How large is the stability *buffer* `1 − ρ`, and how much of it does deleting the most
   influential cell types use up?
3. What happens to the transient (peak amplification) and to the long-time response as the
   buffer shrinks to zero?

Build it in its own subdirectory, `AscoliLab/rho_buffer/`. Do not modify anything outside that
directory. Reuse `kron_reduction/kron_core.py` and `jacobian_analysis/jacobian_core.py` by
import; do not copy them.

## Why this is being done

Modelling courses routinely assume the network sits at steady state, `x* = (I − W)⁻¹ u`, and
read results off that. The assumption is free only when there is a margin:

- For `x* = (I − W)⁻¹ u` to exist and attract, the dominant eigenvalue of W must be below 1.
- The slowest mode settles on a time scale of `τ / (1 − ρ)`, so "at steady state" means "after
  that long".
- The steady-state gain `‖(I − W)⁻¹‖` diverges as `ρ → 1`, and non-normality makes it diverge
  faster than `1 / (1 − ρ)`.

`kron_reduction/` showed that the margin is also what absorbs missing cell types. Deleting one
interneuron (MEC LIII Superficial Multipolar Interneuron) used up two-thirds of the 0.05 buffer
at ρ = 0.95. This project makes that trade-off explicit across operating points. It is not
predicting anything about the 37 unparameterised types; it shows how much room the chosen ρ
leaves for them.

## The key fact that shapes the design

**Gain is a pure scale.** `W(g) = g · Ŵ`, where `ρ(Ŵ) = 1`, so every eigenvalue scales by
`g`. Deleting a set of types S therefore multiplies ρ by a factor `m_S` that does **not**
depend on the operating point:

```
ρ_after_deletion = m_S · ρ_full          (m_S fixed by the connectome, not by g)
critical operating point  ρ*_S = 1 / m_S  (above it, the deletion makes the network unstable)
buffer consumed           = (m_S − 1) · ρ_full / (1 − ρ_full)
```

So the analysis has two parts:

- a **gain-free** part: `m_S` for each deletion, and the critical `ρ*_S` it implies;
- a **gain-dependent** part: margin, settling time, steady-state gain and transient, read at
  the four operating points.

Present them separately. Do not re-derive `m_S` at each ρ as if it were new information.

## Baseline already measured (with-self, `kron_core.load_partition(gain)`)

Treat these numbers as a reproduction check: the notebook should match them before it does
anything new.

| ρ_full | buffer 1 − ρ | slowest time constant | ‖(I − W)⁻¹‖₂ | peak ‖e^{Jt}‖₂ | ‖e^{Jt}‖₂ at t = 50τ | steady state |
|---:|---:|---:|---:|---:|---:|---|
| 0.85 | 0.15 | 6.7 τ | 113 | 23.7× | 0.01 | yes, fast |
| 0.95 | 0.05 | 20 τ | 366 | 27.1× | 1.63 | yes |
| 0.99 | 0.01 | 100 τ | 1,950 | 28.6× | 12.1 | yes, but slow |
| 1.00 | 0 | ∞ | ∞ (singular) | 28.9× | 19.9 (never decays) | **no**: an integrator |

The peak barely changes, from 23.7× to 28.9×, because it is set by non-normality. The tail
changes completely, because it is set by the buffer.

Deletion multipliers, from leave-one-out over all 85 types:

| deleted | E/I | m_S | ρ* (unstable above) | ρ after deletion at 0.85 / 0.95 / 0.99 / 1.0 |
|---|---|---:|---:|---|
| MEC LIII Superficial Multipolar Interneuron | i | 1.0347 | 0.9665 | 0.880 / 0.983 / **1.024** / **1.035** |
| + MEC LIII Superficial Trilayered Interneuron (greedy 2) | i | 1.0404 | 0.9612 | 0.884 / 0.988 / **1.030** / **1.040** |
| + CA3 Basket (greedy 3) | i | 1.0406 | 0.9609 | 0.885 / 0.989 / **1.030** / **1.041** |
| the 13 absent from the 72-type list | 4e / 9i | 1.0307 | 0.9702 | 0.876 / 0.979 / **1.020** / **1.031** |
| EC LI II Multipolar Pyramidal | e | 0.4612 | n/a | halves ρ |
| EC LII III Pyramidal Tripolar | e | 0.8086 | n/a | n/a |

Buffer consumed by deleting MEC LIII SMI: **20%** at 0.85, **66%** at 0.95, **344%** at 0.99
(the network becomes unstable), and undefined at 1.0, where no buffer exists.

How to read this: stabilisers are inhibitory types whose deletion raises ρ. After the top two,
the effect collapses; every other single type has `m ≤ 1.0003`. Destabilisers are the
excitatory engine of the dominant mode, whose deletion lowers ρ.

## Deliverables

### 1. `01_rho_buffer.ipynb`: operating-point comparison

- Reproduce the baseline table above.
- Settling: for a step input, report the time to come within 1%, 5% and 10% of `x*`, both for
  a worst-case input (the right singular vector of `(I − W)⁻¹`) and for unit DG Granule drive.
  This is the "how long is the steady-state assumption a lie" table.
- Steady-state gain against `1 / (1 − ρ)`: overlay the normal-matrix prediction on the measured
  `‖(I − W)⁻¹‖`. The excess is the non-normal contribution.
- At ρ = 1.0: show that `(I − W)` is singular, and identify the input direction that grows
  without bound under constant drive. That direction is the left null vector, dominated by
  EC LI II Multipolar Pyramidal. Show which inputs still reach a steady state: those
  orthogonal to it.
- Figure: `‖e^{Jt}‖` envelopes for the four ρ on one axis, with log-time, so that the peak
  (shared) and the tail (diverging) are both visible.

### 2. `02_influential_deletions.ipynb`: removing influential types

- Leave-one-out `m_r` for all 85 types, ranked. Report each type's critical `ρ*_r`.
- First-order screen `uᵀ Δ_r v / uᵀ v`, as in `kron_core.single_node_table`, against exact
  `m_r − 1`, for all 85 rather than only the 13.
- Greedy cumulative deletion of the top stabilisers, k = 1 … 10. Plot `ρ*` against k: how
  quickly does the safe operating range shrink?
- For each ρ and each deletion, a 2 × 4 grid: stable or unstable, buffer consumed, peak
  amplification, and steady-state gain after deletion. Use the full-85 scale; **do not**
  renormalise after deletion, because renormalising resets the buffer and hides exactly the
  effect being measured.
- Mirror experiment: delete the top destabilisers and report how much buffer that *buys*.

### 3. `README.md`

- One table that answers "which ρ should a steady-state analysis of this connectome use, and
  what does each choice assume about the cell types we do not have?"
- A short statement of the scale invariance above, so a reader does not mistake `m_S` for a
  gain-dependent result.

## Conventions

- Orientation, Dale check and normalisation all go through `jacobian_core.load_jacobian_data`
  and `kron_core.load_partition(gain)`. The matrix on disk is `[pre, post]`; never transpose by
  hand.
- Normalise **once**, on the full 85, then scale. A deleted or reduced operator is never
  renormalised in the main results. A renormalised column may appear only as a labelled
  contrast.
- Continuous time throughout: `J = −I + W`, with stability `max Re λ(J) < 0`, which is
  equivalent here to the dominant eigenvalue of W being below 1 (the dominant eigenvalue is
  real and simple). Do not use `ρ(J) < 1` as a stability test.
- Name the norm on every amplification number (‖·‖₂ on population activity, as above).
- Name blocks in the arrow sense (`E_to_I` = E source onto I receiver).

## Do not assume

- **ρ = 1.0 is not "at steady state".** It is the one operating point with no steady state for
  generic input. Any course result that holds at ρ = 1 is a statement about a restricted input
  subspace.
- **A lower ρ is not "safer" in every sense.** It buys buffer and a fast steady state, and
  costs little peak amplification (23.7× against 27.1×). It is a modelling choice, not a
  measurement, and nothing in the data picks 0.85 over 0.95.
- **The deletion multipliers are not predictions about the 37 missing types.** They measure
  how much a single type *can* move the operating point in this connectome: the scale of the
  unknown, not its value.
- **Do not use the size of a type's weights as a proxy for its influence.** CA2 Pyramidal has
  21× MEC LIII SMI's Kron-correction norm and 1/23 of its spectral effect, with the opposite
  sign (`kron_reduction/` §5).
- **Do not treat steady-state metrics at 0.99 as reliable.** The slowest mode takes 100τ to
  settle, and the steady-state gain is about 2,000. Small changes to W change the answer a lot.
