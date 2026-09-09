# Validation report

## Overall assessment: Share with caveats

The analysis is numerically consistent and the central claims are supported by independently recomputed quantities. It is suitable for technical review, but publication-grade inference should increase gain-null replication and refine the lesion/edge perturbation screens.

## Methodology review

- The source matrix is square (85×85), finite, and has identical row/column labels.
- The convention is explicit throughout: `A[i,j]` is the effect of source `j` on target `i`.
- Raw scale is not interpreted as a biological timescale. Dynamical results compare spectral-radius, spectral-norm, and maximum-row-sum normalizations with `gamma=1`.
- Asymmetry, non-normality, reactivity, instability, and finite-time gain are computed and interpreted as distinct quantities.
- Stationary covariance is compared only at a coupling stable for both empirical and symmetrized matrices.

## Calculation spot-checks

- Spectral-norm-normalized `alpha(A)=0.0156070180` and `omega(A)=0.5023791216` were independently recomputed from SciPy eigenvalue routines and exactly match the saved summary.
- At `g=33.0321338`, direct SVDs of `exp(Jt)` at the optimized time and at ±0.1% perturbations confirm a local maximum of `Gmax=177.6449204` at `t*=1.1036987`.
- At `q=0`, the directionality interpolation returns zero commutator and Henrici indices and `Gmax=1`, as required for a symmetric normal matrix at 90% of its stability threshold.
- The finite-grid resolvent calculation first reaches the right half-plane at approximately `epsilon=0.0274`; this is an approximation, not a certified pseudospectral boundary.
- Empirical null p-values use the finite-sample correction `(1 + exceedances)/(n + 1)`.

## Visual review

All 11 PNG figures were inspected together. Axes, units, titles, signed color scaling, log gain scale, long node labels, and threshold markers are readable. The coupling phase diagram explicitly shades and labels stable/non-reactive, stable/reactive, and unstable regions.

The self-contained HTML report passed canonical artifact validation, packaging, and structural verification. Browser-level interaction and responsive-layout verification did not run because no compatible Chromium executable was installed; the semantic HTML fallback remains available.

## Required caveats

- Fifty spectral nulls per family and 20 gain nulls per family support exploratory empirical p-values, not high-resolution tail inference.
- Lesion `Gmax` is a five-time-point screen around the empirical peak; exact lesion spectral/numerical abscissae and thresholds are included, but top gain lesions should be fully re-optimized.
- Edge `alpha` and `omega` effects are first-order local sensitivities. Gain derivatives use matrix-exponential Frechet derivatives only for the top 40 screened edges.
- Large effects depend strongly on a concentrated extreme-weight tail and on relative coupling. Biological interpretation requires calibrated weights or an externally justified operating range.
- Mathematical sensitivity does not establish biological causation.
