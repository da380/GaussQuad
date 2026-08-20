# GaussQuad: status and remaining work

Originally a hand-over document assessing `da380/GaussQuad@main` as fetched on
2026-08-20. **Rewritten 2026-08-20** after the work below was carried out, and
updated again as the follow-up list was worked through.
Everything quoted was measured on this machine, not inferred.

Baseline for comparison is commit `fbe37c7`, and the release `v1.0.0` is
tagged, so anything needing the old node values can pin it.

---

## 1. What was done

### Algorithm: Eigen removed, one path for all rules

The `n <= 100` Golub–Welsch / `n > 100` Newton split is gone. Both were
replaced by `src/TridiagonalEigen.h`: implicit QL with Wilkinson shifts on the
Jacobi matrix, accumulating **only the first row** of the eigenvector matrix,
which is all Golub–Welsch needs. `O(n^2)` time and `O(n)` storage instead of
`O(n^3)` and `O(n^2)`.

The accuracy gain is structural rather than a tuning: the accumulated row is
only ever acted on by plane rotations, which preserve its norm, so the weights
sum to `mu0` to within a few rounding errors at any degree.

One algorithm now serves Gauss, Radau, Lobatto and every Jacobi weight, so
**the Eigen dependency is gone entirely** — not just for Gauss–Legendre, as
the original plan assumed would be necessary. The two `llt().solve()` calls in
Radau/Lobatto became a 15-line Thomas solve.

| `n` | `\|Sum(w) - 2\|` before | after | build before | after |
|---|---|---|---|---|
| 65 | `5.3e-16` | `1.5e-15` | 0.5 ms | 0.1 ms |
| 101 | `1.17e-13` | `1.4e-15` | 2.4 ms | 0.3 ms |
| 1025 | `3.4e-13` | `3.0e-15` | 152 ms | 23 ms |
| 2049 | `4.0e-12` | `2.3e-15` | 629 ms | 79 ms |
| 4097 | `3.3e-12` | `2.7e-15` | 2620 ms | 305 ms |

The accuracy no longer depends on `n`. `long double` came out *better* than
before (`2.2e-19` at `n = 1025` against `1.8e-16`), so nothing was traded away
to get this.

### Second algorithm: O(n) Gauss–Legendre

`src/GlaserLiuRokhlin.h`, selected with `GaussQuad::Method`.

**Bogaert was not used.** Its expansions are tables of 20-digit fitted
coefficients that could not be reproduced with confidence, and it is
double-only, which would have cost the `float`/`long double` genericity.
Glaser–Liu–Rokhlin (2007) reaches the same `O(n)` with every formula derivable
from the Legendre equation itself, and works at every precision.

| `n` | Golub–Welsch | `\|Sum(w)-2\|` | GLR | `\|Sum(w)-2\|` |
|---|---|---|---|---|
| 1025 | 21.9 ms | `3.0e-15` | 0.16 ms | `1.7e-15` |
| 4097 | 306 ms | `2.7e-15` | 0.63 ms | `7.1e-14` |
| 16385 | 4498 ms | `6.4e-15` | 2.27 ms | `1.1e-13` |

Nodes agree with Golub–Welsch to about `1e-15`. **The weights drift as
`O(n*eps)` where Golub–Welsch stays flat**, so at large `n` GLR is the faster
rule, not the better one. `Method::GolubWelsch` is the default and there is
deliberately no automatic crossover.

Two things that were not obvious and would cost a re-implementer time:

- for even `n` the first march step is a **half** spacing, because the roots
  straddle `theta = pi/2`;
- the Taylor coefficients grow like `(lambda/c)^(m/2)` and **overflow in
  float** long before the series stops being accurate. Scaling the series to
  `t = h/step` fixes it, and improved `double` as well.

### Bugs fixed beyond the original list

- **Every Gauss–Chebyshev rule returned silent `NaN`.** Two independent
  causes: `E(1)` is `0/0` when `a + b = -1`, and `Mu()` expanded
  `Gamma(a+b+2)` as `(a+b+1)Gamma(a+b+1)`, which is `inf - inf` at exactly the
  same place. Confirmed against the baseline — all six weights came back
  `nan`. Both are removable singularities, cancelled by hand.
- **Radau and Lobatto endpoints were not exactly `+-1`** (at `n = 20`,
  `x_max = 1.0000000000000036`, outside the interval). Now assigned exactly.
  This is the one that matters for shared element boundaries in the radial SEM
  bases, and for anything evaluating `sqrt(1-x^2)` or `log(1-x)` at a node.
- **`ChebyshevPolynomial::operator()` and `Derivative` never compiled** —
  non-const `Scale` called from a const member, `lgamma<Real>` (`std::lgamma`
  is not a template), and an unreachable bare `return;` in a function
  returning `Real`.
- `Real sum = 0` in `Zeros` (the original plan's item 2).
- `Integrate` was not `const`, so `const Quadrature1D` was unusable.
- `assert(n > 2)` rejected the 2-point Lobatto rule, which is the trapezoid
  rule and works fine.
- Unqualified `pow` relying on `::pow` leaking from `<cmath>`; copy instead of
  move in the `Quadrature1D` constructor.
- **Symmetry is now exact** when `alpha == beta`. The eigensolver does not know
  the rule is symmetric; imposing it costs five lines and halves the error
  rather than adding to it. Downstream code can rely on it.

### API additions (all additive; every existing call site is unchanged)

- `GaussQuad::Method` on `LegendrePolynomial::GaussQuadrature` and
  `GaussLegendreQuadrature1D`.
- **Gauss–Laguerre and Gauss–Hermite** — the obvious gap, given the machinery
  was already there. Verified against closed forms and against
  `Sum(w x^k) = Gamma(a+k+1)` and `Sum(w x^2m) = Gamma(m+1/2)` to `1e-13`.
- **Interval overloads**: `GaussLegendreQuadrature1D<Real>(n, a, b)`, likewise
  Radau and Lobatto. This is the real fix for the `Transform` trap — the
  common case no longer requires getting the Jacobian convention right.
- `Quadrature1D::Transformed`, `MappedTo`, and `Nodes()` (a `std::views::zip`
  view, guarded on `__cpp_lib_ranges_zip`).
- Documented that the weight function belongs to the rule, not the integrand:
  `Integrate(f)` is `int w(x) f(x) dx`, which is the plain integral of `f` only
  for the Legendre rules. This was an undocumented trap for Chebyshev and
  Jacobi users.

### Tests

65 tests over `float`, `double` and `long double`, from 12. Identity-based, so
they need no reference implementation: weight sums, orthogonality at the exact
highest degree each rule reaches, strict monotonicity, positivity, finiteness,
exact endpoints, exact symmetry, closed forms for Chebyshev/Laguerre/Hermite,
and cross-agreement between the two independent Gauss–Legendre algorithms.
Degrees straddle the old `n = 100` crossover deliberately.

**They were validated by running them against the baseline**, where they catch
the `Sum(w)` drift, the endpoint inexactness, out-of-range nodes, the Chebyshev
`NaN` and the over-strict assert. The baseline is also slow enough that the
suite times out at two minutes where the current code takes 2.4 s.

Two bugs in the *test reference values* were found and fixed along the way: the
`lgamma` closed form for the squared norm loses two digits to cancellation by
`l = 100`, and hits the same `0/0` as the library did.

### Build

`find_package(GaussQuad)` now works — verified end to end by installing to a
prefix and building a consumer that calls all six GSHTrans entry points.
**This unblocks GSHTrans shipping a CMake package.**

Also: `${INCLUDE_INSTALL_DIR}` was never defined anywhere, so the
`INSTALL_INTERFACE` was already empty independently of there being no
`install()`; `CMAKE_CXX_STANDARD` moved onto the target so it stops leaking
into parent projects; `GaussQuad::GaussQuad` alias added; `MY_PROJECT_*`
options renamed; googletest pinned; the committed emacs autosave
`examples/#CMakeLists.txt#` deleted.

---

### Follow-up work

- **Argument checking.** Anything that builds a rule now validates and throws
  `std::invalid_argument`; the evaluation functions keep assertions. Previously
  all of it was `assert`, so in a release build -- which is how everything
  depending on this is compiled -- a bad degree or a non-integrable weight was
  silently undefined. The tests for it are built with `NDEBUG` so they exercise
  the release configuration.
- **Gauss-Radau-Laguerre**, node fixed at the origin. Golub's modification with
  `x1 = 0`, so the shifted system is `J_m` itself, positive definite because
  the Laguerre nodes are. Checked against the closed form at `n = 2` -- nodes
  `{0,2}`, weights `{1/2,1/2}` -- and against the moment identity to `2n-2`.
- **The `Integrable` concept** required `f*w` and `f+f` to be exactly
  `FunctionValue`, which rejected every expression-template integrand (an
  Eigen vector returns a proxy from those operators). Now convertibility;
  types with no arithmetic at all are still rejected.
- **NumericConcepts pinned** to `888126b` rather than tracking `main`, through
  the overridable `GAUSSQUAD_NUMERICCONCEPTS_TAG`. Only affects the fetch
  fallback -- `find_package` still prefers an installed copy.
- **CI** (`.github/workflows/ci.yml`): build and test under g++-13 and g++-14
  in Debug and Release, plus a job that installs to a prefix and builds a
  consumer resolving GaussQuad only through `find_package`, so a broken export
  set fails there rather than downstream.

---

## 2. Still open

Nothing here is blocking. Items resolved since the first revision are recorded
with the decision, so they are not reopened by accident.

### Decided, deliberately not changed

- **Version stays `1.1.0`.** The API is unchanged and every addition is
  additive. Node values do move in the last few ulps and the Chebyshev rules go
  from `NaN` to correct, which is the argument for `2.0.0`; `v1.0.0` is tagged,
  so anything needing the old values can pin it.
- **`Transform` keeps its Jacobian convention.** The points are mapped first
  and `df` is evaluated at the *mapped* points, so it must be a function of the
  new variable -- verified with `y = x^3`, where supplying `df` of the old
  variable gives `0.4615` instead of `0.6667`. It is documented in place, and
  the interval factories mean the common case never meets it. Deliberately
  **not** marked `[[deprecated]]`: GSHTrans calls it, and that would fill its
  build with warnings for no gain.
- **GLR's `O(n*eps)` weight drift stays.** It comes from forming
  `w = 2/((1-x^2) P'^2)` from the marched derivative. It could be hidden by
  rescaling the weights to sum to `mu0` exactly, but that would also make the
  weight-sum test vacuous for that algorithm, which is too high a price for a
  cosmetic fix. `GolubWelsch` is the default and does not drift.
- **`Zeros` keeps its Newton implementation.** Measured, it agrees with
  `GaussQuadrature` to `1.6e-15` at `n = 1025` -- so it is *slower*, about
  sevenfold, not less accurate, and the header comment claiming otherwise has
  been corrected. It is now the only code path in the library that does not go
  through the eigensolver, and the suite checks the two against each other:
  two implementations sharing no code and agreeing is evidence neither is
  wrong. That is worth more than the lines it costs.

### Genuinely open

1. **Interpolation is still fetched at `GIT_TAG main`.** Test-only, and it has
   no tags to pin to. Pin when it gets a release.
2. **CI covers only g++-13 and g++-14**, because no clang is installed on the
   development machine and an unverified matrix leg would just fail on first
   push. Adding `clang++-18` is a one-line change once it can be tested. The
   `Nodes()` view is already guarded on `__cpp_lib_ranges_zip`, so a standard
   library without `std::views::zip` degrades rather than breaks.
3. **Downstream.** GSHTrans can now be given install and export rules of its
   own, which this library was blocking. That is the next piece of work, and it
   is in the other repository.
4. **Not tagged.** The work is committed on `develop`; no release tag has been
   made and `main` has not been updated.
