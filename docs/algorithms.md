# Algorithms and accuracy

Reference notes on how GaussQuad computes its rules, what the numerical
behaviour is, and why the implementation is arranged as it is. For how to
*use* the library, see the [README](../README.md).

Every figure quoted here was measured on the development machine with
`g++-14 -O2 -DNDEBUG`; treat the timings as ratios rather than absolutes.

---

## 1. The rules

An `n`-point quadrature rule approximates a weighted integral by a finite sum,

```
∫ w(x) f(x) dx  ≈  Σ_{i=1}^{n} w_i f(x_i).
```

The weight function `w` belongs to the rule, not to the integrand: for a
Chebyshev rule, `Integrate(f)` returns an approximation to
`∫ (1-x²)^(-1/2) f(x) dx`, not to `∫ f(x) dx`. Only the Legendre rules, whose
weight is one, integrate `f` alone.

| rule | weight | interval | exact for degree |
|---|---|---|---|
| Gauss–Jacobi | `(1-x)^α (1+x)^β` | `[-1,1]` | `2n − 1` |
| Gauss–Legendre | `1` | `[-1,1]` | `2n − 1` |
| Gauss–Chebyshev | `(1-x²)^(-1/2)` | `[-1,1]` | `2n − 1` |
| Gauss–Laguerre | `x^α e^(-x)` | `[0,∞)` | `2n − 1` |
| Gauss–Hermite | `e^(-x²)` | `(-∞,∞)` | `2n − 1` |
| Gauss–Radau | as above, one node fixed at an endpoint | | `2n − 2` |
| Gauss–Lobatto | as above, both endpoints fixed | | `2n − 3` |

The Jacobi weight is integrable only for `α, β > -1`, and the Laguerre weight
only for `α > -1`; a constructor given anything else throws.

---

## 2. Golub–Welsch

Every family of polynomials orthogonal with respect to a weight satisfies a
three-term recurrence, which in its symmetric normalisation reads

```
x p_{k-1}(x) = e_{k-1} p_{k-2}(x) + d_k p_{k-1}(x) + e_k p_k(x).
```

Collecting the coefficients gives the symmetric tridiagonal *Jacobi matrix*
`J_n`, with `d_1 … d_n` on the diagonal and `e_1 … e_{n-1}` off it. Golub and
Welsch [1] showed that the nodes of the `n`-point Gauss rule are exactly
the eigenvalues of `J_n`, and that the weights are

```
w_i = μ₀ q_i²,
```

where `q_i` is the **first component** of the `i`th normalised eigenvector and
`μ₀ = ∫ w(x) dx` is the zeroth moment of the weight. The whole rule therefore
follows from one symmetric tridiagonal eigenproblem.

For the Jacobi weight,

```
μ₀ = 2^(α+β+1) Γ(α+1) Γ(β+1) / Γ(α+β+2),
```

computed through `lgamma` so that it neither overflows nor underflows.

---

## 3. The eigensolver

`src/TridiagonalEigen.h` solves the eigenproblem by implicit QL with Wilkinson
shifts — the classic `tql2`/`imtql2` iteration — with one change: rather than
accumulating the full eigenvector matrix, it accumulates **only its first
row**. That is all Golub–Welsch asks for, and it takes the cost from `O(n³)`
time and `O(n²)` storage down to `O(n²)` and `O(n)`.

The change also fixes the accuracy of the weights, for a structural reason.
The accumulated row starts as the first row of the identity, and every
operation applied to it thereafter is a plane rotation, which preserves the
Euclidean norm. So `Σ q_i² = 1` to within a few rounding errors however many
sweeps the iteration takes, and hence

```
Σ w_i = μ₀
```

to the same accuracy — flat in `n` rather than growing with it. Measured for
Gauss–Legendre, where `μ₀ = 2`:

| `n` | `\|Σw − 2\|`, `float` | `double` | `long double` |
|---|---|---|---|
| 65 | `2.4e-07` | `1.6e-15` | `8.7e-19` |
| 101 | `2.4e-07` | `1.8e-15` | `6.5e-19` |
| 1025 | `2.4e-07` | `4.0e-15` | `1.1e-19` |
| 4097 | `1.2e-06` | `6.2e-15` | `3.3e-19` |
| 16385 | `6.4e-06` | `8.9e-16` | `1.3e-18` |

The iteration is bounded at 50 sweeps per eigenvalue and throws
`std::runtime_error` if it does not converge; nothing observed comes close.

---

## 4. Radau and Lobatto

Golub's modification [2] fixes one or both endpoints by perturbing the
trailing entries of the Jacobi matrix, so that the same eigensolver produces
the constrained rule.

Write `v` for the last coordinate vector of length `n−1`.

For **Radau**, with the node fixed at `x₁`: solve

```
(J_{n-1} − x₁ I) δ = e_{n-1}² v
```

and replace the trailing diagonal entry by `d_n = x₁ + δ_{n-1}`.

For **Lobatto**, with nodes fixed at `x₁` and `x₂`: solve
`(J_{n-1} − x₁ I) γ = v` and `(J_{n-1} − x₂ I) σ = v`, then set

```
e_{n-1}² = (x₂ − x₁) / (γ_{n-1} − σ_{n-1}),   d_n = x₁ + γ_{n-1} e_{n-1}².
```

Both shifted matrices are definite — the shift is an endpoint of the support
of the weight, so it lies outside the spectrum — and so the systems are solved
by unpivoted LU, the Thomas algorithm, in `O(n)`.

Gauss–Radau–Laguerre is the same construction with `x₁ = 0`, where the shift
vanishes and the system is `J_{n-1}` itself, positive definite because the
Laguerre nodes are positive.

The fixed nodes are then **assigned exactly** rather than left to the
eigensolver, which returns them only to within rounding — and a node a few
ulps past `1` is outside the interval it is supposed to lie in. Code that
shares element boundaries between adjacent intervals, or that evaluates
`√(1-x²)` or `log(1-x)` at a node, needs the endpoint to be the endpoint.

---

## 5. Removable singularities in the Jacobi coefficients

Two of the general Jacobi formulae are `0/0` or `∞ − ∞` at `α + β = -1`, which
is exactly the Chebyshev case, and both have to be cancelled by hand.

**The off-diagonal coefficient.** In general

```
e_k² = 4k (k+α) (k+β) (k+α+β) / [(2k+α+β−1) (2k+α+β)² (2k+α+β+1)].
```

At `k = 1` the factor `(k+α+β)` in the numerator cancels against
`(2k+α+β−1)` in the denominator. The cancellation is removable in general,
but both factors vanish when `α + β = -1`, so evaluating the unsimplified
expression there gives `0/0`. Cancelled,

```
e_1² = 4 (1+α) (1+β) / [(2+α+β)² (3+α+β)].
```

**The zeroth moment.** `Γ(α+β+2)` must be taken directly rather than expanded
as `(α+β+1) Γ(α+β+1)`: at `α + β = -1` the expanded form is `0 · ∞` in the
gamma and `∞ − ∞` in the logarithm.

Either one alone turns every weight of every Chebyshev rule into a `NaN`, and
a `NaN` is invisible to a tolerance test — a comparison against it is simply
false. The test suite checks explicitly that no node or weight is non-finite.

---

## 6. Symmetry

When `α = β` the weight function is even, so the rule is symmetric about the
origin; likewise the Hermite rule. The eigensolver does not know this and
returns a rule symmetric only to within rounding.

The symmetry is imposed exactly, by averaging the two halves. This costs five
lines, and because it is an average it *halves* the error rather than adding
to it. Downstream code can rely on `x_i = −x_{n+1-i}` and `w_i = w_{n+1-i}`
holding bitwise.

---

## 7. Glaser–Liu–Rokhlin

For Gauss–Legendre specifically there is a second algorithm, `src/GlaserLiuRokhlin.h`,
selected with `GaussQuad::Method::GlaserLiuRokhlin`. It computes the rule in
`O(n)` time and never evaluates the polynomial by recurrence.

`P_n` satisfies the Legendre equation `(1-x²)y'' − 2xy' + n(n+1)y = 0`. Write
`u(h) = P_n(x + h)` about some point `x`, and put `c = 1 − x²`, `b = −2x`.
The equation becomes

```
(c + bh − h²) u'' + (b − 2h) u' + λu = 0,    λ = n(n+1),
```

and collecting powers of `h` gives the Taylor recurrence

```
u_{m+2} = −[ b (m+1)² u_{m+1} + (λ − m(m+1)) u_m ] / [ c (m+2)(m+1) ].
```

So the entire local behaviour of `P_n` follows from the pair `(u₀, u₁)` at a
single point, at `O(1)` cost. Starting from the exactly known values at
`x = 0`, each root is found by Newton on that local series and then becomes
the expansion point for the next — at a root `u₀ = 0`, so only `P_n'(x_k)` has
to be carried forward. Marching over the roots in `(0,1)` and reflecting gives
all `n` of them in `O(n)` work. The weights follow from
`w = 2 / ((1-x)(1+x) P_n'(x)²)`.

Three points are not obvious from the paper and cost time to rediscover.

- **For even `n` the first march step is a half spacing.** The roots are
  nearly equally spaced in `θ = arccos x`, but for even `n` they straddle
  `θ = π/2`, which is where the seed sits.

- **The Taylor coefficients must be scaled.** Unscaled they grow like
  `(λ/c)^(m/2)`, and for the outer roots — where `c = 1 − x²` is small — this
  overflows the exponent range long before the series stops being accurate. It
  is fatal in `float` and costs digits in `double`. Building the series in the
  scaled variable `t = h / step` makes every coefficient `O(1)`.

- **The algorithm has a precision ceiling.** The outermost node lies about
  `1/n²` from the endpoint, so at large enough `n` it rounds to the endpoint
  exactly, `(1-x)(1+x)` is zero, and the weight is not representable. In
  `float` this happens by `n ≈ 10⁴`. The condition is detected and throws
  `std::runtime_error` naming `GolubWelsch`, which computes its weights from
  the eigenvector row and has no such limit. In `double` and `long double` the
  ceiling is far beyond any usable degree.

**Bogaert's asymptotic expansions were considered and not used.** They are
faster still, but they are tables of twenty-digit fitted coefficients that
could not be reproduced with confidence, and they are `double`-only, which
would have cost the `float`/`long double` genericity. Every formula in
Glaser–Liu–Rokhlin is derivable from the Legendre equation itself, and works
at every precision.

---

## 8. Choosing between the two

| `n` | `GolubWelsch` | `\|Σw − 2\|` | `GlaserLiuRokhlin` | `\|Σw − 2\|` |
|---|---|---|---|---|
| 65 | 0.11 ms | `1.6e-15` | 0.03 ms | `7.5e-15` |
| 1025 | 19 ms | `4.0e-15` | 0.15 ms | `2.7e-15` |
| 4097 | 286 ms | `6.2e-15` | 0.56 ms | `6.8e-14` |
| 16385 | 4.2 s | `8.9e-16` | 2.3 ms | `1.2e-13` |

The nodes agree between the two to about `1e-15`. The weights do not: GLR's
drift as `O(nε)` where Golub–Welsch's stay flat, because they are formed from
the marched derivative rather than from a norm-preserving rotation. **At large
`n`, GLR is the faster rule rather than the better one.**

`GolubWelsch` is the default, it is the only option for weights other than
Legendre, and there is deliberately **no automatic crossover** between the
two. A rule whose algorithm changes with `n` has a discontinuity in its node
values at the switch, which is invisible until something downstream depends on
it.

---

## 9. Zeros by Newton

`JacobiPolynomial::Zeros` finds the roots of `P_n^(α,β)` by Newton's method
with Maehly deflation, seeded from the Chebyshev points. Those roots are
exactly the Gauss nodes, so this is a second, wholly independent route to
them, sharing no code with the eigensolver.

Measured, it agrees with `GaussQuadrature` to `1.6e-15` at `n = 1025` and
takes about seven times as long, so it is the slower route, not the less
accurate one. Prefer `GaussQuadrature` whenever the weights are wanted too.

It is kept because the test suite checks the two against each other, and two
implementations that share no code and agree is evidence that neither is
wrong. The deflation sum must be accumulated in `Real`: in integer arithmetic
it truncates to zero, deflation silently does nothing, and Newton can return
to a root already found — which yields a plausible-looking set of nodes that
is wrong.

---

## 10. Error handling

Anything that **builds** a rule validates its arguments and throws
`std::invalid_argument`: a degree that is too small, points and weights of
differing length, a weight function that is not integrable. Such a call
happens once, and a bad argument otherwise yields a rule that is silently
wrong rather than obviously wrong.

The pure **evaluation** functions — `operator()`, `Derivative` — keep
assertions instead. They sit in inner loops, and a negative degree there is a
programming error rather than bad data.

Failures of the numerics themselves — a QL iteration that will not converge, a
GLR march that produces an invalid rule or outruns the precision — throw
`std::runtime_error`.

The distinction matters because everything depending on this library is
compiled with `NDEBUG`, where an assertion is not a diagnosis of anything.

---

## 11. Dependencies

The library depends on [NumericConcepts](https://github.com/da380/NumericConcepts)
and nothing else. There is no linear algebra dependency: the tridiagonal
eigensolver and the Thomas solve are the only linear algebra needed, and both
are a few dozen lines. The installed `GaussQuadTargets.cmake` links
`NumericConcepts::NumericConcepts` alone.

The tests additionally fetch GoogleTest.

---

## References

**[1]**  
Golub, G. H. and Welsch, J. H., 1969.
Calculation of Gauss quadrature rules.
*Mathematics of Computation*, **23**, 221–230.

**[2]**  
Golub, G. H., 1973.
Some modified matrix eigenvalue problems.
*SIAM Review*, **15**, 318–334.

**[3]**  
Glaser, A., Liu, X. and Rokhlin, V., 2007.
A fast algorithm for the calculation of the roots of special functions.
*SIAM Journal on Scientific Computing*, **29**, 1420–1438.
