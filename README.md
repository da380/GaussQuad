# GaussQuad

A header-only C++23 template library for Gaussian quadrature.

Provides Gauss, Gauss–Radau and Gauss–Lobatto rules for the Jacobi weight
`(1-x)^a (1+x)^b` on `[-1,1]`, with Legendre (`a = b = 0`) and Chebyshev
(`a = b = -1/2`) as named special cases, and Gauss–Laguerre (`x^a e^{-x}` on
`[0,∞)`) and Gauss–Hermite (`e^{-x²}` on the whole line). Everything is
templated on the floating-point type, and `float`, `double` and `long double`
are all tested.

## Method

Nodes and weights come from the method of Golub and Welsch [[1]](#1): the nodes
are the eigenvalues of the symmetric tridiagonal Jacobi matrix of the
three-term recurrence, and the weights are `mu0 * q_i^2`, where `q_i` is the
first component of the `i`th normalised eigenvector and `mu0` the integral of
the weight function. Radau and Lobatto use Golub's modification [[2]](#2), which
fixes one or both endpoints by adjusting the trailing recurrence coefficients.

The eigenproblem is solved in `src/TridiagonalEigen.h` by implicit QL with
Wilkinson shifts, accumulating only the first row of the eigenvector matrix,
which is all that is needed. That makes the cost `O(n^2)` in time and `O(n)` in
storage, and — because the accumulated row is only ever acted on by plane
rotations, which preserve its norm — the weights sum to `mu0` to within a few
rounding errors at any degree.

There is no linear algebra dependency.

### Choosing the algorithm

For Gauss–Legendre specifically there is a second algorithm, selected with
`GaussQuad::Method`:

| | `GolubWelsch` (default) | `GlaserLiuRokhlin` |
|---|---|---|
| cost | `O(n²)` | `O(n)` |
| `\|Σw − 2\|` | flat, ~`10⁻¹⁵` at any `n` | drifts as `O(nε)` |
| `n = 1025` | 23 ms | 0.4 ms |
| `n = 16385` | 4.5 s | 2.4 ms |
| weights | any | Legendre only |

`GlaserLiuRokhlin` [[3]](#3) never evaluates the polynomial by recurrence: it
marches from root to root using the local power series that the Legendre
equation determines, at `O(1)` cost per node. Its nodes agree with
Golub–Welsch to about `10⁻¹⁵`, so at large `n` it is the faster rule rather
than the better one — pick it when construction cost matters.

There is deliberately **no automatic switch** between the two. A rule whose
algorithm changes with `n` is exactly the discontinuity this library used to
have.

```cpp
auto q = GaussQuad::GaussLegendreQuadrature1D<double>(
    n, GaussQuad::Method::GlaserLiuRokhlin);
```

## Use

```cpp
#include <GaussQuad/All>

auto q = GaussQuad::GaussLegendreQuadrature1D<double>(n);

// Map from [-1,1] to [a,b].  Note that the derivative is evaluated at the
// mapped points, so it must be given as a function of the new variable.
q.Transform([&](double x) { return 0.5 * ((b - a) * x + a + b); },
            [&](double)   { return 0.5 *  (b - a); });

auto integral = q.Integrate([](double x) { return std::exp(-x * x); });
```

Rules on `[-1,1]` can be placed on an arbitrary interval directly, which avoids
having to get the Jacobian convention right:

```cpp
auto q = GaussQuad::GaussLegendreQuadrature1D<double>(n, a, b);
auto r = GaussQuad::GaussLobattoLegendreQuadrature1D<double>(n, a, b);
```

`Quadrature1D` exposes `N()`, `X(i)`, `W(i)`, `Points()`, `Weights()`,
`Integrate(f)`, the in-place `Transform(f, df)` and its non-mutating
counterparts `Transformed(f, df)` and `MappedTo(a, b)`. Where the standard
library provides `std::views::zip`, `Nodes()` gives a view of the
(point, weight) pairs:

```cpp
for (auto [x, w] : q.Nodes()) total += w * f(x);
```

Note that the weight function belongs to the rule, not to the integrand:
`Integrate(f)` returns `∫ w(x) f(x) dx`, which is the plain integral of `f`
only for the Legendre rules.

Factory functions exist for the Gauss, Radau and Lobatto rules of the Legendre,
Chebyshev and general Jacobi weights, and for Gauss–Laguerre and Gauss–Hermite:

```cpp
auto l = GaussQuad::GaussLaguerreQuadrature1D<double>(n, alpha); // ∫₀^∞ xᵃe⁻ˣ f
auto h = GaussQuad::GaussHermiteQuadrature1D<double>(n);         // ∫₋∞^∞ e⁻ˣ² f
```

## Building

```
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

The only dependency is [NumericConcepts](https://github.com/da380/NumericConcepts),
which is found with `find_package` if installed and fetched otherwise.
Tests additionally fetch GoogleTest and
[Interpolation](https://github.com/da380/Interpolation).

To install and consume as a package:

```
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/your/prefix
cmake --install build
```

```cmake
find_package(GaussQuad REQUIRED)
target_link_libraries(your_target PRIVATE GaussQuad::GaussQuad)
```

`add_subdirectory` and `FetchContent` also work; set `GAUSSQUAD_BUILD_TESTS=OFF`
and `GAUSSQUAD_BUILD_EXAMPLES=OFF` to skip the developer targets, which are off
by default when GaussQuad is not the top-level project.

## References

<a id="1">[1]</a>
Golub, G. H. and Welsch, J. H., 1969.
Calculation of Gauss quadrature rules.
*Mathematics of Computation*, **23**, 221–230.

<a id="2">[2]</a>
Golub, G. H., 1973.
Some modified matrix eigenvalue problems.
*SIAM Review*, **15**, 318–334.

<a id="3">[3]</a>
Glaser, A., Liu, X. and Rokhlin, V., 2007.
A fast algorithm for the calculation of the roots of special functions.
*SIAM Journal on Scientific Computing*, **29**, 1420–1438.
