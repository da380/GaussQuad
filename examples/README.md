# Examples

Each is standalone, short, and covers one aspect. They are numbered in the
order a new reader should meet them; nothing after 01 depends on having read
the others.

| | | |
|---|---|---|
| 01 | [`BasicIntegration`](01_BasicIntegration.cpp) | Build a rule, look at its points and weights, integrate. Exactness to degree `2n-1`. |
| 02 | [`Intervals`](02_Intervals.cpp) | Rules somewhere other than `[-1,1]`: the interval overloads, `MappedTo`, and the `Transform` convention — including what going wrong looks like. |
| 03 | [`RadauAndLobatto`](03_RadauAndLobatto.cpp) | Fixing one endpoint or both, what each costs in exactness, and that the fixed nodes are exactly the endpoints. |
| 04 | [`WeightFunctions`](04_WeightFunctions.cpp) | Chebyshev, Jacobi, Laguerre, Hermite — and that the weight belongs to the rule, not to the integrand. |
| 05 | [`Algorithms`](05_Algorithms.cpp) | `GolubWelsch` against `GlaserLiuRokhlin`: cost, accuracy, and the precision ceiling of the `O(n)` one. |
| 06 | [`Polynomials`](06_Polynomials.cpp) | The polynomial classes on their own: evaluation, derivatives, zeros, orthogonality. |
| 07 | [`PrecisionAndValueTypes`](07_PrecisionAndValueTypes.cpp) | `float`, `double`, `long double`; complex- and vector-valued integrands. |
| 08 | [`ErrorHandling`](08_ErrorHandling.cpp) | What throws, what asserts, and why the difference matters under `NDEBUG`. |

They are built by default when GaussQuad is the top-level project, and land in
`build/bin`:

```
cmake -S . -B build
cmake --build build
./build/bin/01_BasicIntegration
```

Set `-DGAUSSQUAD_BUILD_EXAMPLES=OFF` to skip them.
