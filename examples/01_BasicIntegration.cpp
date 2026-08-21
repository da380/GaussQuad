// 01 -- Building a rule and integrating with it.
//
// The minimum you need: build a Gauss-Legendre rule, look at its points and
// weights, and integrate.  An n-point Gauss rule is exact for polynomials of
// degree up to 2n-1, which is the property everything else rests on.

#include <GaussQuad/All>
#include <cmath>
#include <format>
#include <iostream>
#include <version>

int main() {
  using Real = double;

  const int n = 5;
  const auto q = GaussQuad::GaussLegendreQuadrature1D<Real>(n);

  std::cout << std::format("A {}-point Gauss-Legendre rule on [-1,1]:\n\n",
                           q.N());
  for (int i = 0; i < q.N(); i++) {
    std::cout << std::format("  x[{}] = {:>23.16e}   w[{}] = {:>22.16e}\n", i,
                             q.X(i), i, q.W(i));
  }

  // The weights sum to the integral of the weight function.  For Legendre the
  // weight is one, so the sum is the length of the interval.
  Real sum = 0;
  for (auto w : q.Weights()) sum += w;
  std::cout << std::format("\n  sum of the weights = {:.17g}\n\n", sum);

  // Integrate.  The Legendre weight is one, so this is the plain integral of
  // f; for every other family it would be the integral of f against that
  // family's weight.  See example 04.
  const auto f = [](Real x) { return std::exp(x); };
  const Real exact = std::exp(Real(1)) - std::exp(Real(-1));
  std::cout << std::format("  int exp(x) dx over [-1,1] = {:.16f}\n", q.Integrate(f));
  std::cout << std::format("                      exact = {:.16f}\n", exact);

  // Exact to degree 2n-1, and no further.  Only even k is shown: the rule is
  // symmetric, so it integrates every odd power exactly whatever n is, and
  // those tell you nothing about the degree of exactness.
  std::cout << std::format("\n  error integrating x^k, exact for k <= {}:\n",
                           2 * n - 1);
  for (int k : {6, 8, 10, 12}) {
    const Real moment = Real(2) / (k + 1);
    const auto got = q.Integrate([k](Real x) { return std::pow(x, k); });
    std::cout << std::format("    k = {:2d}   error = {:.3e}\n", k,
                             std::abs(got - moment));
  }

#ifdef __cpp_lib_ranges_zip
  // Where the standard library has std::views::zip, the rule can be walked
  // directly as (point, weight) pairs.
  Real total = 0;
  for (auto [x, w] : q.Nodes()) total += w * f(x);
  std::cout << std::format("\n  the same integral through Nodes() = {:.16f}\n",
                           total);
#endif
}
