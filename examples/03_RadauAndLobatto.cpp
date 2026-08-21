// 03 -- Gauss, Radau and Lobatto.
//
// Gauss puts every node strictly inside the interval.  Radau fixes one
// endpoint, Lobatto both.  Each fixed node costs one degree of exactness and
// buys a node you can rely on being exactly where you asked for it.

#include <GaussQuad/All>
#include <cmath>
#include <format>
#include <iostream>

int main() {
  using Real = double;
  const int n = 5;

  const auto gauss = GaussQuad::GaussLegendreQuadrature1D<Real>(n);
  const auto radau = GaussQuad::GaussRadauLegendreQuadrature1D<Real>(n);
  const auto lobatto = GaussQuad::GaussLobattoLegendreQuadrature1D<Real>(n);

  std::cout << std::format("  {}-point rules on [-1,1]:\n\n", n);
  std::cout << std::format("  {:>23} {:>23} {:>23}\n", "Gauss", "Radau",
                           "Lobatto");
  for (int i = 0; i < n; i++) {
    std::cout << std::format("  {:>23.16f} {:>23.16f} {:>23.16f}\n", gauss.X(i),
                             radau.X(i), lobatto.X(i));
  }

  // The fixed nodes are assigned exactly rather than left to the eigensolver.
  // These are equality comparisons on purpose: a node a few ulps past 1 is
  // outside the interval, which breaks a shared element boundary and anything
  // evaluating sqrt(1-x^2) or log(1-x) at a node.
  std::cout << "\n  fixed nodes are exactly the endpoints:\n";
  const auto say = [](const char* what, bool holds) {
    std::cout << std::format("    {:<26} {}\n", what, holds);
  };
  say("radau.X(0)     == -1", radau.X(0) == Real(-1));
  say("lobatto.X(0)   == -1", lobatto.X(0) == Real(-1));
  say("lobatto.X(n-1) == +1", lobatto.X(n - 1) == Real(1));
  say("gauss is open, |x| < 1", std::abs(gauss.X(0)) < Real(1));

  // The highest degree each rule integrates exactly, found by walking up the
  // monomials until the answer stops being right.
  const auto highest = [](const auto& q) {
    for (int k = 0; k < 40; k++) {
      const Real moment = (k % 2 == 0) ? Real(2) / (k + 1) : Real(0);
      const auto got = q.Integrate([k](Real x) { return std::pow(x, k); });
      if (std::abs(got - moment) > 1e-12) return k - 1;
    }
    return 39;
  };

  std::cout << std::format("\n  highest degree integrated exactly, n = {}:\n", n);
  std::cout << std::format("    Gauss     {:2d}    (2n-1 = {})\n", highest(gauss),
                           2 * n - 1);
  std::cout << std::format("    Radau     {:2d}    (2n-2 = {})\n", highest(radau),
                           2 * n - 2);
  std::cout << std::format("    Lobatto   {:2d}    (2n-3 = {})\n",
                           highest(lobatto), 2 * n - 3);

  // Radau and Lobatto exist for every weight the library supports, and
  // Gauss-Radau-Laguerre fixes the origin -- the natural endpoint for a
  // semi-infinite domain.
  const auto rl = GaussQuad::GaussRadauLaguerreQuadrature1D<Real>(4);
  std::cout << std::format("\n  Gauss-Radau-Laguerre fixes the origin: x[0] == 0 : {}\n",
                           rl.X(0) == Real(0));
}
