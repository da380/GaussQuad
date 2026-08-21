// 05 -- The two Gauss-Legendre algorithms.
//
// GolubWelsch is the default: O(n^2), available for every weight function,
// and its weight-sum error does not grow with n.  GlaserLiuRokhlin is O(n)
// and Legendre-only, and its weights drift as O(n eps).  At large n it is the
// faster rule rather than the better one.
//
// There is deliberately no automatic switch: a rule whose algorithm changes
// with n has a discontinuity in its node values at the switch, which is
// invisible until something downstream depends on it.

#include <GaussQuad/All>
#include <chrono>
#include <cmath>
#include <format>
#include <iostream>

int main() {
  using Real = double;
  using Clock = std::chrono::steady_clock;
  using Ms = std::chrono::duration<double, std::milli>;

  std::cout << std::format("  {:>6} {:>12} {:>12} {:>14} {:>14} {:>12}\n", "n",
                           "GW (ms)", "GLR (ms)", "GW |sum w-2|",
                           "GLR |sum w-2|", "max |dx|");

  for (int n : {100, 1000, 4000, 8000}) {
    const auto t0 = Clock::now();
    const auto gw = GaussQuad::GaussLegendreQuadrature1D<Real>(n);
    const auto t1 = Clock::now();
    const auto glr = GaussQuad::GaussLegendreQuadrature1D<Real>(
        n, GaussQuad::Method::GlaserLiuRokhlin);
    const auto t2 = Clock::now();

    const auto drift = [](const auto& q) {
      Real sum = 0;
      for (auto w : q.Weights()) sum += w;
      return std::abs(sum - Real(2));
    };

    // The two share no code -- one is an eigenproblem, the other a Newton
    // march over a local power series -- so their agreement is real evidence.
    Real maxdx = 0;
    for (int i = 0; i < n; i++) {
      maxdx = std::max(maxdx, std::abs(gw.X(i) - glr.X(i)));
    }

    std::cout << std::format("  {:>6} {:>12.3f} {:>12.3f} {:>14.2e} {:>14.2e} {:>12.2e}\n",
                             n, Ms(t1 - t0).count(), Ms(t2 - t1).count(),
                             drift(gw), drift(glr), maxdx);
  }

  // The O(n) algorithm has a precision ceiling.  Its weight is
  // 2/((1-x)(1+x)P'^2), and the outermost node lies about 1/n^2 from the
  // endpoint -- so at some n it rounds to the endpoint and the weight is not
  // representable.  In float that happens around n = 10^4.  It is diagnosed
  // rather than returned as an infinity.  GolubWelsch has no such limit.
  try {
    (void)GaussQuad::GaussLegendreQuadrature1D<float>(
        12000, GaussQuad::Method::GlaserLiuRokhlin);
    std::cout << "\n  float GLR at n = 12000 succeeded\n";
  } catch (const std::exception& e) {
    std::cout << std::format("\n  float GLR at n = 12000:\n    {}\n", e.what());
  }
}
