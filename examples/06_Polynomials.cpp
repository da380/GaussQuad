// 06 -- The orthogonal polynomials themselves.
//
// The quadrature rules come from these, and the classes are usable on their
// own: evaluation, derivatives and zeros.  Include <GaussQuad/Polynomial>
// rather than <GaussQuad/All> if the rules are not wanted.

#include <GaussQuad/All>
#include <cmath>
#include <format>
#include <iostream>
#include <numbers>

int main() {
  using Real = double;

  const auto P = GaussQuad::LegendrePolynomial<Real>{};
  const auto T = GaussQuad::ChebyshevPolynomial<Real>{};

  // Evaluation and differentiation, by recursion in the degree.
  std::cout << "  Legendre P_n and P_n' at x = 0.3:\n";
  for (int n = 0; n <= 4; n++) {
    std::cout << std::format("    n = {}   P = {:>10.6f}   P' = {:>10.6f}\n", n,
                             P(n, Real(0.3)), P.Derivative(n, Real(0.3)));
  }

  // Chebyshev polynomials satisfy T_n(cos t) = cos(n t) exactly.
  std::cout << "\n  Chebyshev against T_n(cos t) = cos(n t):\n";
  const Real t = 0.7;
  for (int n : {3, 7, 12}) {
    const auto got = T(n, std::cos(t));
    std::cout << std::format("    n = {:2d}   {:>12.9f}   {:>12.9f}\n", n, got,
                             std::cos(n * t));
  }

  // The zeros of P_n are the nodes of the n-point Gauss rule.  Zeros() finds
  // them by Newton with Maehly deflation, sharing no code with the
  // eigensolver behind GaussQuadrature -- two independent routes to the same
  // numbers.  Zeros() is the slower of the two and does not give the weights.
  const int n = 20;
  const auto zeros = P.Zeros(n);
  const auto nodes = P.GaussQuadrature(n).first;
  Real worst = 0;
  for (int i = 0; i < n; i++) {
    worst = std::max(worst, std::abs(zeros[i] - nodes[i]));
  }
  std::cout << std::format(
      "\n  zeros of P_{} against the Gauss nodes: max difference {:.2e}\n", n,
      worst);

  // Orthogonality, checked with the rule the polynomials generate:
  //   int_{-1}^{1} P_j P_k dx = 2/(2k+1) delta_jk.
  const auto q = GaussQuad::GaussLegendreQuadrature1D<Real>(16);
  std::cout << "\n  int P_j P_k dx over [-1,1]:\n";
  std::cout << std::format("    {:>6}", "");
  for (int k = 0; k <= 4; k++) std::cout << std::format("{:>12}", k);
  std::cout << "\n";
  for (int j = 0; j <= 4; j++) {
    std::cout << std::format("    j = {:2d}", j);
    for (int k = 0; k <= 4; k++) {
      std::cout << std::format("{:>12.8f}", q.Integrate([&](Real x) {
        return P(j, x) * P(k, x);
      }));
    }
    std::cout << "\n";
  }
  std::cout << "\n  the diagonal is 2/(2k+1): 2, 0.6667, 0.4, 0.2857, 0.2222\n";
}
