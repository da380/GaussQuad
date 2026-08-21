// 04 -- The weight function belongs to the rule, not to the integrand.
//
// Integrate(f) returns int w(x) f(x) dx.  It is the plain integral of f only
// for the Legendre rules, whose weight is one.  This is the easiest thing to
// get wrong when moving off Legendre, so it is worth seeing directly.

#include <GaussQuad/All>
#include <cmath>
#include <format>
#include <iostream>
#include <numbers>

int main() {
  using Real = double;
  const int n = 12;

  // Integrating the constant 1 recovers the integral of the weight itself.
  const auto one = [](Real) { return Real(1); };
  const auto pi = std::numbers::pi_v<Real>;

  std::cout << "  Integrate(1) returns the integral of the weight:\n\n";
  std::cout << std::format("    {:<12} {:<20} {:>20} {:>20}\n", "rule",
                           "weight", "Integrate(1)", "exact");

  const auto legendre = GaussQuad::GaussLegendreQuadrature1D<Real>(n);
  std::cout << std::format("    {:<12} {:<20} {:>20.15f} {:>20.15f}\n",
                           "Legendre", "1", legendre.Integrate(one), Real(2));

  const auto chebyshev = GaussQuad::GaussChebyshevQuadrature1D<Real>(n);
  std::cout << std::format("    {:<12} {:<20} {:>20.15f} {:>20.15f}\n",
                           "Chebyshev", "(1-x^2)^(-1/2)",
                           chebyshev.Integrate(one), pi);

  const auto hermite = GaussQuad::GaussHermiteQuadrature1D<Real>(n);
  std::cout << std::format("    {:<12} {:<20} {:>20.15f} {:>20.15f}\n",
                           "Hermite", "exp(-x^2)", hermite.Integrate(one),
                           std::sqrt(pi));

  const auto laguerre = GaussQuad::GaussLaguerreQuadrature1D<Real>(n);
  std::cout << std::format("    {:<12} {:<20} {:>20.15f} {:>20.15f}\n",
                           "Laguerre", "exp(-x)", laguerre.Integrate(one),
                           Real(1));

  // The general Jacobi weight, of which Legendre and Chebyshev are the named
  // special cases.  Its integral is 2^(a+b+1) B(a+1, b+1).
  const Real alpha = 1.5, beta = -0.5;
  const auto jacobi = GaussQuad::GaussJacobiQuadrature1D<Real>(n, alpha, beta);
  const Real mu0 = std::exp(std::lgamma(alpha + 1) + std::lgamma(beta + 1) -
                            std::lgamma(alpha + beta + 2) +
                            (alpha + beta + 1) * std::numbers::ln2_v<Real>);
  std::cout << std::format("    {:<12} {:<20} {:>20.15f} {:>20.15f}\n", "Jacobi",
                           "(1-x)^1.5 (1+x)^-.5", jacobi.Integrate(one), mu0);

  // So a Chebyshev rule does NOT give the plain integral: the 1/sqrt(1-x^2)
  // is already there, and the integrand supplies only what multiplies it.
  std::cout << std::format(
      "\n  int (1-x^2)^(-1/2) x^2 dx = pi/2 = {:.15f}\n"
      "  Chebyshev rule on f = x^2:         {:.15f}\n",
      pi / 2, chebyshev.Integrate([](Real x) { return x * x; }));

  // Laguerre integrates x^k exp(-x) over [0, inf), which is Gamma(k+1) = k!.
  std::cout << "\n  Gauss-Laguerre moments, int_0^inf x^k exp(-x) dx = k!:\n";
  for (int k : {0, 1, 5, 10}) {
    const auto got = laguerre.Integrate([k](Real x) { return std::pow(x, k); });
    std::cout << std::format("    k = {:2d}  {:>18.8e}   exact {:>18.8e}\n", k,
                             got, std::tgamma(Real(k + 1)));
  }
}
