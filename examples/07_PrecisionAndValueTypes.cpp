// 07 -- Genericity: any floating-point precision, any integrand value type.
//
// Everything is templated on the floating-point type, and the integrand may
// return anything closed under addition and under multiplication by that
// type -- a complex number, a vector, an expression-template proxy.

#include <GaussQuad/All>
#include <cmath>
#include <complex>
#include <concepts>
#include <format>
#include <iostream>
#include <limits>

// The rule is built at whatever precision it is asked for, and delivers it:
// the weight-sum error tracks the epsilon of the type rather than the degree.
template <std::floating_point Real>
void Report(const char* name) {
  const auto q = GaussQuad::GaussLegendreQuadrature1D<Real>(1000);
  Real sum = 0;
  for (auto w : q.Weights()) sum += w;
  std::cout << std::format("    {:<12} digits = {:2d}   eps = {:.2e}   "
                           "|sum w - 2| = {:.2e}\n",
                           name, std::numeric_limits<Real>::digits,
                           std::numeric_limits<Real>::epsilon(),
                           std::abs(sum - Real(2)));
}

// An integrand returning a value type that is not a scalar.  Vec2 is written
// so that its arithmetic returns a proxy rather than a Vec2, as an
// expression-template library's would, to show that such types are accepted.
struct Proxy {
  double a, b;
};
struct Vec2 {
  double a{}, b{};
  Vec2() = default;
  Vec2(double x, double y) : a{x}, b{y} {}
  Vec2(Proxy p) : a{p.a}, b{p.b} {}
};
Proxy operator*(Vec2 v, double s) { return {v.a * s, v.b * s}; }
Proxy operator+(Vec2 u, Vec2 v) { return {u.a + v.a, u.b + v.b}; }

int main() {
  std::cout << "  a 1000-point Gauss-Legendre rule at three precisions:\n\n";
  Report<float>("float");
  Report<double>("double");
  Report<long double>("long double");

  // A complex-valued integrand: int_{-1}^{1} exp(ix) dx = 2 sin(1).
  const auto q = GaussQuad::GaussLegendreQuadrature1D<double>(16);
  const auto z = q.Integrate([](double x) {
    return std::exp(std::complex<double>(0, x));
  });
  std::cout << std::format(
      "\n  int exp(ix) dx over [-1,1] = {:.15f} {:+.15f}i\n"
      "                       exact = {:.15f} {:+.15f}i\n",
      z.real(), z.imag(), 2 * std::sin(1.0), 0.0);

  // A vector-valued one, integrated componentwise in a single pass.
  const Vec2 v = q.Integrate([](double x) { return Vec2{x * x, std::cos(x)}; });
  std::cout << std::format(
      "\n  int (x^2, cos x) dx over [-1,1] = ({:.15f}, {:.15f})\n"
      "                            exact = ({:.15f}, {:.15f})\n",
      v.a, v.b, 2.0 / 3.0, 2 * std::sin(1.0));
}
