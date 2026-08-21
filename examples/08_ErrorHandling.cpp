// 08 -- What happens when you ask for something impossible.
//
// Anything that BUILDS a rule validates its arguments and throws: such a call
// happens once, and a bad argument otherwise yields a rule that is silently
// wrong rather than obviously wrong.  The pure evaluation functions --
// operator(), Derivative -- keep assertions instead, since they sit in inner
// loops and a bad degree there is a programming error rather than bad data.
//
// The distinction matters because dependent code is compiled with NDEBUG,
// where an assertion diagnoses nothing at all.

#include <GaussQuad/All>
#include <format>
#include <iostream>
#include <stdexcept>
#include <string>

// Run something that should throw, and report what came out.
template <typename Function>
void Expect(const char* what, Function f) {
  try {
    f();
    std::cout << std::format("    {:<44} did not throw\n", what);
  } catch (const std::invalid_argument& e) {
    std::cout << std::format("    {:<44} invalid_argument\n      {}\n", what,
                             e.what());
  } catch (const std::runtime_error& e) {
    std::cout << std::format("    {:<44} runtime_error\n      {}\n", what,
                             e.what());
  }
}

int main() {
  using Real = double;

  std::cout << "  a degree too small for the rule:\n";
  Expect("GaussLegendreQuadrature1D<double>(0)",
         [] { return GaussQuad::GaussLegendreQuadrature1D<Real>(0); });
  Expect("GaussRadauLegendreQuadrature1D<double>(1)",
         [] { return GaussQuad::GaussRadauLegendreQuadrature1D<Real>(1); });

  std::cout << "\n  a weight function that is not integrable:\n";
  Expect("JacobiPolynomial<double>(-1, 0)",
         [] { return GaussQuad::JacobiPolynomial<Real>(-1, 0); });
  Expect("LaguerrePolynomial<double>(-2)",
         [] { return GaussQuad::LaguerrePolynomial<Real>(-2); });

  std::cout << "\n  a malformed rule:\n";
  Expect("Quadrature1D<double>({{1, 2}, {1}})", [] {
    return GaussQuad::Quadrature1D<Real>(
        {std::vector<Real>{1, 2}, std::vector<Real>{1}});
  });

  // Failures of the numerics themselves are runtime_error rather than
  // invalid_argument: the request was reasonable, the arithmetic could not
  // meet it.  See example 05 for what this one means.
  std::cout << "\n  a request the precision cannot meet:\n";
  Expect("float GLR at n = 12000", [] {
    return GaussQuad::GaussLegendreQuadrature1D<float>(
        12000, GaussQuad::Method::GlaserLiuRokhlin);
  });

  // The smallest valid degrees are valid, including the two-point Lobatto
  // rule -- which is the trapezoid rule.
  std::cout << "\n  the smallest valid rules build without complaint:\n";
  const auto g = GaussQuad::GaussLegendreQuadrature1D<Real>(1);
  const auto l = GaussQuad::GaussLobattoLegendreQuadrature1D<Real>(2);
  std::cout << std::format("    1-point Gauss:   x = {}, w = {}\n", g.X(0), g.W(0));
  std::cout << std::format("    2-point Lobatto: x = {}, {}   w = {}, {}"
                           "   (the trapezoid rule)\n",
                           l.X(0), l.X(1), l.W(0), l.W(1));
}
