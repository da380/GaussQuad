// 02 -- Putting a rule somewhere other than [-1,1].
//
// Three ways, in increasing order of how easy they are to get wrong.  Prefer
// the first.

#include <GaussQuad/All>
#include <cmath>
#include <format>
#include <iostream>

int main() {
  using Real = double;

  const int n = 8;
  const Real a = 0, b = 3;

  const auto f = [](Real x) { return x * x; };  // int_0^3 x^2 dx = 9.

  // 1. Ask for the rule on [a,b] in the first place.  Nothing to get wrong.
  const auto q = GaussQuad::GaussLegendreQuadrature1D<Real>(n, a, b);
  std::cout << std::format("  {:<22}= {:.16f}\n",
                           std::format("built on [{}, {}]", a, b),
                           q.Integrate(f));

  // 2. Map an existing rule.  MappedTo returns a copy and leaves the original
  //    alone, so one rule can be reused across many intervals.
  const auto r = GaussQuad::GaussLegendreQuadrature1D<Real>(n);
  std::cout << std::format("  {:<22}= {:.16f}\n", "r.MappedTo(a, b)",
                           r.MappedTo(a, b).Integrate(f));
  std::cout << std::format("  r itself is untouched: r.X(0) = {:.16f}\n", r.X(0));

  // 3. A general change of variable y = g(x).  Note the convention: the
  //    points are mapped first, and dg is then evaluated at the MAPPED
  //    points.  So dg must be written as a function of the new variable y,
  //    returning dy/dx there -- not as a function of x.  For an affine map dg
  //    is constant and the distinction does not arise.
  auto s = GaussQuad::GaussLegendreQuadrature1D<Real>(n);
  s.Transform([=](Real x) { return (b - a) / 2 * x + (a + b) / 2; },
              [=](Real) { return (b - a) / 2; });
  std::cout << std::format("  {:<22}= {:.16f}\n", "s.Transform(g, dg)",
                           s.Integrate(f));

  std::cout << std::format("\n  {:<22}= {:.16f}\n", "exact", Real(9));

  // Where the convention bites: a non-affine map.  Take y = x^3 on [-1,1],
  // so dy/dx = 3x^2, which as a function of y is 3 y^(2/3).  Written that
  // way the substitution is exact here, since the integrand in x is 3x^8 and
  // the rule reaches degree 2n-1 = 15.
  const auto cube = GaussQuad::GaussLegendreQuadrature1D<Real>(n).Transformed(
      [](Real x) { return x * x * x; },
      [](Real y) { return 3 * std::cbrt(y) * std::cbrt(y); });

  // The same thing done wrongly, with dg given as a function of the old
  // variable.  It throws no error and returns a plausible number.
  const auto wrong = GaussQuad::GaussLegendreQuadrature1D<Real>(n).Transformed(
      [](Real x) { return x * x * x; }, [](Real x) { return 3 * x * x; });

  const auto sq = [](Real y) { return y * y; };
  std::cout << std::format("\n  int y^2 dy over [-1,1] through y = x^3:\n");
  std::cout << std::format("    dg as a function of y = {:.16f}   (exact {:.16f})\n",
                           cube.Integrate(sq), Real(2) / 3);
  std::cout << std::format("    dg as a function of x = {:.16f}   <-- wrong\n",
                           wrong.Integrate(sq));
}
