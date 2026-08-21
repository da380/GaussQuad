#include <GaussQuad/All>
#include <cmath>
#include <iostream>

int main() {
  using Real = double;
  using namespace GaussQuad;

  std::cout.setf(std::ios_base::scientific);
  std::cout.setf(std::ios_base::showpos);
  std::cout.precision(16);

  // A five-point Gauss-Lobatto-Legendre rule on [-1,1], so the first and last
  // points are the endpoints exactly.
  const int n = 5;
  const auto q = GaussLobattoLegendreQuadrature1D<Real>(n);

  for (int i = 0; i < n; i++) {
    std::cout << q.X(i) << " " << q.W(i) << std::endl;
  }

  // The Legendre weight is one, so Integrate is the plain integral.  For any
  // other family it would be the integral of f against that family's weight.
  const auto f = [](Real x) { return x * x; };
  const Real exact = Real(2) / Real(3);

  std::cout << "Numerical value = " << q.Integrate(f)
            << ", exact value = " << exact << std::endl;

  // The same rule placed on [0,2] directly, rather than by transforming it.
  const auto r = GaussLobattoLegendreQuadrature1D<Real>(n, Real(0), Real(2));
  std::cout << "On [0,2]        = " << r.Integrate(f)
            << ", exact value = " << Real(8) / Real(3) << std::endl;
}
