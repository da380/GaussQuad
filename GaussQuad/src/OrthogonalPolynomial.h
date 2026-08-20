#ifndef GAUSS_QUAD_ORTHOGONAL_POLYNOMIAL_GUARD_H
#define GAUSS_QUAD_ORTHOGONAL_POLYNOMIAL_GUARD_H

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <limits>
#include <numbers>
#include <utility>
#include <vector>

#include "Checks.h"
#include "GlaserLiuRokhlin.h"
#include "NumericConcepts/Numeric.hpp"
#include "TridiagonalEigen.h"

namespace GaussQuad {

// Choice of algorithm for the Gauss-Legendre rule.
//
//   GolubWelsch      the default.  O(n^2) time and O(n) storage, and the
//                    accuracy does not depend on n.  Available for every
//                    weight function.
//
//   GlaserLiuRokhlin O(n) time, and about ten times faster by n = 1000 and a
//                    thousand times by n = 16000.  The nodes agree with
//                    Golub-Welsch to around 1e-15, but the weights drift as
//                    O(n*eps), so at very large n it is the faster rule
//                    rather than the better one.  Gauss-Legendre only.
//
// Both are available at every floating-point precision.  There is no
// automatic switch between them: a rule whose algorithm changes with n is
// exactly the discontinuity this library used to have.
enum class Method { GolubWelsch, GlaserLiuRokhlin };

template <NumericConcepts::Real Real>
class JacobiPolynomial {
  using Int = std::ptrdiff_t;

 public:
  // Constructor.
  JacobiPolynomial(Real alpha, Real beta) : _alpha{alpha}, _beta{beta} {
    // The Jacobi weight is integrable only for alpha, beta > -1.
    Internal::Require(alpha > -1, "Jacobi alpha must exceed -1", alpha);
    Internal::Require(beta > -1, "Jacobi beta must exceed -1", beta);
  }

  // Evaluation by upwards recursion.
  Real operator()(Int n, Real x) const {
    assert(n >= 0);
    constexpr auto half = static_cast<Real>(1) / static_cast<Real>(2);
    auto pm1 = static_cast<Real>(1);
    if (n == 0) return pm1;
    auto p = half * (_alpha - _beta + (_alpha + _beta + 2) * x);
    if (n == 1) return p;
    for (auto m = 1; m < n; m++) {
      pm1 = ((A2(m) + A3(m) * x) * p - A4(m) * pm1) / A1(m);
      std::swap(p, pm1);
    }
    return p;
  }

  // Evaluation of derivatives via recursion.  Note that the expression used
  // is singular at x = +-1; use the endpoint values of the quadrature rules
  // rather than differentiating there.
  Real Derivative(Int n, Real x) const {
    switch (n) {
      case 0:
        return 0;
      default:
        auto tmp = 2 * n + _alpha + _beta;
        auto b1 = tmp * (1 - x * x);
        auto b2 = n * (_alpha - _beta - tmp * x);
        auto b3 = 2 * (n + _alpha) * (n + _beta);
        return (b2 * this->operator()(n, x) + b3 * this->operator()(n - 1, x)) /
               b1;
    }
  }

  // Return zeros of the polynomial by Newton's method with Maehly deflation.
  // GaussQuadrature is both faster and more accurate at large n; this is kept
  // for the cases where only the roots are wanted.
  auto Zeros(Int n) const {
    Internal::Require(n >= 0, "cannot find the zeros of a negative degree", n);
    if (n == 0) return std::vector<Real>{};
    std::vector<Real> zeros;
    zeros.reserve(n);
    const auto maxIter = Int{30};
    constexpr auto half = static_cast<Real>(1) / static_cast<Real>(2);
    constexpr auto epsilon = std::numeric_limits<Real>::epsilon();
    const auto dth = std::numbers::pi_v<Real> / static_cast<Real>(2 * n);
    for (auto k = 0; k < n; k++) {
      auto r = -std::cos((2 * k + 1) * dth);
      if (k > 0) r = half * (r + zeros[k - 1]);
      for (auto j = 1; j < maxIter; j++) {
        auto fun = this->operator()(n, r);
        auto der = Derivative(n, r);
        // The deflation sum must be accumulated in Real; in integer
        // arithmetic it truncates to zero and deflation does nothing.
        auto sum = static_cast<Real>(0);
        for (auto i = 0; i < k; i++)
          sum += static_cast<Real>(1) / (r - zeros[i]);
        auto delr = -fun / (der - sum * fun);
        r += delr;
        if (std::abs(delr) < epsilon) break;
      }
      zeros.push_back(r);
    }
    return zeros;
  }

  // Returns points and weights for Gauss quadrature.
  auto GaussQuadrature(int n) const {
    Internal::Require(n > 0, "Gauss quadrature needs at least one point", n);
    auto rule = Internal::GolubWelsch(Diagonal(n), OffDiagonal(n), Mu());
    Symmetrise(rule.first, rule.second);
    return rule;
  }

  // Returns points and weights for Gauss-Radau quadrature, with a node fixed
  // at the left endpoint.  Golub's modification: solve (J_m - x1 I) d =
  // beta_m^2 e_m and replace the trailing diagonal entry by x1 + d_m.
  auto GaussRadauQuadrature(int n) const {
    Internal::Require(n > 1, "Gauss-Radau quadrature needs at least two points",
                      n);
    const auto m = n - 1;

    auto b = Diagonal(m);
    for (auto& bi : b) bi -= X1();
    auto y = std::vector<Real>(m, static_cast<Real>(0));
    y[m - 1] = E(m) * E(m);
    const auto delta = Internal::SolveSymmetricTridiagonal(
        std::move(b), OffDiagonal(m), y)[m - 1];

    auto d = Diagonal(n);
    d[m] = X1() + delta;
    auto rule = Internal::GolubWelsch(std::move(d), OffDiagonal(n), Mu());
    // The fixed node is known exactly; do not leave it to the eigensolver.
    rule.first.front() = X1();
    return rule;
  }

  // Returns points and weights for Gauss-Lobatto quadrature, with nodes fixed
  // at both endpoints.  Golub's modification: solve (J_m - x1 I) g = e_m and
  // (J_m - x2 I) u = e_m, then beta_m^2 = (x2 - x1) / (g_m - u_m) and the
  // trailing diagonal entry is x1 + g_m beta_m^2.
  auto GaussLobattoQuadrature(int n) const {
    Internal::Require(n > 1,
                      "Gauss-Lobatto quadrature needs at least two points", n);
    const auto m = n - 1;

    const auto e = OffDiagonal(m);
    auto y = std::vector<Real>(m, static_cast<Real>(0));
    y[m - 1] = 1;
    const auto solveShifted = [&](Real shift) {
      auto b = Diagonal(m);
      for (auto& bi : b) bi -= shift;
      return Internal::SolveSymmetricTridiagonal(std::move(b), e, y)[m - 1];
    };
    const auto gamma = solveShifted(X1());
    const auto mu = solveShifted(X2());

    using std::sqrt;
    const auto beta = sqrt((X2() - X1()) / (gamma - mu));

    auto d = Diagonal(n);
    d[m] = X1() + gamma * beta * beta;
    auto f = OffDiagonal(n);
    f[m - 1] = beta;
    auto rule = Internal::GolubWelsch(std::move(d), std::move(f), Mu());
    Symmetrise(rule.first, rule.second);
    // The fixed nodes are known exactly; do not leave them to the eigensolver.
    rule.first.front() = X1();
    rule.first.back() = X2();
    return rule;
  }

 private:
  Real _alpha, _beta;

  // Basic data functions.
  constexpr Real X1() const { return -1; }
  constexpr Real X2() const { return 1; }
  // The integral of the weight function,
  //   mu0 = 2^(a+b+1) Gamma(a+1) Gamma(b+1) / Gamma(a+b+2).
  // Gamma(a+b+2) must be taken directly rather than expanded as
  // (a+b+1) Gamma(a+b+1): the expanded form is inf - inf, hence NaN, when
  // a + b = -1, which is exactly the Chebyshev case.
  Real Mu() const {
    using std::exp;
    using std::lgamma;
    constexpr Real ln2 = std::numbers::ln2_v<Real>;
    return exp(lgamma(_alpha + 1) + lgamma(_beta + 1) +
               ln2 * (_alpha + _beta + 1) - lgamma(_alpha + _beta + 2));
  }

  // Recursion coefficient functions.
  Real A1(int n) const {
    return 2 * (n + 1) * (n + _alpha + _beta + 1) * (2 * n + _alpha + _beta);
  }
  Real A2(int n) const {
    return (2 * n + _alpha + _beta + 1) * (_alpha * _alpha - _beta * _beta);
  }
  Real A3(int n) const {
    Real tmp = 2 * n + _alpha + _beta;
    return tmp * (tmp + 1) * (tmp + 2);
  }
  Real A4(int n) const {
    return 2 * (n + _alpha) * (n + _beta) * (2 * n + _alpha + _beta + 2);
  }

  // Diagonal and off-diagonal entries of the Jacobi matrix, indexed from one.
  Real D(int n) const {
    Real num = _beta * _beta - _alpha * _alpha;
    Real den = (2 * n + _alpha + _beta - 2) * (2 * n + _alpha + _beta);
    return num != 0 ? num / den : static_cast<Real>(0);
  }
  Real E(int n) const {
    Real tmp = 2 * n + _alpha + _beta;
    // At n = 1 a factor of (n + alpha + beta) in the numerator cancels
    // against (2n + alpha + beta - 1) in the denominator.  The cancellation
    // is removable, but both factors vanish when alpha + beta = -1 -- the
    // Chebyshev case -- so the general expression evaluates to 0/0 there and
    // poisons the whole rule with NaN.  Cancel it by hand.
    if (n == 1) {
      return std::sqrt(4 * (1 + _alpha) * (1 + _beta) /
                       (tmp * tmp * (tmp + 1)));
    }
    Real num = 4 * n * (n + _alpha) * (n + _beta) * (n + _alpha + _beta);
    Real den = (tmp - 1) * tmp * tmp * (tmp + 1);
    return std::sqrt(num / den);
  }

  // When alpha == beta the weight function is even, so the rule is symmetric
  // about the origin.  The eigensolver does not know that and returns a rule
  // symmetric only to within rounding.  Impose the symmetry exactly: it is a
  // property downstream code should be able to rely on, and averaging the two
  // halves halves the error rather than adding to it.
  void Symmetrise(std::vector<Real>& x, std::vector<Real>& w) const {
    if (_alpha != _beta) return;
    const auto n = static_cast<int>(x.size());
    for (auto i = 0, j = n - 1; i < j; i++, j--) {
      const auto xm = (x[i] - x[j]) / 2;
      x[i] = xm;
      x[j] = -xm;
      const auto wm = (w[i] + w[j]) / 2;
      w[i] = wm;
      w[j] = wm;
    }
    if (n % 2 == 1) x[n / 2] = 0;
  }

  // The n x n Jacobi matrix, as diagonal and off-diagonal vectors.
  std::vector<Real> Diagonal(int n) const {
    auto d = std::vector<Real>(n);
    for (auto i = 0; i < n; i++) d[i] = D(i + 1);
    return d;
  }
  std::vector<Real> OffDiagonal(int n) const {
    auto e = std::vector<Real>(n > 0 ? n - 1 : 0);
    for (auto i = 0; i + 1 < n; i++) e[i] = E(i + 1);
    return e;
  }
};

template <NumericConcepts::Real Real>
class LegendrePolynomial {
 public:
  LegendrePolynomial() : _p{JacobiPolynomial<Real>(0, 0)} {}

  // Evaluation functions.
  Real operator()(int n, Real x) const { return _p(n, x); }
  Real Derivative(int n, Real x) const { return _p.Derivative(n, x); }

  // Zeros and quadrature schemes.
  auto Zeros(int n) const { return _p.Zeros(n); }

  // The Gauss rule, by either algorithm; see Method above.
  auto GaussQuadrature(int n, Method method = Method::GolubWelsch) const {
    Internal::Require(n > 0, "Gauss quadrature needs at least one point", n);
    if (method == Method::GlaserLiuRokhlin) {
      return Internal::GaussLegendreGLR<Real>(n);
    }
    return _p.GaussQuadrature(n);
  }

  auto GaussRadauQuadrature(int n) const { return _p.GaussRadauQuadrature(n); }
  auto GaussLobattoQuadrature(int n) const {
    return _p.GaussLobattoQuadrature(n);
  }

 private:
  JacobiPolynomial<Real> _p;
};

template <NumericConcepts::Real Real>
class ChebyshevPolynomial {
 public:
  ChebyshevPolynomial() : _p{JacobiPolynomial<Real>(-0.5, -0.5)} {}

  // Evaluation functions.
  Real operator()(int n, Real x) const { return Scale(n) * _p(n, x); }
  Real Derivative(int n, Real x) const {
    return Scale(n) * _p.Derivative(n, x);
  }

  // Zeros and quadrature schemes.  These depend only on the Jacobi weight,
  // and so are unaffected by the normalisation.
  auto Zeros(int n) const { return _p.Zeros(n); }
  auto GaussQuadrature(int n) const { return _p.GaussQuadrature(n); }
  auto GaussRadauQuadrature(int n) const { return _p.GaussRadauQuadrature(n); }
  auto GaussLobattoQuadrature(int n) const {
    return _p.GaussLobattoQuadrature(n);
  }

 private:
  JacobiPolynomial<Real> _p;

  // T_n = Scale(n) * P_n^(-1/2,-1/2), with Scale(n) = 4^n (n!)^2 / (2n)!.
  static Real Scale(int n) {
    using std::exp;
    using std::lgamma;
    constexpr Real ln2 = std::numbers::ln2_v<Real>;
    return exp(2 * n * ln2 + 2 * lgamma(static_cast<Real>(n + 1)) -
               lgamma(static_cast<Real>(2 * n + 1)));
  }
};

// Laguerre polynomials, orthogonal on [0, infinity) with respect to the
// weight x^alpha exp(-x).  The associated Gauss rule integrates
// int_0^inf x^alpha exp(-x) f(x) dx.
template <NumericConcepts::Real Real>
class LaguerrePolynomial {
 public:
  LaguerrePolynomial() : _alpha{0} {}
  explicit LaguerrePolynomial(Real alpha) : _alpha{alpha} {
    // The weight is integrable only for alpha > -1.
    Internal::Require(alpha > -1, "Laguerre alpha must exceed -1", alpha);
  }

  // Evaluation by upwards recursion:
  //   (m+1) L_{m+1} = (2m + 1 + alpha - x) L_m - (m + alpha) L_{m-1}.
  Real operator()(int n, Real x) const {
    assert(n >= 0);
    auto pm1 = static_cast<Real>(1);
    if (n == 0) return pm1;
    auto p = 1 + _alpha - x;
    for (auto m = 1; m < n; m++) {
      const auto next =
          ((2 * m + 1 + _alpha - x) * p - (m + _alpha) * pm1) / (m + 1);
      pm1 = p;
      p = next;
    }
    return p;
  }

  // d/dx L_n^alpha = -L_{n-1}^(alpha+1).
  Real Derivative(int n, Real x) const {
    if (n == 0) return 0;
    return -LaguerrePolynomial(_alpha + 1)(n - 1, x);
  }

  // Points and weights for Gauss-Laguerre quadrature.
  auto GaussQuadrature(int n) const {
    Internal::Require(n > 0, "Gauss quadrature needs at least one point", n);
    return Internal::GolubWelsch(Diagonal(n), OffDiagonal(n), Mu());
  }

  // Points and weights for Gauss-Radau-Laguerre quadrature, with a node fixed
  // at the origin -- the natural fixed endpoint for a semi-infinite domain.
  // Golub's modification, exactly as for the Jacobi weights, with x1 = 0.
  auto GaussRadauQuadrature(int n) const {
    Internal::Require(n > 1, "Gauss-Radau quadrature needs at least two points",
                      n);
    const auto m = n - 1;

    // The shift is zero, so the system is J_m itself, which is positive
    // definite because the Laguerre nodes are positive.
    auto y = std::vector<Real>(m, static_cast<Real>(0));
    y[m - 1] = E(m) * E(m);
    const auto delta = Internal::SolveSymmetricTridiagonal(
        Diagonal(m), OffDiagonal(m), y)[m - 1];

    auto d = Diagonal(n);
    d[m] = delta;
    auto rule = Internal::GolubWelsch(std::move(d), OffDiagonal(n), Mu());
    // The fixed node is known exactly.
    rule.first.front() = 0;
    return rule;
  }

 private:
  Real _alpha;

  // Integral of the weight, and the entries of the Jacobi matrix, the latter
  // indexed from one.
  Real Mu() const {
    using std::exp;
    using std::lgamma;
    return exp(lgamma(_alpha + 1));
  }
  Real D(int n) const { return 2 * n + _alpha - 1; }
  Real E(int n) const { return std::sqrt(static_cast<Real>(n) * (n + _alpha)); }
  std::vector<Real> Diagonal(int n) const {
    auto d = std::vector<Real>(n);
    for (auto i = 0; i < n; i++) d[i] = D(i + 1);
    return d;
  }
  std::vector<Real> OffDiagonal(int n) const {
    auto e = std::vector<Real>(n > 0 ? n - 1 : 0);
    for (auto i = 0; i + 1 < n; i++) e[i] = E(i + 1);
    return e;
  }
};

// Hermite polynomials, orthogonal on the whole real line with respect to the
// weight exp(-x^2); the physicists' normalisation.  The associated Gauss rule
// integrates int_{-inf}^{inf} exp(-x^2) f(x) dx.
template <NumericConcepts::Real Real>
class HermitePolynomial {
 public:
  // Evaluation by upwards recursion: H_{m+1} = 2x H_m - 2m H_{m-1}.
  Real operator()(int n, Real x) const {
    assert(n >= 0);
    auto pm1 = static_cast<Real>(1);
    if (n == 0) return pm1;
    auto p = 2 * x;
    for (auto m = 1; m < n; m++) {
      const auto next = 2 * x * p - 2 * m * pm1;
      pm1 = p;
      p = next;
    }
    return p;
  }

  // H_n' = 2n H_{n-1}.
  Real Derivative(int n, Real x) const {
    return n == 0 ? Real(0) : 2 * n * this->operator()(n - 1, x);
  }

  // Points and weights for Gauss-Hermite quadrature.
  auto GaussQuadrature(int n) const {
    Internal::Require(n > 0, "Gauss quadrature needs at least one point", n);
    auto d = std::vector<Real>(n, static_cast<Real>(0));
    auto e = std::vector<Real>(n > 0 ? n - 1 : 0);
    for (auto i = 0; i + 1 < n; i++) {
      e[i] = std::sqrt(static_cast<Real>(i + 1) / 2);
    }
    using std::sqrt;
    auto rule = Internal::GolubWelsch(std::move(d), std::move(e),
                                      sqrt(std::numbers::pi_v<Real>));
    // The weight is even, so the rule is symmetric; impose that exactly.
    auto& x = rule.first;
    auto& w = rule.second;
    for (auto i = 0, j = n - 1; i < j; i++, j--) {
      const auto xm = (x[i] - x[j]) / 2;
      x[i] = xm;
      x[j] = -xm;
      const auto wm = (w[i] + w[j]) / 2;
      w[i] = wm;
      w[j] = wm;
    }
    if (n % 2 == 1) x[n / 2] = 0;
    return rule;
  }
};

}  // namespace GaussQuad

#endif  // GAUSS_QUAD_ORTHOGONAL_POLYNOMIAL_GUARD_H
