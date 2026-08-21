#ifndef GAUSS_QUAD_ORTHOGONAL_POLYNOMIAL_GUARD_H
#define GAUSS_QUAD_ORTHOGONAL_POLYNOMIAL_GUARD_H

/// \file OrthogonalPolynomial.h
/// \brief The orthogonal polynomial families and the quadrature rules
///        belonging to them.

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

/// \brief Choice of algorithm for the Gauss-Legendre rule.
///
/// Both are available at every floating-point precision.  There is
/// deliberately no automatic switch between them: a rule whose algorithm
/// changes with `n` has a discontinuity in its node values at the switch,
/// which is invisible until something downstream depends on it.
enum class Method {
  /// The default.  \f$O(n^2)\f$ time and \f$O(n)\f$ storage, with an
  /// accuracy that does not depend on `n`.  Available for every weight
  /// function.
  GolubWelsch,
  /// \f$O(n)\f$ time: about a hundred times faster by `n = 1000` and two
  /// thousand times by `n = 16000`.  The nodes agree with GolubWelsch to
  /// around `1e-15`, but the weights drift as \f$O(n\epsilon)\f$, so at
  /// large `n` it is the faster rule rather than the better one.  Available
  /// for the Legendre weight only.
  GlaserLiuRokhlin
};

/// \brief Jacobi polynomials \f$P_n^{(\alpha,\beta)}\f$, orthogonal on
///        \f$[-1,1]\f$ with respect to \f$(1-x)^\alpha (1+x)^\beta\f$.
///
/// The general family, of which LegendrePolynomial (\f$\alpha = \beta = 0\f$)
/// and ChebyshevPolynomial (\f$\alpha = \beta = -1/2\f$) are the named
/// special cases.  The associated Gauss rule approximates
/// \f$\int_{-1}^{1} (1-x)^\alpha (1+x)^\beta f(x)\,dx\f$: the weight
/// belongs to the rule, not to the integrand.
///
/// \tparam Real A real floating-point type.
template <NumericConcepts::Real Real>
class JacobiPolynomial {
  using Int = std::ptrdiff_t;

 public:
  /// \brief Construct the family for the given weight exponents.
  /// \param alpha The exponent of \f$(1-x)\f$; must exceed -1.
  /// \param beta The exponent of \f$(1+x)\f$; must exceed -1.
  /// \throws std::invalid_argument If either exponent is -1 or less, for
  ///         which the weight is not integrable.
  JacobiPolynomial(Real alpha, Real beta) : _alpha{alpha}, _beta{beta} {
    // The Jacobi weight is integrable only for alpha, beta > -1.
    Internal::Require(alpha > -1, "Jacobi alpha must exceed -1", alpha);
    Internal::Require(beta > -1, "Jacobi beta must exceed -1", beta);
  }

  /// \brief Evaluate \f$P_n^{(\alpha,\beta)}(x)\f$ by upwards recursion.
  /// \param n The degree; must be non-negative.
  /// \param x The point of evaluation.
  /// \return The value of the polynomial at \p x.
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

  /// \brief Evaluate the first derivative of \f$P_n^{(\alpha,\beta)}\f$.
  ///
  /// \warning The expression used has a removable singularity at
  ///          \f$x = \pm 1\f$ and is not evaluable there.  Take the endpoint
  ///          nodes from the Radau or Lobatto rules rather than
  ///          differentiating at the endpoints.
  ///
  /// \param n The degree; must be non-negative.
  /// \param x The point of evaluation, strictly inside \f$(-1,1)\f$.
  /// \return The value of the derivative at \p x.
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

  /// \brief The zeros of \f$P_n^{(\alpha,\beta)}\f$, by Newton's method
  ///        with Maehly deflation.
  ///
  /// These are exactly the nodes of the `n`-point Gauss rule, so this is a
  /// second and wholly independent route to them, sharing no code with the
  /// eigensolver GaussQuadrature() uses.  The test suite checks the two
  /// against each other.
  ///
  /// Both are \f$O(n^2)\f$.  Measured against GaussQuadrature() this agrees
  /// to `1.6e-15` at `n = 1025` but takes about seven times as long, so
  /// prefer GaussQuadrature() whenever the weights are wanted too.
  ///
  /// \param n The degree; must be non-negative.
  /// \return The `n` zeros in ascending order, or an empty vector for
  ///         `n = 0`.
  /// \throws std::invalid_argument If \p n is negative.
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

  /// \brief Nodes and weights of the `n`-point Gauss-Jacobi rule, exact for
  ///        polynomials of degree up to \f$2n-1\f$.
  ///
  /// The rule is symmetric about the origin when
  /// \f$\alpha = \beta\f$, and that symmetry is imposed exactly.
  ///
  /// \param n The number of points; must be positive.
  /// \return The nodes, in ascending order, and the corresponding weights.
  /// \throws std::invalid_argument If \p n is not positive.
  auto GaussQuadrature(int n) const {
    Internal::Require(n > 0, "Gauss quadrature needs at least one point", n);
    auto rule = Internal::GolubWelsch(Diagonal(n), OffDiagonal(n), Mu());
    Symmetrise(rule.first, rule.second);
    return rule;
  }

  /// \brief Nodes and weights of the `n`-point Gauss-Radau-Jacobi rule, with
  ///        a node fixed at \f$x = -1\f$, exact to degree \f$2n-2\f$.
  ///
  /// Golub's modification: solve \f$(J_m - x_1 I)\delta = e_m^2 v\f$ with
  /// \f$v\f$ the last coordinate vector, and replace the trailing diagonal
  /// entry of the Jacobi matrix by \f$x_1 + \delta_m\f$.  The fixed node is
  /// then assigned exactly rather than left to the eigensolver, which returns
  /// it only to within rounding.
  ///
  /// \param n The number of points; must be at least two.
  /// \return The nodes, in ascending order, and the corresponding weights.
  /// \throws std::invalid_argument If \p n is less than two.
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

  /// \brief Nodes and weights of the `n`-point Gauss-Lobatto-Jacobi rule,
  ///        with nodes fixed at both endpoints, exact to degree \f$2n-3\f$.
  ///
  /// Golub's modification: with \f$v\f$ the last coordinate vector, solve
  /// \f$(J_m - x_1 I)\gamma = v\f$ and \f$(J_m - x_2 I)\sigma = v\f$, then
  /// \f$e_m^2 = (x_2 - x_1)/(\gamma_m - \sigma_m)\f$ and the trailing
  /// diagonal entry is \f$x_1 + \gamma_m e_m^2\f$.  Both fixed nodes are then
  /// assigned exactly.
  ///
  /// The two-point rule is the trapezoid rule, and is valid.
  ///
  /// \param n The number of points; must be at least two.
  /// \return The nodes, in ascending order, and the corresponding weights.
  /// \throws std::invalid_argument If \p n is less than two.
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

/// \brief Legendre polynomials \f$P_n\f$, orthogonal on \f$[-1,1]\f$ with
///        unit weight.
///
/// The Jacobi family with \f$\alpha = \beta = 0\f$.  These are the rules
/// whose Integrate() is the plain integral of the integrand, every other
/// family carrying a weight function of its own.
///
/// \tparam Real A real floating-point type.
template <NumericConcepts::Real Real>
class LegendrePolynomial {
 public:
  LegendrePolynomial() : _p{JacobiPolynomial<Real>(0, 0)} {}

  /// \brief Evaluate \f$P_n(x)\f$.
  /// \param n The degree; must be non-negative.
  /// \param x The point of evaluation.
  /// \return The value of the polynomial at \p x.
  Real operator()(int n, Real x) const { return _p(n, x); }

  /// \brief Evaluate \f$P_n'(x)\f$, for \p x strictly inside
  ///        \f$(-1,1)\f$.
  /// \param n The degree; must be non-negative.
  /// \param x The point of evaluation.
  /// \return The value of the derivative at \p x.
  Real Derivative(int n, Real x) const { return _p.Derivative(n, x); }

  /// \brief The zeros of \f$P_n\f$, by Newton's method; see
  ///        JacobiPolynomial::Zeros.
  /// \param n The degree; must be non-negative.
  /// \return The `n` zeros in ascending order.
  /// \throws std::invalid_argument If \p n is negative.
  auto Zeros(int n) const { return _p.Zeros(n); }

  /// \brief Nodes and weights of the `n`-point Gauss-Legendre rule, exact for
  ///        polynomials of degree up to \f$2n-1\f$.
  ///
  /// This is the one rule for which a choice of algorithm is offered; see
  /// Method for what the choice costs and buys.
  ///
  /// \param n The number of points; must be positive.
  /// \param method Which algorithm to use.
  /// \return The nodes, in ascending order, and the corresponding weights.
  /// \throws std::invalid_argument If \p n is not positive.
  /// \throws std::runtime_error If the chosen algorithm fails to converge, or
  ///         if \p n exceeds what Method::GlaserLiuRokhlin can represent at
  ///         this precision.
  auto GaussQuadrature(int n, Method method = Method::GolubWelsch) const {
    Internal::Require(n > 0, "Gauss quadrature needs at least one point", n);
    if (method == Method::GlaserLiuRokhlin) {
      return Internal::GaussLegendreGLR<Real>(n);
    }
    return _p.GaussQuadrature(n);
  }

  /// \brief Nodes and weights of the `n`-point Gauss-Radau-Legendre rule,
  ///        with a node fixed at \f$x = -1\f$.
  /// \param n The number of points; must be at least two.
  /// \return The nodes, in ascending order, and the corresponding weights.
  /// \throws std::invalid_argument If \p n is less than two.
  auto GaussRadauQuadrature(int n) const { return _p.GaussRadauQuadrature(n); }

  /// \brief Nodes and weights of the `n`-point Gauss-Lobatto-Legendre rule,
  ///        with nodes fixed at both endpoints.
  /// \param n The number of points; must be at least two.
  /// \return The nodes, in ascending order, and the corresponding weights.
  /// \throws std::invalid_argument If \p n is less than two.
  auto GaussLobattoQuadrature(int n) const {
    return _p.GaussLobattoQuadrature(n);
  }

 private:
  JacobiPolynomial<Real> _p;
};

/// \brief Chebyshev polynomials of the first kind, \f$T_n\f$, orthogonal on
///        \f$[-1,1]\f$ with respect to \f$(1-x^2)^{-1/2}\f$.
///
/// The Jacobi family with \f$\alpha = \beta = -1/2\f$, rescaled to the
/// standard normalisation \f$T_n(\cos\theta) = \cos n\theta\f$.
///
/// \note The associated rules approximate
///       \f$\int_{-1}^{1} (1-x^2)^{-1/2} f(x)\,dx\f$, not
///       \f$\int_{-1}^{1} f(x)\,dx\f$.
///
/// \tparam Real A real floating-point type.
template <NumericConcepts::Real Real>
class ChebyshevPolynomial {
 public:
  ChebyshevPolynomial() : _p{JacobiPolynomial<Real>(-0.5, -0.5)} {}

  /// \brief Evaluate \f$T_n(x)\f$.
  /// \param n The degree; must be non-negative.
  /// \param x The point of evaluation.
  /// \return The value of the polynomial at \p x.
  Real operator()(int n, Real x) const { return Scale(n) * _p(n, x); }

  /// \brief Evaluate \f$T_n'(x)\f$, for \p x strictly inside
  ///        \f$(-1,1)\f$.
  /// \param n The degree; must be non-negative.
  /// \param x The point of evaluation.
  /// \return The value of the derivative at \p x.
  Real Derivative(int n, Real x) const {
    return Scale(n) * _p.Derivative(n, x);
  }

  /// \brief The zeros of \f$T_n\f$.
  ///
  /// The zeros and the rules below depend only on the weight function, so the
  /// normalisation does not enter them.
  ///
  /// \param n The degree; must be non-negative.
  /// \return The `n` zeros in ascending order.
  /// \throws std::invalid_argument If \p n is negative.
  auto Zeros(int n) const { return _p.Zeros(n); }

  /// \brief Nodes and weights of the `n`-point Gauss-Chebyshev rule.
  /// \param n The number of points; must be positive.
  /// \return The nodes, in ascending order, and the corresponding weights.
  /// \throws std::invalid_argument If \p n is not positive.
  auto GaussQuadrature(int n) const { return _p.GaussQuadrature(n); }

  /// \brief Nodes and weights of the `n`-point Gauss-Radau-Chebyshev rule.
  /// \param n The number of points; must be at least two.
  /// \return The nodes, in ascending order, and the corresponding weights.
  /// \throws std::invalid_argument If \p n is less than two.
  auto GaussRadauQuadrature(int n) const { return _p.GaussRadauQuadrature(n); }

  /// \brief Nodes and weights of the `n`-point Gauss-Lobatto-Chebyshev rule.
  /// \param n The number of points; must be at least two.
  /// \return The nodes, in ascending order, and the corresponding weights.
  /// \throws std::invalid_argument If \p n is less than two.
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

/// \brief Generalised Laguerre polynomials \f$L_n^{(\alpha)}\f$, orthogonal
///        on \f$[0,\infty)\f$ with respect to \f$x^\alpha e^{-x}\f$.
///
/// The associated Gauss rule approximates
/// \f$\int_0^\infty x^\alpha e^{-x} f(x)\,dx\f$.
///
/// \tparam Real A real floating-point type.
template <NumericConcepts::Real Real>
class LaguerrePolynomial {
 public:
  /// \brief Construct the family with \f$\alpha = 0\f$.
  LaguerrePolynomial() : _alpha{0} {}

  /// \brief Construct the family for a given weight exponent.
  /// \param alpha The exponent of \f$x\f$ in the weight; must exceed -1.
  /// \throws std::invalid_argument If \p alpha is -1 or less, for which the
  ///         weight is not integrable.
  explicit LaguerrePolynomial(Real alpha) : _alpha{alpha} {
    // The weight is integrable only for alpha > -1.
    Internal::Require(alpha > -1, "Laguerre alpha must exceed -1", alpha);
  }

  /// \brief Evaluate \f$L_n^{(\alpha)}(x)\f$ by upwards recursion,
  ///        \f$(m+1)L_{m+1} = (2m+1+\alpha-x)L_m - (m+\alpha)L_{m-1}\f$.
  /// \param n The degree; must be non-negative.
  /// \param x The point of evaluation.
  /// \return The value of the polynomial at \p x.
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

  /// \brief Evaluate the derivative, using
  ///        \f$\frac{d}{dx}L_n^{(\alpha)} = -L_{n-1}^{(\alpha+1)}\f$.
  /// \param n The degree; must be non-negative.
  /// \param x The point of evaluation.
  /// \return The value of the derivative at \p x.
  Real Derivative(int n, Real x) const {
    if (n == 0) return 0;
    return -LaguerrePolynomial(_alpha + 1)(n - 1, x);
  }

  /// \brief Nodes and weights of the `n`-point Gauss-Laguerre rule, exact for
  ///        polynomials of degree up to \f$2n-1\f$.
  /// \param n The number of points; must be positive.
  /// \return The nodes, in ascending order, and the corresponding weights.
  /// \throws std::invalid_argument If \p n is not positive.
  auto GaussQuadrature(int n) const {
    Internal::Require(n > 0, "Gauss quadrature needs at least one point", n);
    return Internal::GolubWelsch(Diagonal(n), OffDiagonal(n), Mu());
  }

  /// \brief Nodes and weights of the `n`-point Gauss-Radau-Laguerre rule,
  ///        with a node fixed at the origin, exact to degree \f$2n-2\f$.
  ///
  /// The origin is the natural fixed endpoint for a semi-infinite domain.
  /// Golub's modification applies exactly as for the Jacobi weights, with
  /// \f$x_1 = 0\f$: the shift vanishes, so the system to solve is the Jacobi
  /// matrix itself, positive definite because the Laguerre nodes are.
  ///
  /// \param n The number of points; must be at least two.
  /// \return The nodes, in ascending order, and the corresponding weights.
  /// \throws std::invalid_argument If \p n is less than two.
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

/// \brief Hermite polynomials \f$H_n\f$ in the physicists' normalisation,
///        orthogonal on the whole real line with respect to \f$e^{-x^2}\f$.
///
/// The associated Gauss rule approximates
/// \f$\int_{-\infty}^{\infty} e^{-x^2} f(x)\,dx\f$.
///
/// \tparam Real A real floating-point type.
template <NumericConcepts::Real Real>
class HermitePolynomial {
 public:
  /// \brief Evaluate \f$H_n(x)\f$ by upwards recursion,
  ///        \f$H_{m+1} = 2xH_m - 2mH_{m-1}\f$.
  /// \param n The degree; must be non-negative.
  /// \param x The point of evaluation.
  /// \return The value of the polynomial at \p x.
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

  /// \brief Evaluate the derivative, using \f$H_n' = 2nH_{n-1}\f$.
  /// \param n The degree; must be non-negative.
  /// \param x The point of evaluation.
  /// \return The value of the derivative at \p x.
  Real Derivative(int n, Real x) const {
    return n == 0 ? Real(0) : 2 * n * this->operator()(n - 1, x);
  }

  /// \brief Nodes and weights of the `n`-point Gauss-Hermite rule, exact for
  ///        polynomials of degree up to \f$2n-1\f$.
  ///
  /// The weight is even, so the rule is symmetric about the origin; that
  /// symmetry is imposed exactly.
  ///
  /// \param n The number of points; must be positive.
  /// \return The nodes, in ascending order, and the corresponding weights.
  /// \throws std::invalid_argument If \p n is not positive.
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
