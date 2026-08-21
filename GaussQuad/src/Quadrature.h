#ifndef GAUSS_QUAD_QUADRATURE_GUARD_H
#define GAUSS_QUAD_QUADRATURE_GUARD_H

/// \file Quadrature.h
/// \brief The Quadrature1D class and the factory functions that build the
///        library's rules.

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <numeric>
#include <ranges>
#include <utility>
#include <version>

#include "Checks.h"
#include "NumericConcepts/Functions.hpp"
#include "NumericConcepts/Numeric.hpp"
#include "OrthogonalPolynomial.h"

namespace GaussQuad {

/// \brief Requirements on a function for Quadrature1D::Integrate to accept it.
///
/// The function must be callable with a \p Real and return a
/// \p FunctionValue, and that value type must be closed under addition and
/// under multiplication by \p Real.
///
/// Closure is required up to *convertibility*, not identity.  An
/// expression-template type -- an Eigen vector, say -- returns a proxy from
/// `operator*` and `operator+` rather than itself, and requiring identity
/// would reject every such integrand.  Quadrature1D::Integrate names
/// \p FunctionValue as the return type of its combining step, so the proxy is
/// evaluated there.
///
/// \tparam Real The floating-point type of the rule.
/// \tparam Function The integrand's type.
/// \tparam FunctionValue The integrand's return type.
template <typename Real, typename Function, typename FunctionValue>
concept Integrable = requires(Real w, FunctionValue f) {
  requires NumericConcepts::Real<Real>;
  requires NumericConcepts::Function<Function, FunctionValue, Real>;
  { f* w } -> std::convertible_to<FunctionValue>;
  { f + f } -> std::convertible_to<FunctionValue>;
};

/// \brief A one-dimensional quadrature rule: a set of points and the weights
///        that go with them.
///
/// Build one with the factory functions below rather than directly --
/// GaussLegendreQuadrature1D and its relatives -- unless the nodes and weights
/// come from somewhere else.
///
/// \note The weight function belongs to the rule, not to the integrand.  See
///       Integrate().
///
/// \tparam Real A real floating-point type.
template <NumericConcepts::Real Real>
class Quadrature1D {
  using Vector = std::vector<Real>;
  using VectorPair = std::pair<Vector, Vector>;

 public:
  /// \brief Construct an empty rule.
  Quadrature1D() = default;

  /// \brief Construct from a (points, weights) pair, as the polynomial
  ///        classes return.
  /// \param pair The points and the weights, of equal and non-zero length.
  /// \throws std::invalid_argument If the rule has no points, or if the two
  ///         vectors differ in length.
  Quadrature1D(VectorPair pair)
      : _x{std::move(std::get<0>(pair))}, _w{std::move(std::get<1>(pair))} {
    Internal::Require(!_x.empty(), "a quadrature rule needs at least one point",
                      _x.size());
    Internal::Require(_x.size() == _w.size(),
                      "points and weights differ in length", _w.size());
  }

  /// \brief The number of points in the rule.
  /// \return The number of points.
  int N() const { return static_cast<int>(_x.size()); }

  /// \brief The `i`th point.
  /// \param i An index in `[0, N())`; not checked.
  /// \return The point.
  auto X(int i) const { return _x[i]; }

  /// \brief The `i`th weight.
  /// \param i An index in `[0, N())`; not checked.
  /// \return The weight.
  auto W(int i) const { return _w[i]; }

  /// \brief All the points, in ascending order.
  /// \return A reference to the points, valid for the life of the rule.
  const Vector& Points() const { return _x; }

  /// \brief All the weights, in the order of the points.
  /// \return A reference to the weights, valid for the life of the rule.
  const Vector& Weights() const { return _w; }

#ifdef __cpp_lib_ranges_zip
  /// \brief A view of the rule as (point, weight) pairs:
  /// \code
  /// for (auto [x, w] : q.Nodes()) total += w * f(x);
  /// \endcode
  ///
  /// Declared only where the standard library provides `std::views::zip`.
  /// \return A view of the (point, weight) pairs.
  auto Nodes() const { return std::views::zip(_x, _w); }
#endif

  /// \brief Approximate the integral of \p f against the weight function of
  ///        the rule.
  ///
  /// \note The weight belongs to the rule, not to the integrand.  For a
  ///       Jacobi rule this returns an approximation to
  ///       \f$\int (1-x)^\alpha (1+x)^\beta f(x)\,dx\f$, and for Laguerre
  ///       or Hermite the corresponding weighted integral.  Only for the
  ///       Legendre rules, whose weight is one, is it the plain integral of
  ///       \p f.
  ///
  /// \tparam Function The integrand's type.
  /// \tparam FunctionValue The integrand's return type; deduced.
  /// \param f The integrand, callable with a \p Real.
  /// \return The weighted sum \f$\sum_i w_i f(x_i)\f$.
  template <typename Function,
            typename FunctionValue = std::invoke_result_t<Function, Real>>
  requires Integrable<Real, Function, FunctionValue>
  auto Integrate(const Function& f) const {
    return std::inner_product(
        _x.cbegin(), _x.cend(), _w.cbegin(), FunctionValue{}, std::plus<>(),
        [f](Real x, Real w) -> FunctionValue { return f(x) * w; });
  }

  /// \brief Map the rule through a change of variable \f$y = f(x)\f$, in
  ///        place.
  ///
  /// \warning Note the convention, which is easy to get wrong: the points are
  ///          mapped first, and \p df is then evaluated at the **mapped**
  ///          points.  So \p df must be supplied as a function of the new
  ///          variable \f$y\f$, returning \f$dy/dx\f$ there -- not as a
  ///          function of \f$x\f$.  For the affine maps that are the usual
  ///          case \p df is constant and the distinction does not arise; for
  ///          an affine map onto an interval, prefer MappedTo(), which cannot
  ///          be got wrong.
  ///
  /// \tparam Function1 The map's type.
  /// \tparam Function2 The Jacobian's type.
  /// \param f The change of variable.
  /// \param df The Jacobian, as a function of the new variable.
  template <typename Function1, typename Function2>
  void Transform(Function1 f, Function2 df) {
    std::transform(_x.begin(), _x.end(), _x.begin(), f);
    std::transform(_x.begin(), _x.end(), _w.begin(), _w.begin(),
                   [&f, &df](auto x, auto w) { return df(x) * w; });
  }

  /// \brief As Transform(), but returning a new rule and leaving this one
  ///        untouched; the same convention for \p df applies.
  /// \tparam Function1 The map's type.
  /// \tparam Function2 The Jacobian's type.
  /// \param f The change of variable.
  /// \param df The Jacobian, as a function of the new variable.
  /// \return The transformed rule.
  template <typename Function1, typename Function2>
  Quadrature1D Transformed(Function1 f, Function2 df) const {
    auto q = *this;
    q.Transform(f, df);
    return q;
  }

  /// \brief A copy of the rule mapped affinely from \f$[-1,1]\f$ onto
  ///        \f$[a,b]\f$.
  ///
  /// Only meaningful for a rule that currently lives on \f$[-1,1]\f$, which
  /// is every rule this library builds except the Laguerre and Hermite ones.
  ///
  /// \param a The lower limit of the new interval.
  /// \param b The upper limit of the new interval.
  /// \return The mapped rule.
  Quadrature1D MappedTo(Real a, Real b) const {
    const auto scale = (b - a) / 2;
    const auto shift = (a + b) / 2;
    auto q = *this;
    for (auto& x : q._x) x = scale * x + shift;
    for (auto& w : q._w) w *= scale;
    return q;
  }

 private:
  Vector _x;
  Vector _w;
};

/// \name Factory functions
///
/// These build a Quadrature1D from the corresponding polynomial family.  Each
/// throws std::invalid_argument for a degree too small for the rule -- one
/// point for a Gauss rule, two for Radau or Lobatto -- or for a weight
/// exponent of -1 or less.
///
/// \note Except for the Legendre rules, the weight function belongs to the
///       rule; see Quadrature1D::Integrate.
/// @{

/// \brief The `n`-point Gauss-Legendre rule on \f$[-1,1]\f$.
/// \tparam Real A real floating-point type.
/// \param n The number of points; must be positive.
/// \param method Which algorithm to use; see Method.
/// \return The rule, as a Quadrature1D.
template <NumericConcepts::Real Real>
auto GaussLegendreQuadrature1D(int n, Method method = Method::GolubWelsch) {
  return Quadrature1D(LegendrePolynomial<Real>{}.GaussQuadrature(n, method));
}

/// \brief The `n`-point Gauss-Radau-Legendre rule on \f$[-1,1]\f$, with a
///        node fixed at \f$-1\f$.
/// \tparam Real A real floating-point type.
/// \param n The number of points; must be at least two.
/// \return The rule, as a Quadrature1D.
template <NumericConcepts::Real Real>
auto GaussRadauLegendreQuadrature1D(int n) {
  return Quadrature1D(LegendrePolynomial<Real>{}.GaussRadauQuadrature(n));
}

/// \brief The `n`-point Gauss-Lobatto-Legendre rule on \f$[-1,1]\f$, with
///        nodes fixed at both endpoints.
/// \tparam Real A real floating-point type.
/// \param n The number of points; must be at least two.
/// \return The rule, as a Quadrature1D.
template <NumericConcepts::Real Real>
auto GaussLobattoLegendreQuadrature1D(int n) {
  return Quadrature1D(LegendrePolynomial<Real>{}.GaussLobattoQuadrature(n));
}

/// \brief The `n`-point Gauss-Chebyshev rule, for the weight
///        \f$(1-x^2)^{-1/2}\f$ on \f$[-1,1]\f$.
/// \tparam Real A real floating-point type.
/// \param n The number of points; must be positive.
/// \return The rule, as a Quadrature1D.
template <NumericConcepts::Real Real>
auto GaussChebyshevQuadrature1D(int n) {
  return Quadrature1D(ChebyshevPolynomial<Real>{}.GaussQuadrature(n));
}

/// \brief The `n`-point Gauss-Radau-Chebyshev rule, with a node fixed at
///        \f$-1\f$.
/// \tparam Real A real floating-point type.
/// \param n The number of points; must be at least two.
/// \return The rule, as a Quadrature1D.
template <NumericConcepts::Real Real>
auto GaussRadauChebyshevQuadrature1D(int n) {
  return Quadrature1D(ChebyshevPolynomial<Real>{}.GaussRadauQuadrature(n));
}

/// \brief The `n`-point Gauss-Lobatto-Chebyshev rule, with nodes fixed at
///        both endpoints.
/// \tparam Real A real floating-point type.
/// \param n The number of points; must be at least two.
/// \return The rule, as a Quadrature1D.
template <NumericConcepts::Real Real>
auto GaussLobattoChebyshevQuadrature1D(int n) {
  return Quadrature1D(ChebyshevPolynomial<Real>{}.GaussLobattoQuadrature(n));
}

/// \brief The `n`-point Gauss-Laguerre rule, integrating
///        \f$\int_0^\infty x^\alpha e^{-x} f(x)\,dx\f$.
/// \tparam Real A real floating-point type.
/// \param n The number of points; must be positive.
/// \param alpha The exponent of \f$x\f$ in the weight; must exceed -1.
/// \return The rule, as a Quadrature1D.
template <NumericConcepts::Real Real>
auto GaussLaguerreQuadrature1D(int n, Real alpha = 0) {
  return Quadrature1D(LaguerrePolynomial<Real>{alpha}.GaussQuadrature(n));
}

/// \brief The `n`-point Gauss-Radau-Laguerre rule: as
///        GaussLaguerreQuadrature1D, with a node fixed at the origin.
/// \tparam Real A real floating-point type.
/// \param n The number of points; must be at least two.
/// \param alpha The exponent of \f$x\f$ in the weight; must exceed -1.
/// \return The rule, as a Quadrature1D.
template <NumericConcepts::Real Real>
auto GaussRadauLaguerreQuadrature1D(int n, Real alpha = 0) {
  return Quadrature1D(LaguerrePolynomial<Real>{alpha}.GaussRadauQuadrature(n));
}

/// \brief The `n`-point Gauss-Hermite rule, integrating
///        \f$\int_{-\infty}^{\infty} e^{-x^2} f(x)\,dx\f$.
/// \tparam Real A real floating-point type.
/// \param n The number of points; must be positive.
/// \return The rule, as a Quadrature1D.
template <NumericConcepts::Real Real>
auto GaussHermiteQuadrature1D(int n) {
  return Quadrature1D(HermitePolynomial<Real>{}.GaussQuadrature(n));
}

/// \brief The `n`-point Gauss-Jacobi rule, for the weight
///        \f$(1-x)^\alpha (1+x)^\beta\f$ on \f$[-1,1]\f$.
/// \tparam Real A real floating-point type.
/// \param n The number of points; must be positive.
/// \param alpha The exponent of \f$(1-x)\f$; must exceed -1.
/// \param beta The exponent of \f$(1+x)\f$; must exceed -1.
/// \return The rule, as a Quadrature1D.
template <NumericConcepts::Real Real>
auto GaussJacobiQuadrature1D(int n, Real alpha, Real beta) {
  return Quadrature1D(JacobiPolynomial<Real>{alpha, beta}.GaussQuadrature(n));
}

/// \brief The `n`-point Gauss-Radau-Jacobi rule, with a node fixed at
///        \f$-1\f$.
/// \tparam Real A real floating-point type.
/// \param n The number of points; must be at least two.
/// \param alpha The exponent of \f$(1-x)\f$; must exceed -1.
/// \param beta The exponent of \f$(1+x)\f$; must exceed -1.
/// \return The rule, as a Quadrature1D.
template <NumericConcepts::Real Real>
auto GaussRadauJacobiQuadrature1D(int n, Real alpha, Real beta) {
  return Quadrature1D(
      JacobiPolynomial<Real>{alpha, beta}.GaussRadauQuadrature(n));
}

/// \brief The `n`-point Gauss-Lobatto-Jacobi rule, with nodes fixed at both
///        endpoints.
/// \tparam Real A real floating-point type.
/// \param n The number of points; must be at least two.
/// \param alpha The exponent of \f$(1-x)\f$; must exceed -1.
/// \param beta The exponent of \f$(1+x)\f$; must exceed -1.
/// \return The rule, as a Quadrature1D.
template <NumericConcepts::Real Real>
auto GaussLobattoJacobiQuadrature1D(int n, Real alpha, Real beta) {
  return Quadrature1D(
      JacobiPolynomial<Real>{alpha, beta}.GaussLobattoQuadrature(n));
}

/// \brief The `n`-point Gauss-Legendre rule on \f$[a,b]\f$.
///
/// Equivalent to building the rule and calling Quadrature1D::MappedTo, and
/// the reason the Quadrature1D::Transform convention need not be met in the
/// common case.
///
/// \tparam Real A real floating-point type.
/// \param n The number of points; must be positive.
/// \param a The lower limit of the interval.
/// \param b The upper limit of the interval.
/// \param method Which algorithm to use; see Method.
/// \return The rule, as a Quadrature1D.
template <NumericConcepts::Real Real>
auto GaussLegendreQuadrature1D(int n, Real a, Real b,
                               Method method = Method::GolubWelsch) {
  return GaussLegendreQuadrature1D<Real>(n, method).MappedTo(a, b);
}

/// \brief The `n`-point Gauss-Radau-Legendre rule on \f$[a,b]\f$, with a
///        node fixed at \p a.
/// \tparam Real A real floating-point type.
/// \param n The number of points; must be at least two.
/// \param a The lower limit of the interval.
/// \param b The upper limit of the interval.
/// \return The rule, as a Quadrature1D.
template <NumericConcepts::Real Real>
auto GaussRadauLegendreQuadrature1D(int n, Real a, Real b) {
  return GaussRadauLegendreQuadrature1D<Real>(n).MappedTo(a, b);
}

/// \brief The `n`-point Gauss-Lobatto-Legendre rule on \f$[a,b]\f$, with
///        nodes fixed at \p a and \p b.
/// \tparam Real A real floating-point type.
/// \param n The number of points; must be at least two.
/// \param a The lower limit of the interval.
/// \param b The upper limit of the interval.
/// \return The rule, as a Quadrature1D.
template <NumericConcepts::Real Real>
auto GaussLobattoLegendreQuadrature1D(int n, Real a, Real b) {
  return GaussLobattoLegendreQuadrature1D<Real>(n).MappedTo(a, b);
}

/// @}

}  // namespace GaussQuad

#endif  // GAUSS_QUAD_QUADRATURE_GUARD_H
