#ifndef GAUSS_QUAD_QUADRATURE_GUARD_H
#define GAUSS_QUAD_QUADRATURE_GUARD_H

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

// Concept for functions that can be integrated using quadrature.
template <typename Real, typename Function, typename FunctionValue>
concept Integrable = requires(Real w, FunctionValue f) {
  requires NumericConcepts::Real<Real>;
  requires NumericConcepts::Function<Function, FunctionValue, Real>;
  { f* w } -> std::same_as<FunctionValue>;
  { f + f } -> std::same_as<FunctionValue>;
};

template <NumericConcepts::Real Real>
class Quadrature1D {
  using Vector = std::vector<Real>;
  using VectorPair = std::pair<Vector, Vector>;

 public:
  Quadrature1D() = default;

  // Constructor given pair of vectors for points and weights.
  Quadrature1D(VectorPair pair)
      : _x{std::move(std::get<0>(pair))}, _w{std::move(std::get<1>(pair))} {
    Internal::Require(!_x.empty(), "a quadrature rule needs at least one point",
                      _x.size());
    Internal::Require(_x.size() == _w.size(),
                      "points and weights differ in length", _w.size());
  }

  // Return the number of points.
  int N() const { return static_cast<int>(_x.size()); }

  // Return the ith points or weights.
  auto X(int i) const { return _x[i]; }
  auto W(int i) const { return _w[i]; }

  // Return constant references to the points and weights.
  const Vector& Points() const { return _x; }
  const Vector& Weights() const { return _w; }

#ifdef __cpp_lib_ranges_zip
  // A view of the rule as (point, weight) pairs, for
  //   for (auto [x, w] : q.Nodes()) ...
  auto Nodes() const { return std::views::zip(_x, _w); }
#endif

  // Approximate the integral of f against the weight function of the rule.
  //
  // Note that the weight is part of the rule, not of f: for a Jacobi rule
  // this returns an approximation to int (1-x)^a (1+x)^b f(x) dx, and for
  // Laguerre or Hermite the corresponding weighted integral.  Only for the
  // Legendre rules, whose weight is one, is it the plain integral of f.
  template <typename Function,
            typename FunctionValue = std::invoke_result_t<Function, Real>>
  requires Integrable<Real, Function, FunctionValue>
  auto Integrate(const Function& f) const {
    return std::inner_product(
        _x.cbegin(), _x.cend(), _w.cbegin(), FunctionValue{}, std::plus<>(),
        [f](Real x, Real w) -> FunctionValue { return f(x) * w; });
  }

  // Map the rule through a change of variable y = f(x), in place.
  //
  // Note the convention, which is easy to get wrong: the points are mapped
  // first, and df is then evaluated at the MAPPED points.  So df must be
  // supplied as a function of the new variable y, returning dy/dx there --
  // not as a function of x.  For the affine maps that are the usual case df
  // is constant and the distinction does not arise.
  template <typename Function1, typename Function2>
  void Transform(Function1 f, Function2 df) {
    std::transform(_x.begin(), _x.end(), _x.begin(), f);
    std::transform(_x.begin(), _x.end(), _w.begin(), _w.begin(),
                   [&f, &df](auto x, auto w) { return df(x) * w; });
  }

  // As Transform, but returning a new rule and leaving this one alone.
  template <typename Function1, typename Function2>
  Quadrature1D Transformed(Function1 f, Function2 df) const {
    auto q = *this;
    q.Transform(f, df);
    return q;
  }

  // Return a copy of the rule mapped affinely from [-1,1] onto [a,b].  Only
  // meaningful for a rule that is currently on [-1,1], which is every rule
  // this library builds except the Laguerre and Hermite ones.
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

// Factory functions for the Quadrature1D type.

template <NumericConcepts::Real Real>
auto GaussLegendreQuadrature1D(int n, Method method = Method::GolubWelsch) {
  return Quadrature1D(LegendrePolynomial<Real>{}.GaussQuadrature(n, method));
}

template <NumericConcepts::Real Real>
auto GaussRadauLegendreQuadrature1D(int n) {
  return Quadrature1D(LegendrePolynomial<Real>{}.GaussRadauQuadrature(n));
}

template <NumericConcepts::Real Real>
auto GaussLobattoLegendreQuadrature1D(int n) {
  return Quadrature1D(LegendrePolynomial<Real>{}.GaussLobattoQuadrature(n));
}

template <NumericConcepts::Real Real>
auto GaussChebyshevQuadrature1D(int n) {
  return Quadrature1D(ChebyshevPolynomial<Real>{}.GaussQuadrature(n));
}

template <NumericConcepts::Real Real>
auto GaussRadauChebyshevQuadrature1D(int n) {
  return Quadrature1D(ChebyshevPolynomial<Real>{}.GaussRadauQuadrature(n));
}

template <NumericConcepts::Real Real>
auto GaussLobattoChebyshevQuadrature1D(int n) {
  return Quadrature1D(ChebyshevPolynomial<Real>{}.GaussLobattoQuadrature(n));
}

// Gauss-Laguerre: integrates x^alpha exp(-x) f(x) over [0, infinity).
template <NumericConcepts::Real Real>
auto GaussLaguerreQuadrature1D(int n, Real alpha = 0) {
  return Quadrature1D(LaguerrePolynomial<Real>{alpha}.GaussQuadrature(n));
}

// Gauss-Hermite: integrates exp(-x^2) f(x) over the whole real line.
template <NumericConcepts::Real Real>
auto GaussHermiteQuadrature1D(int n) {
  return Quadrature1D(HermitePolynomial<Real>{}.GaussQuadrature(n));
}

template <NumericConcepts::Real Real>
auto GaussJacobiQuadrature1D(int n, Real alpha, Real beta) {
  return Quadrature1D(JacobiPolynomial<Real>{alpha, beta}.GaussQuadrature(n));
}

template <NumericConcepts::Real Real>
auto GaussRadauJacobiQuadrature1D(int n, Real alpha, Real beta) {
  return Quadrature1D(
      JacobiPolynomial<Real>{alpha, beta}.GaussRadauQuadrature(n));
}

template <NumericConcepts::Real Real>
auto GaussLobattoJacobiQuadrature1D(int n, Real alpha, Real beta) {
  return Quadrature1D(
      JacobiPolynomial<Real>{alpha, beta}.GaussLobattoQuadrature(n));
}

// Overloads placing the rules that live on [-1,1] onto an arbitrary interval
// [a,b].  Equivalent to building the rule and calling MappedTo(a, b).

template <NumericConcepts::Real Real>
auto GaussLegendreQuadrature1D(int n, Real a, Real b,
                               Method method = Method::GolubWelsch) {
  return GaussLegendreQuadrature1D<Real>(n, method).MappedTo(a, b);
}

template <NumericConcepts::Real Real>
auto GaussRadauLegendreQuadrature1D(int n, Real a, Real b) {
  return GaussRadauLegendreQuadrature1D<Real>(n).MappedTo(a, b);
}

template <NumericConcepts::Real Real>
auto GaussLobattoLegendreQuadrature1D(int n, Real a, Real b) {
  return GaussLobattoLegendreQuadrature1D<Real>(n).MappedTo(a, b);
}

}  // namespace GaussQuad

#endif  // GAUSS_QUAD_QUADRATURE_GUARD_H
