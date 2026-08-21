#ifndef GAUSS_QUAD_TRIDIAGONAL_EIGEN_GUARD_H
#define GAUSS_QUAD_TRIDIAGONAL_EIGEN_GUARD_H

/// \file TridiagonalEigen.h
/// \brief The symmetric tridiagonal solvers the Golub-Welsch construction
///        needs: a Thomas solve and a first-row-only QL eigensolver.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "NumericConcepts/Numeric.hpp"

namespace GaussQuad {

namespace Internal {

/// \brief Solve a symmetric tridiagonal system by unpivoted LU, the Thomas
///        algorithm, in O(m) time and no extra storage.
///
/// Solves \f$T u = y\f$, with \f$T\f$ the symmetric tridiagonal matrix
/// having diagonal `d[0..m-1]` and off-diagonal `e[0..m-2]`.
///
/// The only matrices this is applied to are \f$J - xI\f$ with \f$x\f$ an
/// endpoint of the support of the weight function.  Those are definite -- the
/// shift lies outside the spectrum -- so no pivoting is needed.
///
/// \tparam Real A real floating-point type.
/// \param d Diagonal entries, taken by value and used as scratch.
/// \param e Off-diagonal entries; at least `m - 1` of them.
/// \param y Right-hand side, taken by value and used as scratch.
/// \return The solution vector, of the same length as \p d.
template <NumericConcepts::Real Real>
std::vector<Real> SolveSymmetricTridiagonal(std::vector<Real> d,
                                            const std::vector<Real>& e,
                                            std::vector<Real> y) {
  const auto m = static_cast<int>(d.size());
  assert(m > 0);
  assert(static_cast<int>(e.size()) + 1 >= m);
  assert(static_cast<int>(y.size()) == m);
  for (auto i = 1; i < m; i++) {
    const auto f = e[i - 1] / d[i - 1];
    d[i] -= f * e[i - 1];
    y[i] -= f * y[i - 1];
  }
  y[m - 1] /= d[m - 1];
  for (auto i = m - 2; i >= 0; i--) y[i] = (y[i] - e[i] * y[i + 1]) / d[i];
  return y;
}

/// \brief Nodes and weights of a Gauss rule from its three-term recurrence
///        coefficients, by the method of Golub and Welsch.
///
/// The nodes are the eigenvalues of the symmetric tridiagonal Jacobi matrix
/// with diagonal `d[0..n-1]` and off-diagonal `e[0..n-2]`, and the weights are
/// \f$w_i = \mu_0 q_i^2\f$, where \f$q_i\f$ is the first component of the
/// \f$i\f$th normalised eigenvector.
///
/// The eigenproblem is solved by implicit QL with Wilkinson shifts, as in the
/// classic `tql2`/`imtql2` routines, with one change: only the **first row**
/// of the eigenvector matrix is accumulated.  That is all Golub-Welsch needs,
/// and it takes the cost from \f$O(n^3)\f$ time and \f$O(n^2)\f$ storage
/// down to \f$O(n^2)\f$ and \f$O(n)\f$.
///
/// Because the accumulated row is acted on only by plane rotations, which
/// preserve its Euclidean norm, the weights sum to \f$\mu_0\f$ to within a
/// few rounding errors at any \f$n\f$, rather than drifting with \f$n\f$.
///
/// \tparam Real A real floating-point type.
/// \param d Diagonal of the Jacobi matrix, taken by value and used as scratch.
/// \param e Off-diagonal of the Jacobi matrix; at least `n - 1` entries.
/// \param mu0 The integral of the weight function over its support.
/// \return The nodes, in ascending order, and the corresponding weights.
/// \throws std::runtime_error If the QL iteration fails to converge.
template <NumericConcepts::Real Real>
std::pair<std::vector<Real>, std::vector<Real>> GolubWelsch(std::vector<Real> d,
                                                            std::vector<Real> e,
                                                            Real mu0) {
  const auto n = static_cast<int>(d.size());
  assert(n > 0);
  assert(static_cast<int>(e.size()) + 1 >= n);

  if (n == 1) return {std::move(d), std::vector<Real>(1, mu0)};

  // The algorithm below reads e[n-1]; pad it with an explicit zero.
  e.resize(n);
  e[n - 1] = 0;

  // First row of the eigenvector matrix, initially that of the identity.
  auto z = std::vector<Real>(n, Real{0});
  z[0] = 1;

  using std::abs;
  using std::hypot;
  constexpr auto eps = std::numeric_limits<Real>::epsilon();
  constexpr auto maxIter = 50;

  for (auto l = 0; l < n; l++) {
    auto iter = 0;
    while (true) {
      // Look for a small off-diagonal element at which to split the matrix.
      auto m = l;
      for (; m < n - 1; m++) {
        const auto dd = abs(d[m]) + abs(d[m + 1]);
        if (abs(e[m]) <= eps * dd) break;
      }
      if (m == l) break;
      if (iter++ == maxIter) {
        throw std::runtime_error(
            "GaussQuad: tridiagonal QL iteration failed to converge");
      }

      // Wilkinson shift, formed as in tql2 to avoid cancellation.
      auto g = (d[l + 1] - d[l]) / (2 * e[l]);
      auto r = hypot(g, Real{1});
      g = d[m] - d[l] + e[l] / (g + (g >= 0 ? r : -r));

      auto s = Real{1};
      auto c = Real{1};
      auto p = Real{0};

      auto i = m - 1;
      for (; i >= l; i--) {
        auto f = s * e[i];
        const auto b = c * e[i];
        r = hypot(f, g);
        e[i + 1] = r;
        if (r == 0) {
          // Recover from underflow.
          d[i + 1] -= p;
          e[m] = 0;
          break;
        }
        s = f / r;
        c = g / r;
        g = d[i + 1] - p;
        r = (d[i] - g) * s + 2 * c * b;
        p = s * r;
        d[i + 1] = g + p;
        g = c * r - b;
        // Apply the rotation to the accumulated first row.
        f = z[i + 1];
        z[i + 1] = s * z[i] + c * f;
        z[i] = c * z[i] - s * f;
      }
      if (r == 0 && i >= l) continue;
      d[l] -= p;
      e[l] = g;
      e[m] = 0;
    }
  }

  // Sort into ascending order of node, carrying the weights along.
  auto order = std::vector<int>(n);
  std::iota(order.begin(), order.end(), 0);
  std::ranges::sort(order, {}, [&d](auto i) { return d[i]; });

  auto x = std::vector<Real>(n);
  auto w = std::vector<Real>(n);
  for (auto i = 0; i < n; i++) {
    x[i] = d[order[i]];
    w[i] = mu0 * z[order[i]] * z[order[i]];
  }
  return {std::move(x), std::move(w)};
}

}  // namespace Internal

}  // namespace GaussQuad

#endif  // GAUSS_QUAD_TRIDIAGONAL_EIGEN_GUARD_H
