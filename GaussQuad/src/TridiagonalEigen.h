#ifndef GAUSS_QUAD_TRIDIAGONAL_EIGEN_GUARD_H
#define GAUSS_QUAD_TRIDIAGONAL_EIGEN_GUARD_H

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

// Solve T u = y, with T the symmetric tridiagonal matrix having diagonal
// d[0..m-1] and off-diagonal e[0..m-2], by unpivoted LU (the Thomas
// algorithm).  The only matrices this is applied to are J - xI with x an
// endpoint of the support of the weight function, which are definite, so no
// pivoting is needed.  d and y are taken by value and used as scratch.
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

// Nodes and weights of the Gaussian quadrature rule belonging to a set of
// three-term recurrence coefficients, by the method of Golub and Welsch: the
// nodes are the eigenvalues of the symmetric tridiagonal Jacobi matrix with
// diagonal d[0..n-1] and off-diagonal e[0..n-2], and the weights are
// mu0 * q_i^2, where q_i is the first component of the ith normalised
// eigenvector and mu0 the integral of the weight function.
//
// The eigenproblem is solved by implicit QL with Wilkinson shifts, as in the
// classic tql2/imtql2 routines, with one change: only the FIRST ROW of the
// eigenvector matrix is accumulated.  That is all Golub-Welsch needs, and it
// takes the cost from O(n^3) time and O(n^2) storage down to O(n^2) and O(n).
//
// Because the accumulated row is acted on only by plane rotations, its
// Euclidean norm is preserved, and so the weights sum to mu0 to within a few
// rounding errors at any n.
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
