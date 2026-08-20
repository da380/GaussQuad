#ifndef GAUSS_QUAD_GLASER_LIU_ROKHLIN_GUARD_H
#define GAUSS_QUAD_GLASER_LIU_ROKHLIN_GUARD_H

#include <cassert>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

#include "NumericConcepts/Numeric.hpp"

namespace GaussQuad {

namespace Internal {

// Gauss-Legendre nodes and weights in O(n) time and O(n) storage, by the
// method of Glaser, Liu and Rokhlin (2007).
//
// P_n satisfies the Legendre equation (1-x^2)y'' - 2xy' + n(n+1)y = 0.  Write
// u(h) = P_n(x + h) about some point x, and put c = 1 - x^2 and b = -2x; the
// equation becomes
//
//   (c + b h - h^2) u'' + (b - 2h) u' + lambda u = 0,   lambda = n(n+1),
//
// and collecting powers of h gives the Taylor recurrence
//
//   u_{m+2} = -[ b (m+1)^2 u_{m+1} + (lambda - m(m+1)) u_m ] / [c (m+2)(m+1)].
//
// So the whole local behaviour of P_n follows from the pair (u_0, u_1) at a
// single point, at O(1) cost.  Starting from the known values at x = 0, each
// root is found by Newton on that local series and then becomes the expansion
// point for the next -- at a root u_0 = 0, so only P_n'(x_k) has to be
// carried.  Marching over the roots in (0,1) and reflecting gives all n of
// them in O(n) work, with the polynomial never evaluated by recurrence.
//
// Compared with Golub-Welsch this trades a little accuracy for a lot of
// speed: the nodes agree to about 1e-15, but the weights drift as O(n*eps)
// rather than staying flat, so the sum of the weights is near 1e-13 by
// n = 16385, where it costs about 2 ms against Golub-Welsch's 5 s.
template <NumericConcepts::Real Real>
std::pair<std::vector<Real>, std::vector<Real>> GaussLegendreGLR(int n) {
  assert(n > 0);

  // Enough Taylor terms to reach the precision of the type.
  constexpr auto digits = std::numeric_limits<Real>::digits;
  constexpr auto order = digits <= 24 ? 20 : (digits <= 53 ? 30 : 44);

  using std::abs;
  using std::acos;
  using std::cos;
  constexpr auto eps = std::numeric_limits<Real>::epsilon();

  auto x = std::vector<Real>(n);
  auto w = std::vector<Real>(n);

  const auto odd = n % 2 == 1;
  const auto half = n / 2;  // number of roots strictly greater than zero
  const auto lambda = static_cast<Real>(n) * (n + 1);

  // Seed with the exactly known values at the origin.  Both are formed as a
  // product of ratios, which neither overflows nor underflows: P_n(0) decays
  // like sqrt(2/(pi n)) and P_n'(0) grows like sqrt(2n/pi).
  auto u0 = Real(0);
  auto u1 = Real(0);
  if (odd) {
    // P_n(0) = 0 and P_n'(0) = (-1)^((n-1)/2) n!! / (n-1)!!.
    u1 = 1;
    for (auto k = 3; k <= n; k += 2) u1 *= static_cast<Real>(k) / (k - 1);
    if (((n - 1) / 2) % 2 == 1) u1 = -u1;
  } else {
    // P_n(0) = (-1)^(n/2) (n-1)!! / n!! and P_n'(0) = 0.
    u0 = 1;
    for (auto k = 1; k <= n - 1; k += 2) u0 *= static_cast<Real>(k) / (k + 1);
    if ((n / 2) % 2 == 1) u0 = -u0;
  }

  // Keep P_n'(0): for odd n it gives the weight of the node at the origin,
  // and u is about to be overwritten by each expansion in turn.
  const auto derivativeAtZero = u1;

  auto u = std::vector<Real>(order + 1);
  auto centre = Real(0);
  auto theta = std::numbers::pi_v<Real> / 2;
  const auto step = std::numbers::pi_v<Real> / (n + Real(0.5));

  for (auto k = 0; k < half; k++) {
    // The roots are nearly equally spaced in theta.  For even n they straddle
    // theta = pi/2, so the first step from the seed is only half a spacing.
    const auto first = (!odd && k == 0);
    const auto guess = cos(theta - (first ? step / 2 : step)) - centre;

    // Build the series in the scaled variable t = h / guess.  Unscaled, the
    // Taylor coefficients grow like (lambda/c)^(m/2), which for the outer
    // roots -- where c = 1 - x^2 is small -- overflows the exponent range
    // long before the series itself stops being accurate.  In t every
    // coefficient is O(1).
    const auto scale = guess;
    const auto c = 1 - centre * centre;
    const auto b = -2 * centre;
    u[0] = u0;
    u[1] = u1 * scale;
    for (auto m = 0; m + 2 <= order; m++) {
      u[m + 2] =
          -(b * (m + 1) * (m + 1) * u[m + 1] * scale +
            (lambda - static_cast<Real>(m) * (m + 1)) * u[m] * scale * scale) /
          (c * (m + 2) * (m + 1));
    }

    // Newton from t = 1, which is the asymptotic estimate of the next root.
    auto t = Real(1);
    auto converged = false;
    auto previous = std::numeric_limits<Real>::max();
    for (auto it = 0; it < 100; it++) {
      auto f = Real(0);
      auto df = Real(0);
      for (auto m = order; m >= 1; m--) {
        f = f * t + u[m];
        df = df * t + m * u[m];
      }
      f = f * t + u[0];
      const auto delta = -f / df;
      t += delta;
      const auto size = abs(delta);
      // Stop on convergence, or once the correction stops shrinking: the
      // series is evaluated in Real, so at low precision the iteration stalls
      // at its rounding level before delta reaches eps * |t|.
      if (size <= 8 * eps * abs(t) || (it >= 2 && size >= previous)) {
        converged = true;
        break;
      }
      previous = size;
    }
    if (!converged) {
      throw std::runtime_error(
          "GaussQuad: Glaser-Liu-Rokhlin iteration failed to converge");
    }

    const auto root = centre + scale * t;
    // Recover the derivative at the new root from the same series, undoing
    // the scaling: dP/dx = (1/scale) dv/dt.
    auto der = Real(0);
    for (auto m = order; m >= 1; m--) der = der * t + m * u[m];
    der /= scale;

    const auto i = (odd ? half + 1 : half) + k;
    x[i] = root;
    w[i] = 2 / ((1 - root * root) * der * der);

    // The new root becomes the next expansion point; there u_0 vanishes.
    centre = root;
    u0 = 0;
    u1 = der;
    theta = acos(root);
  }

  if (odd) {
    x[half] = 0;
    w[half] = 2 / (derivativeAtZero * derivativeAtZero);
  }

  // Reflect: the rule is symmetric about the origin.
  for (auto k = 0; k < half; k++) {
    const auto i = (odd ? half + 1 : half) + k;
    x[half - 1 - k] = -x[i];
    w[half - 1 - k] = w[i];
  }

  // The march is the one part of this that could go quietly wrong: a Newton
  // step landing on a root that has already been found would give a rule that
  // still looks plausible.  Check that it did not.
  for (auto i = 1; i < n; i++) {
    if (!(x[i - 1] < x[i]) || !(abs(x[i]) <= 1) || !(w[i] > 0)) {
      throw std::runtime_error(
          "GaussQuad: Glaser-Liu-Rokhlin produced an invalid rule");
    }
  }

  return {std::move(x), std::move(w)};
}

}  // namespace Internal

}  // namespace GaussQuad

#endif  // GAUSS_QUAD_GLASER_LIU_ROKHLIN_GUARD_H
