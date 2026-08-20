#include <gtest/gtest.h>

#include <GaussQuad/All>
#include <Interpolation/Polynomial>
#include <cmath>
#include <complex>
#include <concepts>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Identity-based checks.
//
// These need no reference implementation and no external library: every one of
// them is a property the rule must have exactly, whatever algorithm produced
// it.  The degrees deliberately straddle n = 100, which used to be the point
// at which the implementation switched algorithms.
// ---------------------------------------------------------------------------

template <std::floating_point Real>
constexpr auto tol = 100 * std::numeric_limits<Real>::epsilon();

// The degrees every rule is checked at.  95-105 brackets the old crossover.
const std::vector<int> Degrees() {
  return {2, 3, 4, 5, 10, 32, 64, 95, 99, 100, 101, 105, 200, 257, 1000};
}

enum class Rule { Gauss, Radau, Lobatto };

std::string Name(Rule r) {
  switch (r) {
    case Rule::Gauss:
      return "Gauss";
    case Rule::Radau:
      return "Radau";
    default:
      return "Lobatto";
  }
}

template <std::floating_point Real>
auto MakeRule(Rule r, int n, Real alpha, Real beta) {
  auto p = GaussQuad::JacobiPolynomial<Real>(alpha, beta);
  switch (r) {
    case Rule::Gauss:
      return p.GaussQuadrature(n);
    case Rule::Radau:
      return p.GaussRadauQuadrature(n);
    default:
      return p.GaussLobattoQuadrature(n);
  }
}

// The highest l for which sum_i w_i P_l(x_i)^2 is still integrated exactly:
// Gauss is exact to degree 2n-1, Radau to 2n-2 and Lobatto to 2n-3.
int MaxOrthogonalDegree(Rule r, int n) {
  switch (r) {
    case Rule::Gauss:
      return n - 1;
    case Rule::Radau:
      return n - 1;
    default:
      return n - 2;
  }
}

// mu0 = integral of the Jacobi weight over [-1,1].
template <std::floating_point Real>
Real WeightIntegral(Real alpha, Real beta) {
  using std::exp;
  using std::lgamma;
  return exp(lgamma(alpha + 1) + lgamma(beta + 1) +
             std::numbers::ln2_v<Real> * (alpha + beta + 1) -
             lgamma(alpha + beta + 2));
}

// The exact value of the integral of the Jacobi weight times P_l^(a,b)^2.
//
// Built up by the ratio h_l / h_{l-1}, rather than from the closed form in
// lgamma.  The closed form is a difference of lgammas of order l*log(l),
// which at l = 100 already costs two digits to cancellation, and it is 0/0
// at l = 0 when a + b = -1 -- so the reference would fail before the library
// did.  Each factor here is O(1) and the product is well conditioned.
template <std::floating_point Real>
Real SquaredNorm(int l, Real a, Real b, Real mu0) {
  auto h = mu0;
  for (auto k = 1; k <= l; k++) {
    // (2k+a+b-1) and (k+a+b) both vanish at k = 1 when a + b = -1, so cancel
    // that factor by hand -- the same removable singularity the library has
    // to handle in its off-diagonal recurrence coefficient.
    h *= (k == 1) ? ((1 + a) * (1 + b)) / (3 + a + b)
                  : ((2 * k + a + b - 1) * (k + a) * (k + b)) /
                        ((2 * k + a + b + 1) * (k + a + b) * k);
  }
  return h;
}

// Run every structural and identity check on one rule.
template <std::floating_point Real>
void CheckRule(Rule r, int n, Real alpha, Real beta) {
  SCOPED_TRACE(Name(r) + ", n = " + std::to_string(n) + ", alpha = " +
               std::to_string(alpha) + ", beta = " + std::to_string(beta));

  const auto [x, w] = MakeRule<Real>(r, n, alpha, beta);
  const auto mu0 = WeightIntegral(alpha, beta);

  ASSERT_EQ(static_cast<int>(x.size()), n);
  ASSERT_EQ(static_cast<int>(w.size()), n);

  // Nothing may be NaN or infinite.  Silent NaN in the weights is a failure
  // mode this library has had before, and it hides from tolerance tests.
  for (auto i = 0; i < n; i++) {
    ASSERT_TRUE(std::isfinite(x[i])) << "non-finite node at i = " << i;
    ASSERT_TRUE(std::isfinite(w[i])) << "non-finite weight at i = " << i;
  }

  // Nodes strictly increasing and strictly inside the interval, except for
  // the endpoints that Radau and Lobatto fix.
  for (auto i = 1; i < n; i++)
    EXPECT_LT(x[i - 1], x[i]) << "nodes not strictly increasing at i = " << i;
  for (auto i = 0; i < n; i++) {
    EXPECT_GE(x[i], Real(-1));
    EXPECT_LE(x[i], Real(1));
  }

  // Gaussian weights are positive.
  for (auto i = 0; i < n; i++) EXPECT_GT(w[i], Real(0)) << "at i = " << i;

  // Fixed nodes must be exactly the endpoints, not merely close to them:
  // adjacent spectral elements have to agree on a shared node bit for bit.
  if (r == Rule::Radau || r == Rule::Lobatto) EXPECT_EQ(x.front(), Real(-1));
  if (r == Rule::Lobatto) EXPECT_EQ(x.back(), Real(1));

  // Identity 1: the weights sum to the integral of the weight function.
  auto sum = Real(0);
  for (auto wi : w) sum += wi;
  EXPECT_NEAR(sum, mu0, tol<Real> * mu0) << "sum of weights";

  // Identity 2: the rule reproduces the norm of the highest-degree Jacobi
  // polynomial it is still exact for.  This is the check that separates a
  // rule that merely sums correctly from one whose nodes are right.
  const auto l = MaxOrthogonalDegree(r, n);
  if (l >= 0) {
    const auto p = GaussQuad::JacobiPolynomial<Real>(alpha, beta);
    auto quad = Real(0);
    for (auto i = 0; i < n; i++) {
      const auto v = p(l, x[i]);
      quad += w[i] * v * v;
    }
    const auto exact = SquaredNorm(l, alpha, beta, mu0);
    // The reference value is a product of l factors, so carries a relative
    // error of order l*eps; the quadrature sum itself is good to eps*mu0.
    // Allow for both, or the test measures the reference rather than the rule.
    EXPECT_NEAR(quad, exact, tol<Real> * (mu0 + l * exact))
        << "orthogonality at l = " << l;
  }
}

// Symmetric weights give a rule symmetric about the origin.
template <std::floating_point Real>
void CheckSymmetry(Rule r, int n, Real alpha) {
  SCOPED_TRACE(Name(r) + " symmetry, n = " + std::to_string(n));
  const auto [x, w] = MakeRule<Real>(r, n, alpha, alpha);
  // An even weight function gives a rule that is symmetric about the origin.
  // The library imposes this exactly, so test it exactly.
  for (auto i = 0; i < n; i++) {
    EXPECT_EQ(x[i], -x[n - 1 - i]) << "at i = " << i;
    EXPECT_EQ(w[i], w[n - 1 - i]) << "at i = " << i;
  }
}

// ---------------------------------------------------------------------------

using Reals = ::testing::Types<float, double, long double>;

template <typename Real>
class Identities : public ::testing::Test {};
TYPED_TEST_SUITE(Identities, Reals);

TYPED_TEST(Identities, GaussLegendre) {
  for (auto n : Degrees()) CheckRule<TypeParam>(Rule::Gauss, n, 0, 0);
}

TYPED_TEST(Identities, RadauLegendre) {
  for (auto n : Degrees()) CheckRule<TypeParam>(Rule::Radau, n, 0, 0);
}

TYPED_TEST(Identities, LobattoLegendre) {
  for (auto n : Degrees()) CheckRule<TypeParam>(Rule::Lobatto, n, 0, 0);
}

TYPED_TEST(Identities, GaussChebyshev) {
  for (auto n : Degrees()) CheckRule<TypeParam>(Rule::Gauss, n, -0.5, -0.5);
}

TYPED_TEST(Identities, RadauChebyshev) {
  for (auto n : Degrees()) CheckRule<TypeParam>(Rule::Radau, n, -0.5, -0.5);
}

TYPED_TEST(Identities, LobattoChebyshev) {
  for (auto n : Degrees()) CheckRule<TypeParam>(Rule::Lobatto, n, -0.5, -0.5);
}

TYPED_TEST(Identities, GaussJacobiAssorted) {
  const TypeParam pairs[][2] = {{0.5, 0.5}, {1, 0},    {0, 1},
                                {2, 3},     {-0.5, 2}, {1.5, -0.75}};
  for (const auto& ab : pairs)
    for (auto n : {2, 5, 32, 99, 100, 101, 200})
      CheckRule<TypeParam>(Rule::Gauss, n, ab[0], ab[1]);
}

TYPED_TEST(Identities, RadauLobattoJacobiAssorted) {
  const TypeParam pairs[][2] = {{0.5, 0.5}, {1, 0}, {2, 3}, {-0.5, 2}};
  for (const auto& ab : pairs)
    for (auto n : {2, 5, 32, 99, 100, 101, 200}) {
      CheckRule<TypeParam>(Rule::Radau, n, ab[0], ab[1]);
      CheckRule<TypeParam>(Rule::Lobatto, n, ab[0], ab[1]);
    }
}

TYPED_TEST(Identities, Symmetry) {
  for (auto n : {5, 32, 99, 100, 101, 200}) {
    CheckSymmetry<TypeParam>(Rule::Gauss, n, TypeParam(0));
    CheckSymmetry<TypeParam>(Rule::Lobatto, n, TypeParam(0));
    CheckSymmetry<TypeParam>(Rule::Gauss, n, TypeParam(-0.5));
  }
}

// The accuracy must not degrade with n.  This is the check that would have
// caught both the step at n = 101 and the O(n*eps) drift beyond it.
TYPED_TEST(Identities, AccuracyIsIndependentOfDegree) {
  using Real = TypeParam;
  for (auto n : {10, 100, 101, 1000, 4000}) {
    const auto [x, w] = MakeRule<Real>(Rule::Gauss, n, Real(0), Real(0));
    auto sum = Real(0);
    for (auto wi : w) sum += wi;
    EXPECT_NEAR(sum, Real(2), tol<Real> * 2) << "n = " << n;
  }
}

// Closed forms for the Chebyshev rules, which have them.
TEST(ClosedForm, GaussChebyshev) {
  using Real = double;
  constexpr auto pi = std::numbers::pi_v<Real>;
  for (auto n : {1, 2, 6, 50, 101, 300}) {
    const auto q = GaussQuad::GaussChebyshevQuadrature1D<Real>(n);
    for (auto i = 0; i < n; i++) {
      EXPECT_NEAR(q.X(i), -std::cos((2 * i + 1) * pi / (2 * n)), tol<Real>)
          << "n = " << n << ", i = " << i;
      EXPECT_NEAR(q.W(i), pi / n, 10 * tol<Real> * pi)
          << "n = " << n << ", i = " << i;
    }
  }
}

TEST(ClosedForm, LobattoChebyshev) {
  using Real = double;
  constexpr auto pi = std::numbers::pi_v<Real>;
  for (auto n : {2, 5, 20, 101}) {
    const auto q = GaussQuad::GaussLobattoChebyshevQuadrature1D<Real>(n);
    for (auto i = 0; i < n; i++) {
      EXPECT_NEAR(q.X(i), -std::cos(i * pi / (n - 1)), tol<Real>)
          << "n = " << n << ", i = " << i;
      const auto expected =
          (i == 0 || i == n - 1) ? pi / (2 * (n - 1)) : pi / (n - 1);
      EXPECT_NEAR(q.W(i), expected, 10 * tol<Real> * pi)
          << "n = " << n << ", i = " << i;
    }
  }
}

// The Chebyshev polynomials themselves, which never compiled before.
TEST(ClosedForm, ChebyshevPolynomial) {
  using Real = double;
  const auto T = GaussQuad::ChebyshevPolynomial<Real>{};
  for (auto n = 0; n <= 12; n++) {
    for (auto x : {-0.9, -0.37, 0.0, 0.37, 0.9}) {
      EXPECT_NEAR(T(n, x), std::cos(n * std::acos(x)), tol<Real>)
          << "T_" << n << "(" << x << ")";
    }
  }
}

// Zeros() finds the roots of the polynomial by Newton with Maehly deflation.
// Its answer must be the Gauss nodes, which are computed by a wholly
// independent route.  Before the deflation sum was accumulated in Real rather
// than int, deflation did nothing, so a root could be found twice.
TEST(Zeros, MatchesGaussNodes) {
  using Real = double;
  for (auto n : {1, 2, 5, 20, 64, 101}) {
    const auto z = GaussQuad::LegendrePolynomial<Real>{}.Zeros(n);
    const auto x =
        GaussQuad::LegendrePolynomial<Real>{}.GaussQuadrature(n).first;
    ASSERT_EQ(z.size(), x.size()) << "n = " << n;
    for (auto i = 0u; i < z.size(); i++) {
      EXPECT_NEAR(z[i], x[i], 1e-12) << "n = " << n << ", i = " << i;
      // Distinct roots: deflation exists to prevent Newton returning to one
      // it has already found.
      if (i > 0) EXPECT_LT(z[i - 1], z[i]) << "n = " << n << ", i = " << i;
    }
  }
}

// The two-point Lobatto rule is the trapezoid rule; it used to be rejected by
// an over-strict assertion.
TEST(EdgeCase, LobattoTwoPoint) {
  const auto q = GaussQuad::GaussLobattoLegendreQuadrature1D<double>(2);
  EXPECT_EQ(q.X(0), -1.0);
  EXPECT_EQ(q.X(1), 1.0);
  EXPECT_DOUBLE_EQ(q.W(0), 1.0);
  EXPECT_DOUBLE_EQ(q.W(1), 1.0);
}

TEST(EdgeCase, GaussOnePoint) {
  const auto q = GaussQuad::GaussLegendreQuadrature1D<double>(1);
  EXPECT_NEAR(q.X(0), 0.0, tol<double>);
  EXPECT_DOUBLE_EQ(q.W(0), 2.0);
}

// Quadrature1D must be usable through a const reference.
TEST(Api, IntegrateIsConst) {
  const auto q = GaussQuad::GaussLegendreQuadrature1D<double>(10);
  EXPECT_NEAR(q.Integrate([](double x) { return x * x; }), 2.0 / 3.0,
              tol<double>);
}

// ---------------------------------------------------------------------------
// The O(n) Gauss-Legendre algorithm.
// ---------------------------------------------------------------------------

template <typename Real>
class Asymptotic : public ::testing::Test {};
TYPED_TEST_SUITE(Asymptotic, Reals);

TYPED_TEST(Asymptotic, StructureAndIdentities) {
  using Real = TypeParam;
  constexpr auto eps = std::numeric_limits<Real>::epsilon();
  for (auto n : {1, 2, 3, 4, 5, 8, 17, 32, 64, 65, 100, 101, 257, 1000}) {
    SCOPED_TRACE("n = " + std::to_string(n));
    const auto [x, w] = GaussQuad::LegendrePolynomial<Real>{}.GaussQuadrature(
        n, GaussQuad::Method::GlaserLiuRokhlin);
    ASSERT_EQ(static_cast<int>(x.size()), n);
    for (auto i = 0; i < n; i++) {
      ASSERT_TRUE(std::isfinite(x[i]));
      ASSERT_TRUE(std::isfinite(w[i]));
      EXPECT_GT(w[i], Real(0));
      EXPECT_LE(std::abs(x[i]), Real(1));
      if (i > 0) EXPECT_LT(x[i - 1], x[i]);
    }
    // The weights drift as O(n*eps) here, unlike Golub-Welsch; allow for it
    // rather than pretending the two algorithms have the same error growth.
    auto sum = Real(0);
    for (auto wi : w) sum += wi;
    EXPECT_NEAR(sum, Real(2), std::max(tol<Real> * 2, n * eps));
  }
}

// The two algorithms are wholly independent -- one is an eigenproblem, the
// other a Newton march over a local power series -- so agreement between them
// is real evidence rather than a restatement.
TYPED_TEST(Asymptotic, AgreesWithGolubWelsch) {
  using Real = TypeParam;
  constexpr auto eps = std::numeric_limits<Real>::epsilon();
  for (auto n : {1, 2, 5, 17, 64, 101, 257, 1000}) {
    SCOPED_TRACE("n = " + std::to_string(n));
    const auto p = GaussQuad::LegendrePolynomial<Real>{};
    const auto a = p.GaussQuadrature(n, GaussQuad::Method::GolubWelsch);
    const auto b = p.GaussQuadrature(n, GaussQuad::Method::GlaserLiuRokhlin);
    for (auto i = 0; i < n; i++) {
      EXPECT_NEAR(a.first[i], b.first[i], 200 * n * eps) << "node " << i;
      EXPECT_NEAR(a.second[i], b.second[i], 200 * n * eps) << "weight " << i;
    }
  }
}

TEST(Asymptotic, DefaultIsGolubWelsch) {
  const auto p = GaussQuad::LegendrePolynomial<double>{};
  EXPECT_EQ(p.GaussQuadrature(64).first,
            p.GaussQuadrature(64, GaussQuad::Method::GolubWelsch).first);
}

// ---------------------------------------------------------------------------
// Laguerre and Hermite.
// ---------------------------------------------------------------------------

TEST(Laguerre, Moments) {
  using Real = double;
  // sum_i w_i x_i^k = int_0^inf x^(alpha+k) exp(-x) dx = Gamma(alpha+k+1).
  for (auto alpha : {0.0, 1.5, -0.5}) {
    for (auto n : {1, 2, 8, 40, 120}) {
      const auto q = GaussQuad::GaussLaguerreQuadrature1D<Real>(n, alpha);
      for (auto k = 0; k <= std::min(2 * n - 1, 12); k++) {
        long double sum = 0;
        for (auto i = 0; i < n; i++)
          sum += static_cast<long double>(q.W(i)) * std::pow(q.X(i), k);
        const auto exact = std::exp(std::lgamma(alpha + k + 1));
        EXPECT_NEAR(static_cast<double>(sum) / exact, 1.0, 1e-13)
            << "alpha = " << alpha << ", n = " << n << ", k = " << k;
      }
      for (auto i = 0; i < n; i++) EXPECT_GT(q.X(i), 0.0);
    }
  }
}

TEST(Laguerre, ClosedFormTwoPoint) {
  const auto q = GaussQuad::GaussLaguerreQuadrature1D<double>(2);
  const auto r = std::sqrt(2.0);
  EXPECT_NEAR(q.X(0), 2 - r, tol<double>);
  EXPECT_NEAR(q.X(1), 2 + r, tol<double>);
  EXPECT_NEAR(q.W(0), (2 + r) / 4, tol<double>);
  EXPECT_NEAR(q.W(1), (2 - r) / 4, tol<double>);
}

TEST(Laguerre, RadauMoments) {
  using Real = double;
  // Gauss-Radau-Laguerre is exact to degree 2n-2, with a node fixed at zero.
  for (auto alpha : {0.0, 1.5, -0.5}) {
    for (auto n : {2, 3, 8, 40, 120}) {
      const auto q = GaussQuad::GaussRadauLaguerreQuadrature1D<Real>(n, alpha);
      EXPECT_EQ(q.X(0), 0.0) << "alpha = " << alpha << ", n = " << n;
      for (auto i = 1; i < n; i++) EXPECT_GT(q.X(i), 0.0);
      for (auto k = 0; k <= std::min(2 * n - 2, 12); k++) {
        long double sum = 0;
        for (auto i = 0; i < n; i++)
          sum += static_cast<long double>(q.W(i)) * std::pow(q.X(i), k);
        const auto exact = std::exp(std::lgamma(alpha + k + 1));
        EXPECT_NEAR(static_cast<double>(sum) / exact, 1.0, 1e-13)
            << "alpha = " << alpha << ", n = " << n << ", k = " << k;
      }
    }
  }
}

TEST(Laguerre, RadauClosedFormTwoPoint) {
  // Exact for degree 2, so w0+w1 = 1, w1 x1 = 1, w1 x1^2 = 2: {0,2},{1/2,1/2}.
  const auto q = GaussQuad::GaussRadauLaguerreQuadrature1D<double>(2);
  EXPECT_EQ(q.X(0), 0.0);
  EXPECT_NEAR(q.X(1), 2.0, tol<double>);
  EXPECT_NEAR(q.W(0), 0.5, tol<double>);
  EXPECT_NEAR(q.W(1), 0.5, tol<double>);
}

TEST(Hermite, Moments) {
  using Real = double;
  // sum_i w_i x_i^(2m) = int exp(-x^2) x^(2m) dx = Gamma(m+1/2); odd
  // moments vanish.
  for (auto n : {1, 2, 8, 40, 120}) {
    const auto q = GaussQuad::GaussHermiteQuadrature1D<Real>(n);
    for (auto k = 0; k <= std::min(2 * n - 1, 12); k++) {
      long double sum = 0;
      for (auto i = 0; i < n; i++)
        sum += static_cast<long double>(q.W(i)) * std::pow(q.X(i), k);
      if (k % 2 == 1) {
        EXPECT_NEAR(static_cast<double>(sum), 0.0, 1e-13)
            << "n=" << n << " k=" << k;
      } else {
        const auto exact = std::exp(std::lgamma(k / 2 + 0.5));
        EXPECT_NEAR(static_cast<double>(sum) / exact, 1.0, 1e-13)
            << "n = " << n << ", k = " << k;
      }
    }
    // The weight is even, so the rule is exactly symmetric.
    for (auto i = 0; i < n; i++) {
      EXPECT_EQ(q.X(i), -q.X(n - 1 - i));
      EXPECT_EQ(q.W(i), q.W(n - 1 - i));
    }
  }
}

TEST(Hermite, ClosedFormThreePoint) {
  const auto q = GaussQuad::GaussHermiteQuadrature1D<double>(3);
  const auto r = std::sqrt(1.5);
  const auto s = std::sqrt(std::numbers::pi_v<double>);
  EXPECT_NEAR(q.X(0), -r, tol<double>);
  EXPECT_EQ(q.X(1), 0.0);
  EXPECT_NEAR(q.X(2), r, tol<double>);
  EXPECT_NEAR(q.W(0), s / 6, tol<double>);
  EXPECT_NEAR(q.W(1), 2 * s / 3, tol<double>);
}

TEST(Laguerre, PolynomialValues) {
  const auto L = GaussQuad::LaguerrePolynomial<double>{};
  // L_0 = 1, L_1 = 1-x, L_2 = 1 - 2x + x^2/2, L_3 = 1 - 3x + 3x^2/2 - x^3/6.
  for (auto x : {0.0, 0.4, 2.3}) {
    EXPECT_NEAR(L(0, x), 1.0, tol<double>);
    EXPECT_NEAR(L(1, x), 1 - x, tol<double>);
    EXPECT_NEAR(L(2, x), 1 - 2 * x + x * x / 2, tol<double>);
    EXPECT_NEAR(L(3, x), 1 - 3 * x + 1.5 * x * x - x * x * x / 6, tol<double>);
    EXPECT_NEAR(L.Derivative(2, x), -2 + x, tol<double>);
  }
}

TEST(Hermite, PolynomialValues) {
  const auto H = GaussQuad::HermitePolynomial<double>{};
  // H_0 = 1, H_1 = 2x, H_2 = 4x^2-2, H_3 = 8x^3-12x.
  for (auto x : {-1.1, 0.0, 0.4}) {
    EXPECT_NEAR(H(0, x), 1.0, tol<double>);
    EXPECT_NEAR(H(1, x), 2 * x, tol<double>);
    EXPECT_NEAR(H(2, x), 4 * x * x - 2, tol<double>);
    EXPECT_NEAR(H(3, x), 8 * x * x * x - 12 * x, tol<double>);
    EXPECT_NEAR(H.Derivative(3, x), 24 * x * x - 12, tol<double>);
  }
}

// ---------------------------------------------------------------------------
// The additive Quadrature1D API.
// ---------------------------------------------------------------------------

TEST(Api, MappedToAndIntervalFactories) {
  using Real = double;
  const Real a = 2, b = 5;
  const auto q = GaussQuad::GaussLegendreQuadrature1D<Real>(8, a, b);
  EXPECT_NEAR(q.Integrate([](Real x) { return x * x; }),
              (b * b * b - a * a * a) / 3, 1e-13);
  // MappedTo and the factory overload must agree exactly.
  const auto m = GaussQuad::GaussLegendreQuadrature1D<Real>(8).MappedTo(a, b);
  EXPECT_EQ(q.Points(), m.Points());
  EXPECT_EQ(q.Weights(), m.Weights());
  // A mapped Lobatto rule still lands exactly on the endpoints.
  const auto l =
      GaussQuad::GaussLobattoLegendreQuadrature1D<Real>(6, -3.0, 1.0);
  EXPECT_EQ(l.X(0), -3.0);
  EXPECT_EQ(l.X(5), 1.0);
}

TEST(Api, TransformedDoesNotMutate) {
  using Real = double;
  const auto q = GaussQuad::GaussLegendreQuadrature1D<Real>(6);
  const auto before = q.Points();
  const auto t =
      q.Transformed([](Real x) { return 2 * x; }, [](Real) { return 2.0; });
  EXPECT_EQ(q.Points(), before);
  for (auto i = 0; i < q.N(); i++) EXPECT_NEAR(t.X(i), 2 * q.X(i), tol<Real>);
}

#ifdef __cpp_lib_ranges_zip
TEST(Api, NodesView) {
  const auto q = GaussQuad::GaussLegendreQuadrature1D<double>(7);
  auto i = 0;
  double sum = 0;
  for (auto [x, w] : q.Nodes()) {
    EXPECT_EQ(x, q.X(i));
    EXPECT_EQ(w, q.W(i));
    sum += w * x * x;
    i++;
  }
  EXPECT_EQ(i, q.N());
  EXPECT_NEAR(sum, 2.0 / 3.0, tol<double>);
}
#endif

// An expression-template value type: operator* and operator+ return a proxy
// rather than the value type, as Eigen's do.  The Integrable concept used to
// require identity and so rejected every such integrand.
namespace {
struct Proxy {
  double a, b;
};
struct Vec2 {
  double a{}, b{};
  Vec2() = default;
  Vec2(double x, double y) : a{x}, b{y} {}
  Vec2(Proxy p) : a{p.a}, b{p.b} {}
};
inline Proxy operator*(const Vec2& v, double w) { return {v.a * w, v.b * w}; }
inline Proxy operator+(const Vec2& x, const Vec2& y) {
  return {x.a + y.a, x.b + y.b};
}
}  // namespace

TEST(Api, IntegratesExpressionTemplateValues) {
  const auto q = GaussQuad::GaussLegendreQuadrature1D<double>(8);
  const Vec2 r = q.Integrate([](double x) { return Vec2{x * x, 1.0}; });
  EXPECT_NEAR(r.a, 2.0 / 3.0, 1e-14);
  EXPECT_NEAR(r.b, 2.0, 1e-14);
}

// ---------------------------------------------------------------------------
// Argument checking.  These used to be assertions, so in a release build a
// bad argument was silently undefined rather than diagnosed.
// ---------------------------------------------------------------------------

TEST(Validation, RejectsBadDegree) {
  using Real = double;
  const auto p = GaussQuad::LegendrePolynomial<Real>{};
  EXPECT_THROW(p.GaussQuadrature(0), std::invalid_argument);
  EXPECT_THROW(p.GaussQuadrature(-3), std::invalid_argument);
  EXPECT_THROW(p.GaussQuadrature(0, GaussQuad::Method::GlaserLiuRokhlin),
               std::invalid_argument);
  EXPECT_THROW(p.GaussRadauQuadrature(1), std::invalid_argument);
  EXPECT_THROW(p.GaussLobattoQuadrature(1), std::invalid_argument);
  EXPECT_THROW(p.Zeros(-1), std::invalid_argument);
  EXPECT_THROW(GaussQuad::GaussLaguerreQuadrature1D<Real>(0),
               std::invalid_argument);
  EXPECT_THROW(GaussQuad::GaussHermiteQuadrature1D<Real>(0),
               std::invalid_argument);
  // The smallest valid degrees must still work.
  EXPECT_NO_THROW(p.GaussQuadrature(1));
  EXPECT_NO_THROW(p.GaussRadauQuadrature(2));
  EXPECT_NO_THROW(p.GaussLobattoQuadrature(2));
  EXPECT_TRUE(p.Zeros(0).empty());
}

TEST(Validation, RejectsNonIntegrableWeights) {
  using Real = double;
  // The Jacobi weight is integrable only for alpha, beta > -1.
  EXPECT_THROW(GaussQuad::JacobiPolynomial<Real>(-1, 0), std::invalid_argument);
  EXPECT_THROW(GaussQuad::JacobiPolynomial<Real>(0, -2), std::invalid_argument);
  EXPECT_THROW(GaussQuad::LaguerrePolynomial<Real>(-1), std::invalid_argument);
  EXPECT_NO_THROW(GaussQuad::JacobiPolynomial<Real>(-0.999, -0.5));
  EXPECT_NO_THROW(GaussQuad::LaguerrePolynomial<Real>(-0.999));
}

TEST(Validation, RejectsMalformedRule) {
  using Vector = std::vector<double>;
  EXPECT_THROW(GaussQuad::Quadrature1D<double>({Vector{}, Vector{}}),
               std::invalid_argument);
  EXPECT_THROW(GaussQuad::Quadrature1D<double>({Vector{1, 2}, Vector{1}}),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// End-to-end: integrate random polynomials of the highest degree each rule is
// exact for.  Kept from the original suite.
// ---------------------------------------------------------------------------

template <std::floating_point Real>
constexpr auto eps = 2000 * std::numeric_limits<Real>::epsilon();

template <std::floating_point Real, bool Complex = false>
int TestExactness(Rule r, int n) {
  auto q = [&] {
    switch (r) {
      case Rule::Gauss:
        return GaussQuad::GaussLegendreQuadrature1D<Real>(n);
      case Rule::Radau:
        return GaussQuad::GaussRadauLegendreQuadrature1D<Real>(n);
      default:
        return GaussQuad::GaussLobattoLegendreQuadrature1D<Real>(n);
    }
  }();
  const int m = (r == Rule::Gauss) ? 2 * n - 1 : 2 * n - 3;
  if constexpr (Complex) {
    auto p = Interpolation::Polynomial1D<std::complex<Real>>::Random(m);
    Real error = std::abs(q.Integrate(p) - p.Integrate(-1, 1));
    return (error < eps<Real>) ? 0 : 1;
  } else {
    auto p = Interpolation::Polynomial1D<Real>::Random(m);
    Real error = std::abs(q.Integrate(p) - p.Integrate(-1, 1));
    return (error < eps<Real>) ? 0 : 1;
  }
}

int RandomDegree() {
  std::random_device rd{};
  std::mt19937_64 gen{rd()};
  std::uniform_int_distribution d{5, 200};
  return d(gen);
}

#define EXACTNESS_TEST(suite, name, rule, Real, Complex)                \
  TEST(suite, name) {                                                   \
    EXPECT_EQ((TestExactness<Real, Complex>(rule, RandomDegree())), 0); \
  }

EXACTNESS_TEST(Gauss, RealDouble, Rule::Gauss, double, false)
EXACTNESS_TEST(Gauss, ComplexDouble, Rule::Gauss, double, true)
EXACTNESS_TEST(Gauss, RealLongDouble, Rule::Gauss, long double, false)
EXACTNESS_TEST(Gauss, ComplexLongDouble, Rule::Gauss, long double, true)
EXACTNESS_TEST(Radau, RealDouble, Rule::Radau, double, false)
EXACTNESS_TEST(Radau, ComplexDouble, Rule::Radau, double, true)
EXACTNESS_TEST(Radau, RealLongDouble, Rule::Radau, long double, false)
EXACTNESS_TEST(Radau, ComplexLongDouble, Rule::Radau, long double, true)
EXACTNESS_TEST(Lobatto, RealDouble, Rule::Lobatto, double, false)
EXACTNESS_TEST(Lobatto, ComplexDouble, Rule::Lobatto, double, true)
EXACTNESS_TEST(Lobatto, RealLongDouble, Rule::Lobatto, long double, false)
EXACTNESS_TEST(Lobatto, ComplexLongDouble, Rule::Lobatto, long double, true)
