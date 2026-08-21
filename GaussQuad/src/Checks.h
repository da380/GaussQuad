#ifndef GAUSS_QUAD_CHECKS_GUARD_H
#define GAUSS_QUAD_CHECKS_GUARD_H

#include <stdexcept>
#include <string>

namespace GaussQuad {

namespace Internal {

// Argument checking for the library's entry points.
//
// The convention is: anything that builds and returns a quadrature rule
// validates its arguments and throws, because such a call happens once and a
// bad argument otherwise yields a rule that is silently wrong rather than
// obviously wrong.  The pure evaluation functions -- operator(), Derivative --
// keep their assertions instead: they sit in inner loops, and a negative
// degree there is a programming error rather than bad data.

template <typename T>
[[noreturn]] inline void Invalid(const char* what, T value) {
  throw std::invalid_argument(std::string("GaussQuad: ") + what + " (got " +
                              std::to_string(value) + ")");
}

template <typename T>
inline void Require(bool condition, const char* what, T value) {
  if (!condition) Invalid(what, value);
}

}  // namespace Internal

}  // namespace GaussQuad

#endif  // GAUSS_QUAD_CHECKS_GUARD_H
