#ifndef GAUSS_QUAD_CHECKS_GUARD_H
#define GAUSS_QUAD_CHECKS_GUARD_H

/// \file Checks.h
/// \brief Argument checking for the library's entry points.

#include <stdexcept>
#include <string>

namespace GaussQuad {

/// \brief Implementation details.
///
/// Nothing in this namespace is part of the public interface, and it may
/// change without notice.  It is documented because the algorithms in it are
/// the substance of the library.
namespace Internal {

// The convention is: anything that builds and returns a quadrature rule
// validates its arguments and throws, because such a call happens once and a
// bad argument otherwise yields a rule that is silently wrong rather than
// obviously wrong.  The pure evaluation functions -- operator(), Derivative --
// keep their assertions instead: they sit in inner loops, and a negative
// degree there is a programming error rather than bad data.
//
// The distinction matters because everything depending on this library is
// compiled with NDEBUG, where an assertion diagnoses nothing.

/// \brief Throw std::invalid_argument reporting a bad argument and its value.
/// \tparam T Type of the offending value; must be accepted by std::to_string.
/// \param what What was required of the argument, as a phrase completing
///             "GaussQuad: ...".
/// \param value The value that failed the requirement.
/// \throws std::invalid_argument Always.
template <typename T>
[[noreturn]] inline void Invalid(const char* what, T value) {
  throw std::invalid_argument(std::string("GaussQuad: ") + what + " (got " +
                              std::to_string(value) + ")");
}

/// \brief Throw unless a requirement on an argument holds.
/// \tparam T Type of the value being checked.
/// \param condition The requirement; nothing happens if it is true.
/// \param what What was required, as a phrase completing "GaussQuad: ...".
/// \param value The value being checked, reported if the requirement fails.
/// \throws std::invalid_argument If \p condition is false.
template <typename T>
inline void Require(bool condition, const char* what, T value) {
  if (!condition) Invalid(what, value);
}

}  // namespace Internal

}  // namespace GaussQuad

#endif  // GAUSS_QUAD_CHECKS_GUARD_H
