cmake_minimum_required(VERSION 3.15)

project(GaussQuad
  VERSION 1.1.0
  DESCRIPTION "Header-only C++ library for Gaussian quadrature"
  LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "In-source builds are not allowed. Make a separate build "
                      "directory and run CMake from there. You may need to "
                      "remove CMakeCache.txt.")
endif()

include(GNUInstallDirs)

# Whether this is the top-level project, which decides whether the developer
# targets are defined by default.
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(GAUSSQUAD_IS_TOP_LEVEL ON)
else()
  set(GAUSSQUAD_IS_TOP_LEVEL OFF)
endif()

# Collect developer binaries in one place, but only when this is the top-level
# project: setting these unconditionally would relocate a parent project's
# output too.
if(GAUSSQUAD_IS_TOP_LEVEL)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

option(GAUSSQUAD_BUILD_EXAMPLES "Build the GaussQuad examples" ${GAUSSQUAD_IS_TOP_LEVEL})
option(GAUSSQUAD_BUILD_TESTS    "Build the GaussQuad tests"    ${GAUSSQUAD_IS_TOP_LEVEL})
option(GAUSSQUAD_INSTALL        "Generate the GaussQuad install rules" ON)

set(GAUSSQUAD_NUMERICCONCEPTS_TAG "main" CACHE STRING
    "Git tag of NumericConcepts to fetch if it is not already available")

# --- Dependencies -----------------------------------------------------------
#
# NumericConcepts is the only dependency, and is itself header-only.  Prefer an
# installed copy so that a downstream package can resolve it through
# find_dependency; fall back to fetching one.

find_package(NumericConcepts QUIET)
if(NOT NumericConcepts_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    NumericConcepts
    GIT_REPOSITORY https://github.com/da380/NumericConcepts.git
    GIT_TAG ${GAUSSQUAD_NUMERICCONCEPTS_TAG}
  )
  FetchContent_MakeAvailable(NumericConcepts)
  # FetchContent gives the bare target name; the export set needs the
  # namespaced one that find_package would provide.
  if(NOT TARGET NumericConcepts::NumericConcepts)
    add_library(NumericConcepts::NumericConcepts ALIAS NumericConcepts)
  endif()
endif()

# --- The library ------------------------------------------------------------

add_library(GaussQuad INTERFACE)
add_library(GaussQuad::GaussQuad ALIAS GaussQuad)

target_link_libraries(GaussQuad INTERFACE NumericConcepts::NumericConcepts)

# Set the standard on the target rather than through CMAKE_CXX_STANDARD, which
# would leak into a parent project that adds this one as a subdirectory.
target_compile_features(GaussQuad INTERFACE cxx_std_23)

target_include_directories(GaussQuad INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# --- Install and export -----------------------------------------------------

if(GAUSSQUAD_INSTALL)
  include(CMakePackageConfigHelpers)

  install(TARGETS GaussQuad EXPORT GaussQuadTargets)

  # The headers are extensionless module files plus src/*.h, and are included
  # as <GaussQuad/All>, so the whole directory goes under the include root.
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/GaussQuad
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

  install(EXPORT GaussQuadTargets
          FILE GaussQuadTargets.cmake
          NAMESPACE GaussQuad::
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GaussQuad)

  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/GaussQuadConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

  configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/GaussQuadConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/GaussQuadConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GaussQuad)

  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/GaussQuadConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/GaussQuadConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GaussQuad)
endif()

# --- Developer targets ------------------------------------------------------

if(GAUSSQUAD_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(GAUSSQUAD_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
