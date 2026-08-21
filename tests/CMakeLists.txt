include(FetchContent)

# Pinned: an unpinned tag makes the build non-reproducible and lets an upstream
# change break this project without anything here having changed.
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Interpolation is used only by the end-to-end exactness tests.
FetchContent_Declare(
  Interpolation
  GIT_REPOSITORY https://github.com/da380/Interpolation.git
  GIT_TAG main
)
FetchContent_MakeAvailable(Interpolation)

add_executable(Tests Tests.cpp)
target_link_libraries(Tests PRIVATE GaussQuad::GaussQuad gtest_main Interpolation)

include(GoogleTest)
gtest_discover_tests(Tests)
