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

add_executable(Tests Tests.cpp)
target_link_libraries(Tests PRIVATE GaussQuad::GaussQuad gtest_main)

include(GoogleTest)
gtest_discover_tests(Tests)
