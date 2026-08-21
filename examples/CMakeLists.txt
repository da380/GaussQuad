# One target per example.  Each is standalone and covers a single aspect, and
# they are numbered in the order a new reader should meet them.
set(GAUSSQUAD_EXAMPLES
  01_BasicIntegration
  02_Intervals
  03_RadauAndLobatto
  04_WeightFunctions
  05_Algorithms
  06_Polynomials
  07_PrecisionAndValueTypes
  08_ErrorHandling
)

foreach(example IN LISTS GAUSSQUAD_EXAMPLES)
  add_executable(${example} ${example}.cpp)
  target_link_libraries(${example} PRIVATE GaussQuad::GaussQuad)
endforeach()
