cmake_minimum_required(VERSION 3.16)

find_program(CLANGXX clang++)
if(CLANGXX AND NOT CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER ${CLANGXX} CACHE STRING "C++ compiler" FORCE)
endif()

project(abpfor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------------------
# Compile flags
# ---------------------------------------------------------------------------
set(ABPFOR_FLAGS
  -O3 -DNDEBUG -g
  -fstrict-aliasing -ffunction-sections -fdata-sections
  -fomit-frame-pointer -falign-functions=64
  -ftree-vectorize -funroll-loops -ffp-contract=off
  -msse4.2
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  list(APPEND ABPFOR_FLAGS -fvectorize)
endif()

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mbranches-within-32B-boundaries" HAS_BRANCH_BOUNDS)
if(HAS_BRANCH_BOUNDS)
  list(APPEND ABPFOR_FLAGS -mbranches-within-32B-boundaries)
endif()

# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
add_library(abpfor STATIC
  src/pack_dispatch_delta32.cpp
  src/pack_dispatch_delta64.cpp
  src/api_b128.cpp
  src/api_b256.cpp
)
target_include_directories(abpfor PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_options(abpfor PRIVATE ${ABPFOR_FLAGS} -mavx2)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()

function(abpfor_add_test name)
  add_executable(${name} tests/${name}.cpp)
  target_link_libraries(${name} PRIVATE abpfor)
  target_compile_options(${name} PRIVATE ${ABPFOR_FLAGS} -mavx2)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

abpfor_add_test(test_bits)
abpfor_add_test(test_pack)
abpfor_add_test(test_cost)
abpfor_add_test(test_roundtrip)
abpfor_add_test(test_sse_pack)
abpfor_add_test(test_sse_fused)
abpfor_add_test(test_api)
abpfor_add_test(test_avx2_pack)
abpfor_add_test(test_avx2_fused)
abpfor_add_test(test_comprehensive)
abpfor_add_test(test_cross_platform)
abpfor_add_test(test_scalar_fallback)
abpfor_add_test(test_interleaved64)
abpfor_add_test(test_delta0)
