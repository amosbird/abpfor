cmake_minimum_required(VERSION 3.16)

find_program(CLANGXX clang++)
if(CLANGXX AND NOT CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER ${CLANGXX} CACHE STRING "C++ compiler" FORCE)
endif()

project(abpfor LANGUAGES CXX)

# Clang only. The codec relies on clang's codegen for its SIMD paths, and GCC's
# -O3 loop vectoriser miscompiles the interleaved unpack at several bit widths
# (see README.md). Fail loudly rather than silently building a broken library.
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  message(FATAL_ERROR
    "abpfor requires clang (got ${CMAKE_CXX_COMPILER_ID}). "
    "GCC -O3 miscompiles the interleaved unpack path; see README.md.")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------------------
# Compile flags
# ---------------------------------------------------------------------------
# NDEBUG is set unconditionally: this library validates nothing by design (see
# the UB notes in abpfor.h), so it carries no runtime asserts of its own, and
# NDEBUG additionally disables any assertions the standard library headers would
# otherwise compile in.
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

# Warning set. abpfor is vendored into ClickHouse, so it follows ClickHouse's
# policy (cmake/warnings.cmake): -Weverything -Wpedantic, then disable only the
# ones that are noise for this kind of code, each with a reason. Clang only —
# see README for why GCC is not supported.
#
# The codec truncates deliberately all over (packing values into bytes,
# narrowing bounded block lengths, popcounts), so every such conversion is
# written as an explicit cast. An implicit narrowing here is a bug, not style,
# which is why the conversion diagnostics stay on.
set(ABPFOR_WARNINGS
  -Weverything -Wpedantic
  -Wno-c++98-compat                # C++20 library; pre-C++11 compatibility is irrelevant
  -Wno-c++98-compat-pedantic        # ditto
  -Wno-pre-c++14-compat             # ditto
  -Wno-pre-c++17-compat             # ditto
  -Wno-c++20-compat                 # we target C++20, not older standards
  -Wno-unsafe-buffer-usage          # too aggressive: this is a bit-packing codec, raw pointer arithmetic is the job
  -Wno-unused-template              # fires on static templates in headers included by several TUs
  -Wno-padded                       # padding is a micro-optimisation, not actionable globally
  -Wno-global-constructors          # the scatter/shuffle lookup tables are constant global data
  -Wno-used-but-marked-unused       # glibc marks printf unused; only fires at -O3, in the tests
  -Wno-sign-conversion              # matches ClickHouse, which also leaves this off
)
option(ABPFOR_WERROR "Treat warnings as errors" OFF)
if(ABPFOR_WERROR)
  list(APPEND ABPFOR_WARNINGS -Werror)
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
target_compile_options(abpfor PRIVATE ${ABPFOR_FLAGS} ${ABPFOR_WARNINGS} -mavx2)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()

function(abpfor_add_test name)
  add_executable(${name} tests/${name}.cpp)
  target_link_libraries(${name} PRIVATE abpfor)
  target_compile_options(${name} PRIVATE ${ABPFOR_FLAGS} ${ABPFOR_WARNINGS} -mavx2)
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
abpfor_add_test(test_narrowing)
