#pragma once

// abpfor::arch — compile-time target architecture detection.
//
// SIMD selection is entirely compile-time. There is no runtime CPU feature
// dispatch: on x86-64 the library wants AVX2 (build with -mavx2), and on
// AArch64 NEON is mandatory by the ABI.
//
// ABPFOR_ARCH_X86 / ABPFOR_ARCH_ARM64 gate the interleaved SIMD codec in
// detail/codec.h. Both are always defined, to 1 or 0 — the guards below are
// value tests (`#if ABPFOR_ARCH_X86`), and an undefined macro there would be
// an error under -Wundef rather than a silent 0. If both are 0 the scalar path
// is used, which is correct but roughly 3-4x slower on decode.
//
// ABPFOR_ARCH_X86 tests __AVX2__, not the processor family. Every block it
// guards uses 256-bit intrinsics, so gating on __x86_64__ alone would feed
// _mm256_* to a compiler invoked without -mavx2 and fail with "always_inline
// function requires target feature 'avx'". A consumer targeting x86-64-v1/v2
// (ClickHouse does this in its compat build, -DX86_ARCH_LEVEL=1) therefore
// lands on the scalar path here instead of failing to compile.

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) && defined(__AVX2__)
#define ABPFOR_ARCH_X86 1
#else
#define ABPFOR_ARCH_X86 0
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define ABPFOR_ARCH_ARM64 1
#else
#define ABPFOR_ARCH_ARM64 0
#endif
