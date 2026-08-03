#pragma once

// abpfor::arch — compile-time target architecture detection.
//
// SIMD selection is entirely compile-time. There is no runtime CPU feature
// dispatch: on x86-64 the library requires AVX2 as a hard baseline (build with
// -mavx2), and on AArch64 NEON is mandatory by the ABI. Building for x86
// without AVX2 produces a binary that will fault on an AVX2-less CPU — that is
// the documented trade for keeping the hot decode loop branch-free.
//
// ABPFOR_ARCH_X86 / ABPFOR_ARCH_ARM64 gate the interleaved SIMD codec in
// detail/codec.h. If neither is defined the scalar path is used, which is
// correct but roughly 3-4x slower on decode.

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define ABPFOR_ARCH_X86 1
#elif defined(__aarch64__) || defined(_M_ARM64)
#define ABPFOR_ARCH_ARM64 1
#endif
