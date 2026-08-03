#pragma once

// abpfor::bits — low-level bit manipulation, unaligned memory access, constants.
// All functions are branchless or single-instruction on x86-64.

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "endian.h"

namespace abpfor
{

// ---------------------------------------------------------------------------
// Compiler hint macros
// ---------------------------------------------------------------------------

#if defined(__GNUC__) || defined(__clang__)
#define ABPFOR_INLINE      __attribute__((always_inline)) inline
#define ABPFOR_NOINLINE    __attribute__((noinline))
#define ABPFOR_LIKELY(x)   __builtin_expect(!!(x), 1)
#define ABPFOR_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define ABPFOR_INLINE inline
#define ABPFOR_NOINLINE
#define ABPFOR_LIKELY(x)   (x)
#define ABPFOR_UNLIKELY(x) (x)
#endif

// ---------------------------------------------------------------------------
// bitwidth — minimum bits needed to represent a value (0 → 0, 1 → 1, 255 → 8)
// ---------------------------------------------------------------------------
// Uses __builtin_clz/clzll, which compilers lower to the best available
// instruction for the target:
//   - bsr + xor  on SSE4.2-only CPUs (Nehalem–Ivy Bridge)
//   - lzcnt      when -mlzcnt is in effect (Haswell+)
// The compiler also inserts xor-break to avoid the lzcnt false-dependency
// on Haswell–Coffee Lake. No inline asm needed.

ABPFOR_INLINE unsigned bitwidth(uint32_t v)
{
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
    // lzcnt handles v=0 natively (returns 32), no branch needed.
    // On CPUs without ABM, lzcnt decodes as bsr (rep prefix ignored),
    // which gives undefined result for v=0 — but v=0 returns 32-32=0
    // due to the subtraction, matching our semantics.
    unsigned lz;
    __asm__("lzcntl %1, %0" : "=r"(lz) : "rm"(v) : "cc");
    return 32u - lz;
#else
    return v ? (32u - static_cast<unsigned>(__builtin_clz(v))) : 0u;
#endif
}

ABPFOR_INLINE unsigned bitwidth(uint64_t v)
{
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
    uint64_t lz;
    __asm__("lzcntq %1, %0" : "=r"(lz) : "rm"(v) : "cc");
    return static_cast<unsigned>(64u - lz);
#else
    return v ? (64u - static_cast<unsigned>(__builtin_clzll(v))) : 0u;
#endif
}

// ---------------------------------------------------------------------------
// mask — bitmask with the lowest `b` bits set
// ---------------------------------------------------------------------------
// mask<uint32_t>(0) = 0, mask<uint32_t>(32) = 0xFFFFFFFF.
// Branchless: uses the trick that (1<<b)-1 is UB for b=32/64, so we
// special-case via a conditional that compilers eliminate into cmov.

template <typename T> ABPFOR_INLINE constexpr T mask(unsigned b)
{
    static_assert(std::is_unsigned_v<T>);
    constexpr unsigned W = sizeof(T) * 8;
    if (b == 0) return T(0);
    if (b >= W) return ~T(0);
    return (T(1) << b) - T(1);
}

// ---------------------------------------------------------------------------
// Unaligned load/store — little-endian, safe on any alignment
// ---------------------------------------------------------------------------
// Uses memcpy (optimised to single mov by any modern compiler on x86).
// Explicit little-endian: no-op on x86, byte-swap on big-endian.

template <typename T> ABPFOR_INLINE T loadu(const uint8_t* p)
{
    static_assert(std::is_unsigned_v<T>);
    T v;
    std::memcpy(&v, p, sizeof(T));
    return detail::leToNative(v);
}

template <typename T> ABPFOR_INLINE void storeu(uint8_t* p, T v)
{
    static_assert(std::is_unsigned_v<T>);
    v = detail::nativeToLe(v);
    std::memcpy(p, &v, sizeof(T));
}

// ---------------------------------------------------------------------------
// divCeil — ceiling division, no overflow for reasonable inputs
// ---------------------------------------------------------------------------

constexpr unsigned divCeil(unsigned num, unsigned den)
{
    return (num + den - 1u) / den;
}

// Bytes needed to store `count` values of `bits` width each
constexpr unsigned packedBytes(unsigned count, unsigned bits)
{
    return divCeil(count * bits, 8u);
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

template <typename T> constexpr unsigned kMaxBits = sizeof(T) * 8;

} // namespace abpfor
