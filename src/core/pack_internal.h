#pragma once

// Shared helpers for 32-bit and 64-bit bitpack implementations.

#include "bits.h"
#include "endian.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>

namespace abpfor::detail::bitops
{

// --- Load/store helpers ---

// Bit unpacking loads whole 64-bit words and lets the final load run past the
// packed bits rather than assembling that word byte by byte. The extra bytes
// are read but discarded, so the caller must leave this many readable bytes
// after the compressed data. Documented in include/abpfor.h and pinned by
// test_unpack_read_bounds in tests/test_pack.cpp -- do not raise it without
// updating both.
inline constexpr unsigned kUnpackSlack = 3;

#if defined(__i386__) || defined(__x86_64__)
// x86 (always LE): may_alias direct access
using U64Alias = uint64_t __attribute__((__may_alias__));
using U32Alias = uint32_t __attribute__((__may_alias__));
using U16Alias = uint16_t __attribute__((__may_alias__));

inline uint64_t loadU64Fast(const unsigned char* in)
{
    return *reinterpret_cast<const U64Alias*>(in);
}
inline void storeU64Fast(unsigned char* out, uint64_t v)
{
    *reinterpret_cast<U64Alias*>(out) = v;
}
inline uint32_t loadU32Fast(const unsigned char* in)
{
    return *reinterpret_cast<const U32Alias*>(in);
}
inline void storeU32Fast(unsigned char* out, uint32_t v)
{
    *reinterpret_cast<U32Alias*>(out) = v;
}
inline uint16_t loadU16Fast(const unsigned char* in)
{
    return *reinterpret_cast<const U16Alias*>(in);
}
inline void storeU16Fast(unsigned char* out, uint16_t v)
{
    *reinterpret_cast<U16Alias*>(out) = v;
}
#else
// Portable: memcpy + endian conversion
inline uint64_t loadU64Fast(const unsigned char* in)
{
    return detail::loadU64(in);
}
inline void storeU64Fast(unsigned char* out, uint64_t v)
{
    detail::storeU64(out, v);
}
inline uint32_t loadU32Fast(const unsigned char* in)
{
    return detail::loadU32(in);
}
inline void storeU32Fast(unsigned char* out, uint32_t v)
{
    detail::storeU32(out, v);
}
inline uint16_t loadU16Fast(const unsigned char* in)
{
    return detail::loadU16(in);
}
inline void storeU16Fast(unsigned char* out, uint16_t v)
{
    detail::storeU16(out, v);
}
#endif

constexpr unsigned gcd_u32(unsigned a, unsigned b)
{
    return b == 0u ? a : gcd_u32(b, a % b);
}
constexpr unsigned word_count_for(unsigned b, unsigned n)
{
    return (n * b + 63u) / 64u;
}
constexpr unsigned max_words_for_block()
{
    return 32u;
}

constexpr unsigned choose_block_size(unsigned b, unsigned n)
{
    if (n == 0u) return 0u;
    const unsigned g = gcd_u32(64u, b);
    unsigned period = 64u / g;
    unsigned max_words = max_words_for_block();
    unsigned k = period;
    while (k > 1u && k > n) k >>= 1u;
    if (k > n) k = 1u;
    for (;;)
    {
        if (word_count_for(b, k) <= max_words && ((k * b) % 8u == 0u)) return k;
        if (k == 1u) break;
        k >>= 1u;
    }
    return n;
}

template <unsigned R> static ABPFOR_INLINE void store_partial(unsigned char*& op, uint64_t v)
{
    static_assert(R >= 1 && R <= 7);
    if constexpr (R >= 4)
    {
        storeU32Fast(op, static_cast<uint32_t>(v));
        op += 4u;
        if constexpr (R >= 6)
        {
            storeU16Fast(op, static_cast<uint16_t>(v >> 32));
            op += 2u;
            if constexpr (R == 7)
            {
                *op++ = static_cast<unsigned char>(v >> 48);
            }
        }
        else if constexpr (R == 5)
        {
            *op++ = static_cast<unsigned char>(v >> 32);
        }
    }
    else if constexpr (R >= 2)
    {
        storeU16Fast(op, static_cast<uint16_t>(v));
        op += 2u;
        if constexpr (R == 3)
        {
            *op++ = static_cast<unsigned char>(v >> 16);
        }
    }
    else
    {
        *op++ = static_cast<unsigned char>(v);
    }
}

template <unsigned R> static ABPFOR_INLINE uint64_t load_partial(const unsigned char*& ip)
{
    static_assert(R >= 1 && R <= 7);
    uint64_t v = 0;
    if constexpr (R >= 4)
    {
        v |= static_cast<uint64_t>(loadU32Fast(ip));
        ip += 4u;
        if constexpr (R >= 6)
        {
            v |= static_cast<uint64_t>(loadU16Fast(ip)) << 32;
            ip += 2u;
            if constexpr (R == 7)
            {
                v |= static_cast<uint64_t>(ip[0]) << 48;
                ip += 1u;
            }
        }
        else if constexpr (R == 5)
        {
            v |= static_cast<uint64_t>(ip[0]) << 32;
            ip += 1u;
        }
    }
    else if constexpr (R >= 2)
    {
        v |= static_cast<uint64_t>(loadU16Fast(ip));
        ip += 2u;
        if constexpr (R == 3)
        {
            v |= static_cast<uint64_t>(ip[0]) << 16;
            ip += 1u;
        }
    }
    else
    {
        v |= static_cast<uint64_t>(ip[0]);
        ip += 1u;
    }
    return v;
}

} // namespace abpfor::detail::bitops
