#pragma once

// abpfor SSE interleaved (Interleave4) pack/unpack — 128 elements, 4-lane.
//
// Layout: 128 values are viewed as 32 groups of 4.  Each group occupies
// one SSE lane (32-bit).  At bit-width B, group g's values sit at bit
// offset g*B within each lane.  The 4 lanes are stored as consecutive
// __m128i stripes.
//
// The core template `UnpackGroup` extracts one group's 4 values.
// A function-pointer table (indexed by B) dispatches at runtime.
// Each table entry is a fully-specialised template instantiation with
// all shifts/masks as compile-time immediates.

#include "../core/bits.h"

#include <cstring>
#include <immintrin.h>

namespace abpfor
{

// ---------------------------------------------------------------------------
// Compile-time mask for SSE lanes
// ---------------------------------------------------------------------------

template <unsigned B> ABPFOR_INLINE __m128i sseMask()
{
    if constexpr (B == 32)
        return _mm_set1_epi32(-1);
    else
        return _mm_set1_epi32(static_cast<int>((1u << B) - 1u));
}

// ---------------------------------------------------------------------------
// UnpackGroup — extract group G's 4 values from the stripe array.
// ---------------------------------------------------------------------------
// Recurses at compile time over G = 0 .. MaxG-1.
//
// Template params:
//   B         — bit-width (compile-time constant, 1-32)
//   G         — current group index
//   MaxG      — total groups (32 for 128 elements)
//   LoadedIdx — which stripe is currently in register `iv` (-1 = none)

template <unsigned B, unsigned G, unsigned MaxG, int LoadedIdx> struct UnpackGroup
{
    static ABPFOR_INLINE void run(const __m128i*& ip, __m128i& iv, uint32_t* out, const __m128i& vmask)
    {
        constexpr unsigned bitOffset = G * B;
        constexpr int stripeIdx = static_cast<int>(bitOffset / 32u);
        constexpr unsigned shift = bitOffset % 32u;
        constexpr bool spans = (shift + B > 32u) && (B < 32u);

        // Load next stripe if we've moved past the current one
        if constexpr (stripeIdx > LoadedIdx) iv = _mm_loadu_si128(ip + stripeIdx);

        // Extract: shift right to align, then mask
        __m128i ov;
        if constexpr (shift == 0)
            ov = iv;
        else
            ov = _mm_srli_epi32(iv, shift);

        if constexpr (spans)
        {
            // Value straddles two stripes — OR in high bits from next stripe
            __m128i next = _mm_loadu_si128(ip + stripeIdx + 1);
            iv = next; // keep for subsequent groups reading from stripeIdx+1
            constexpr unsigned lo = 32u - shift;
            ov = _mm_or_si128(ov, _mm_slli_epi32(next, lo));
            ov = _mm_and_si128(ov, vmask);
        }
        else if constexpr (B < 32u)
        {
            ov = _mm_and_si128(ov, vmask);
        }

        // Store 4 values
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + G * 4), ov);

        // Recurse to next group
        constexpr int nextLoaded = spans ? stripeIdx + 1 : stripeIdx;
        UnpackGroup<B, G + 1, MaxG, nextLoaded>::run(ip, iv, out, vmask);
    }
};

// Base case: all groups processed
template <unsigned B, unsigned MaxG, int LoadedIdx> struct UnpackGroup<B, MaxG, MaxG, LoadedIdx>
{
    static ABPFOR_INLINE void run(const __m128i*&, __m128i&, uint32_t*, const __m128i&) {}
};

// ---------------------------------------------------------------------------
// unpackI4_impl<B> — unpack 128 elements at compile-time bit-width B
// ---------------------------------------------------------------------------

template <unsigned B> ABPFOR_INLINE const uint8_t* unpackI4_impl(const uint8_t* in, uint32_t* out)
{
    if constexpr (B == 0)
    {
        std::memset(out, 0, 128 * sizeof(uint32_t));
        return in;
    }
    else if constexpr (B == 32)
    {
        std::memcpy(out, in, 128 * sizeof(uint32_t));
        return in + 128 * sizeof(uint32_t);
    }
    else
    {
        const __m128i* ip = reinterpret_cast<const __m128i*>(in);
        __m128i iv = _mm_setzero_si128();
        const __m128i vmask = sseMask<B>();

        UnpackGroup<B, 0, 32, -1>::run(ip, iv, out, vmask);

        return in + packedBytes(128, B);
    }
}

// ---------------------------------------------------------------------------
// PackGroup — pack group G's 4 values into the stripe array.
// ---------------------------------------------------------------------------

template <unsigned B, unsigned G, unsigned MaxG> struct PackGroup
{
    static ABPFOR_INLINE void run(const uint32_t* in, __m128i* stripes)
    {
        constexpr unsigned bitOffset = G * B;
        constexpr unsigned stripeIdx = bitOffset / 32u;
        constexpr unsigned shift = bitOffset % 32u;
        constexpr bool spans = (shift + B > 32u) && (B < 32u);

        __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + G * 4));

        // Always OR — stripes are zero-initialised, so first write is safe.
        // Necessary because a prior group may have spanned into this stripe.
        stripes[stripeIdx] = _mm_or_si128(stripes[stripeIdx], _mm_slli_epi32(v, shift));

        if constexpr (spans)
        {
            constexpr unsigned lo = 32u - shift;
            stripes[stripeIdx + 1] = _mm_srli_epi32(v, lo);
        }

        PackGroup<B, G + 1, MaxG>::run(in, stripes);
    }
};

template <unsigned B, unsigned MaxG> struct PackGroup<B, MaxG, MaxG>
{
    static ABPFOR_INLINE void run(const uint32_t*, __m128i*) {}
};

// ---------------------------------------------------------------------------
// packI4_impl<B> — pack 128 elements at compile-time bit-width B
// ---------------------------------------------------------------------------

template <unsigned B> ABPFOR_INLINE uint8_t* packI4_impl(const uint32_t* in, uint8_t* out)
{
    if constexpr (B == 0)
    {
        return out;
    }
    else if constexpr (B == 32)
    {
        std::memcpy(out, in, 128 * sizeof(uint32_t));
        return out + 128 * sizeof(uint32_t);
    }
    else
    {
        constexpr unsigned totalBits = 32u * B; // 32 groups × B bits per group
        constexpr unsigned numStripes = (totalBits + 31u) / 32u;

        __m128i stripes[numStripes];
        std::memset(stripes, 0, sizeof(stripes));

        PackGroup<B, 0, 32>::run(in, stripes);

        // Write stripes to output
        unsigned totalBytes = packedBytes(128, B);
        std::memcpy(out, stripes, totalBytes);
        return out + totalBytes;
    }
}

// ---------------------------------------------------------------------------
// Runtime dispatch via function-pointer table
// ---------------------------------------------------------------------------

using UnpackI4Fn = const uint8_t* (*)(const uint8_t*, uint32_t*);
using PackI4Fn = uint8_t* (*)(const uint32_t*, uint8_t*);

namespace detail
{

template <unsigned B> const uint8_t* unpackI4_dispatch(const uint8_t* in, uint32_t* out)
{
    return unpackI4_impl<B>(in, out);
}

template <unsigned B> uint8_t* packI4_dispatch(const uint8_t* in, uint32_t* out)
{
    return packI4_impl<B>(reinterpret_cast<const uint32_t*>(in), out);
}

// shortcut: explicit table — 33 entries, generated by the compiler
inline const UnpackI4Fn kUnpackI4Table[33] = {
    unpackI4_impl<0>,  unpackI4_impl<1>,  unpackI4_impl<2>,  unpackI4_impl<3>,  unpackI4_impl<4>,  unpackI4_impl<5>,
    unpackI4_impl<6>,  unpackI4_impl<7>,  unpackI4_impl<8>,  unpackI4_impl<9>,  unpackI4_impl<10>, unpackI4_impl<11>,
    unpackI4_impl<12>, unpackI4_impl<13>, unpackI4_impl<14>, unpackI4_impl<15>, unpackI4_impl<16>, unpackI4_impl<17>,
    unpackI4_impl<18>, unpackI4_impl<19>, unpackI4_impl<20>, unpackI4_impl<21>, unpackI4_impl<22>, unpackI4_impl<23>,
    unpackI4_impl<24>, unpackI4_impl<25>, unpackI4_impl<26>, unpackI4_impl<27>, unpackI4_impl<28>, unpackI4_impl<29>,
    unpackI4_impl<30>, unpackI4_impl<31>, unpackI4_impl<32>,
};

inline const PackI4Fn kPackI4Table[33] = {
    packI4_impl<0>,  packI4_impl<1>,  packI4_impl<2>,  packI4_impl<3>,  packI4_impl<4>,  packI4_impl<5>,
    packI4_impl<6>,  packI4_impl<7>,  packI4_impl<8>,  packI4_impl<9>,  packI4_impl<10>, packI4_impl<11>,
    packI4_impl<12>, packI4_impl<13>, packI4_impl<14>, packI4_impl<15>, packI4_impl<16>, packI4_impl<17>,
    packI4_impl<18>, packI4_impl<19>, packI4_impl<20>, packI4_impl<21>, packI4_impl<22>, packI4_impl<23>,
    packI4_impl<24>, packI4_impl<25>, packI4_impl<26>, packI4_impl<27>, packI4_impl<28>, packI4_impl<29>,
    packI4_impl<30>, packI4_impl<31>, packI4_impl<32>,
};

} // namespace detail

// ---------------------------------------------------------------------------
// Public API: packI4 / unpackI4  (runtime bit-width dispatch)
// ---------------------------------------------------------------------------

inline const uint8_t* unpackI4(const uint8_t* in, uint32_t* out, unsigned b)
{
    return detail::kUnpackI4Table[b](in, out);
}

inline uint8_t* packI4(const uint32_t* in, uint8_t* out, unsigned b)
{
    return detail::kPackI4Table[b](in, out);
}


} // namespace abpfor
