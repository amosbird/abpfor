#pragma once

// abpfor AVX2 interleaved (Interleave8) pack/unpack — 256 elements, 8-lane.
//
// Same principle as sse_pack.h but with __m256i (8 × 32-bit lanes).
// 256 values → 32 groups × 8 lanes.  Group g's B-bit values sit at
// bit offset g*B within each 32-bit lane of a 256-bit stripe.

#include "../core/bits.h"
#include "sse_pack.h"

#include <cstring>
#include <immintrin.h>

namespace abpfor
{

// ---------------------------------------------------------------------------
// Compile-time mask
// ---------------------------------------------------------------------------

template <unsigned B> ABPFOR_INLINE __m256i avx2Mask()
{
    if constexpr (B == 32)
        return _mm256_set1_epi32(-1);
    else
        return _mm256_set1_epi32(static_cast<int>((1u << B) - 1u));
}

// ---------------------------------------------------------------------------
// UnpackGroup8 — extract group G's 8 values from the stripe array.
// ---------------------------------------------------------------------------

template <unsigned B, unsigned G, unsigned MaxG, int LoadedIdx> struct UnpackGroup8
{
    static ABPFOR_INLINE void run(const __m256i*& ip, __m256i& iv, uint32_t* out, const __m256i& vmask)
    {
        constexpr unsigned bitOffset = G * B;
        constexpr int stripeIdx = static_cast<int>(bitOffset / 32u);
        constexpr unsigned shift = bitOffset % 32u;
        constexpr bool spans = (shift + B > 32u) && (B < 32u);

        if constexpr (stripeIdx > LoadedIdx) iv = _mm256_loadu_si256(ip + stripeIdx);

        __m256i ov;
        if constexpr (shift == 0)
            ov = iv;
        else
            ov = _mm256_srli_epi32(iv, shift);

        if constexpr (spans)
        {
            __m256i next = _mm256_loadu_si256(ip + stripeIdx + 1);
            iv = next;
            constexpr unsigned lo = 32u - shift;
            ov = _mm256_or_si256(ov, _mm256_slli_epi32(next, lo));
            ov = _mm256_and_si256(ov, vmask);
        }
        else if constexpr (B < 32u)
        {
            ov = _mm256_and_si256(ov, vmask);
        }

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + G * 8), ov);

        constexpr int nextLoaded = spans ? stripeIdx + 1 : stripeIdx;
        UnpackGroup8<B, G + 1, MaxG, nextLoaded>::run(ip, iv, out, vmask);
    }
};

template <unsigned B, unsigned MaxG, int LoadedIdx> struct UnpackGroup8<B, MaxG, MaxG, LoadedIdx>
{
    static ABPFOR_INLINE void run(const __m256i*&, __m256i&, uint32_t*, const __m256i&) {}
};

// ---------------------------------------------------------------------------
// unpackI8_impl<B>
// ---------------------------------------------------------------------------

template <unsigned B> ABPFOR_INLINE const uint8_t* unpackI8_impl(const uint8_t* in, uint32_t* out)
{
    if constexpr (B == 0)
    {
        std::memset(out, 0, 256 * sizeof(uint32_t));
        return in;
    }
    else if constexpr (B == 32)
    {
        std::memcpy(out, in, 256 * sizeof(uint32_t));
        return in + 256 * sizeof(uint32_t);
    }
    else
    {
        const __m256i* ip = reinterpret_cast<const __m256i*>(in);
        __m256i iv = _mm256_setzero_si256();
        const __m256i vmask = avx2Mask<B>();

        UnpackGroup8<B, 0, 32, -1>::run(ip, iv, out, vmask);

        return in + packedBytes(256, B);
    }
}

// ---------------------------------------------------------------------------
// PackGroup8 — pack group G's 8 values into the stripe array.
// ---------------------------------------------------------------------------

template <unsigned B, unsigned G, unsigned MaxG> struct PackGroup8
{
    static ABPFOR_INLINE void run(const uint32_t* in, __m256i* stripes)
    {
        constexpr unsigned bitOffset = G * B;
        constexpr unsigned stripeIdx = bitOffset / 32u;
        constexpr unsigned shift = bitOffset % 32u;
        constexpr bool spans = (shift + B > 32u) && (B < 32u);

        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + G * 8));

        // Always OR — stripes are zero-initialised
        stripes[stripeIdx] = _mm256_or_si256(stripes[stripeIdx], _mm256_slli_epi32(v, shift));

        if constexpr (spans)
        {
            constexpr unsigned lo = 32u - shift;
            stripes[stripeIdx + 1] = _mm256_srli_epi32(v, lo);
        }

        PackGroup8<B, G + 1, MaxG>::run(in, stripes);
    }
};

template <unsigned B, unsigned MaxG> struct PackGroup8<B, MaxG, MaxG>
{
    static ABPFOR_INLINE void run(const uint32_t*, __m256i*) {}
};

// ---------------------------------------------------------------------------
// packI8_impl<B>
// ---------------------------------------------------------------------------

template <unsigned B> ABPFOR_INLINE uint8_t* packI8_impl(const uint32_t* in, uint8_t* out)
{
    if constexpr (B == 0)
    {
        return out;
    }
    else if constexpr (B == 32)
    {
        std::memcpy(out, in, 256 * sizeof(uint32_t));
        return out + 256 * sizeof(uint32_t);
    }
    else
    {
        constexpr unsigned totalBits = 32u * B;
        constexpr unsigned numStripes = (totalBits + 31u) / 32u;

        alignas(32) __m256i stripes[numStripes];
        std::memset(stripes, 0, sizeof(stripes));

        PackGroup8<B, 0, 32>::run(in, stripes);

        unsigned totalBytes = packedBytes(256, B);
        std::memcpy(out, stripes, totalBytes);
        return out + totalBytes;
    }
}

// ---------------------------------------------------------------------------
// Runtime dispatch
// ---------------------------------------------------------------------------

using UnpackI8Fn = const uint8_t* (*)(const uint8_t*, uint32_t*);
using PackI8Fn = uint8_t* (*)(const uint32_t*, uint8_t*);

namespace detail
{

inline const UnpackI8Fn kUnpackI8Table[33] = {
    unpackI8_impl<0>,  unpackI8_impl<1>,  unpackI8_impl<2>,  unpackI8_impl<3>,  unpackI8_impl<4>,  unpackI8_impl<5>,
    unpackI8_impl<6>,  unpackI8_impl<7>,  unpackI8_impl<8>,  unpackI8_impl<9>,  unpackI8_impl<10>, unpackI8_impl<11>,
    unpackI8_impl<12>, unpackI8_impl<13>, unpackI8_impl<14>, unpackI8_impl<15>, unpackI8_impl<16>, unpackI8_impl<17>,
    unpackI8_impl<18>, unpackI8_impl<19>, unpackI8_impl<20>, unpackI8_impl<21>, unpackI8_impl<22>, unpackI8_impl<23>,
    unpackI8_impl<24>, unpackI8_impl<25>, unpackI8_impl<26>, unpackI8_impl<27>, unpackI8_impl<28>, unpackI8_impl<29>,
    unpackI8_impl<30>, unpackI8_impl<31>, unpackI8_impl<32>,
};

inline const PackI8Fn kPackI8Table[33] = {
    packI8_impl<0>,  packI8_impl<1>,  packI8_impl<2>,  packI8_impl<3>,  packI8_impl<4>,  packI8_impl<5>,
    packI8_impl<6>,  packI8_impl<7>,  packI8_impl<8>,  packI8_impl<9>,  packI8_impl<10>, packI8_impl<11>,
    packI8_impl<12>, packI8_impl<13>, packI8_impl<14>, packI8_impl<15>, packI8_impl<16>, packI8_impl<17>,
    packI8_impl<18>, packI8_impl<19>, packI8_impl<20>, packI8_impl<21>, packI8_impl<22>, packI8_impl<23>,
    packI8_impl<24>, packI8_impl<25>, packI8_impl<26>, packI8_impl<27>, packI8_impl<28>, packI8_impl<29>,
    packI8_impl<30>, packI8_impl<31>, packI8_impl<32>,
};

} // namespace detail

// AVX2 is a hard compile-time baseline (the whole library is built with -mavx2,
// so no branch here could produce a non-AVX2 code path anyway).
//
// UB: b must be <= 32. Larger values index past the end of the dispatch table.
// The caller owns input validation; decoding an untrusted byte stream without
// first checking the header's bit-width field is undefined behaviour.
inline const uint8_t* unpackI8(const uint8_t* in, uint32_t* out, unsigned b)
{
    return detail::kUnpackI8Table[b](in, out);
}

// UB: b must be <= 32 (see unpackI8).
inline uint8_t* packI8(const uint32_t* in, uint8_t* out, unsigned b)
{
    return detail::kPackI8Table[b](in, out);
}

} // namespace abpfor
