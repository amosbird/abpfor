#pragma once

// abpfor AVX2 fused decode — single-pass unpack + patch + prefixSum (256 elements).
//
// Same design as sse_fused.h but with __m256i (8 lanes, 32 groups).
//
// Key AVX2 difference: pshufb is lane-local (operates independently on
// the low and high 128-bit halves).  Outlier scatter therefore splits
// each group's 8-bit outlier mask into two 4-bit nibbles and applies
// separate pshufb per half-lane.
//
// Prefix sum also requires cross-lane carry propagation:
//   1. In-lane scan (same as SSE, per 128-bit half)
//   2. Broadcast high element of low half → add to high half

#include "../core/bits.h"
#include "avx2_pack.h"
#include "sse_fused.h"

#include <immintrin.h>

namespace abpfor
{

// ---------------------------------------------------------------------------
// Scatter LUT for AVX2 — reuse the SSE 4-bit LUT per 128-bit half-lane.
// ---------------------------------------------------------------------------

namespace detail
{

// Same table as sse_fused.h, but we declare a separate copy to avoid
// cross-header dependency.  16 entries × 16 bytes = 256 bytes.
alignas(16) inline const int8_t kScatterLUT8[16][16] = {
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, 0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1},
    {0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, -1, -1, -1, -1},
    {0, 1, 2, 3, -1, -1, -1, -1, 4, 5, 6, 7, -1, -1, -1, -1},
    {-1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1, -1},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3},
    {0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 6, 7},
    {-1, -1, -1, -1, 0, 1, 2, 3, -1, -1, -1, -1, 4, 5, 6, 7},
    {0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1, -1, 8, 9, 10, 11},
    {-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7},
    {0, 1, 2, 3, -1, -1, -1, -1, 4, 5, 6, 7, 8, 9, 10, 11},
    {-1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
};

} // namespace detail

// ---------------------------------------------------------------------------
// AVX2 prefix sum: inclusive scan of 8 × 32-bit lanes with carry.
// ---------------------------------------------------------------------------
// Input:  ov = [a, b, c, d, e, f, g, h],  sv = broadcast of previous carry
// Output: ov = [S+a, S+a+b, ..., S+a+..+h],  sv = broadcast of last element
// where S = previous carry + per-lane offset from cv.

ABPFOR_INLINE __m256i avx2PrefixSum(__m256i ov, __m256i& sv, const __m256i& cv)
{
    // In-lane scan (each 128-bit half independently)
    ov = _mm256_add_epi32(ov, _mm256_slli_si256(ov, 4));
    ov = _mm256_add_epi32(ov, _mm256_slli_si256(ov, 8));

    // Cross-lane: broadcast lane[3] of low half → add to high half
    // permute2x128(ov, ov, 0x00) gives [low, low]
    // shuffle(_, 0xFF) broadcasts element 3 → [lo[3], lo[3], lo[3], lo[3], lo[3]...]
    __m256i lo_bcast = _mm256_shuffle_epi32(_mm256_permute2x128_si256(ov, ov, 0x00), _MM_SHUFFLE(3, 3, 3, 3));
    // Zero the low half of lo_bcast (only add to high half)
    __m256i cross = _mm256_permute2x128_si256(_mm256_setzero_si256(), lo_bcast, 0x20);

    ov = _mm256_add_epi32(ov, cross);

    // Add carry + offset
    ov = _mm256_add_epi32(ov, _mm256_add_epi32(sv, cv));

    // Update carry: broadcast last element (lane 7)
    sv = _mm256_shuffle_epi32(_mm256_permute2x128_si256(ov, ov, 0x11), _MM_SHUFFLE(3, 3, 3, 3));

    return ov;
}

// ---------------------------------------------------------------------------
// AVX2 outlier scatter — split 8-lane mask into two 4-bit nibbles.
// ---------------------------------------------------------------------------
// Each group has 8 potential outlier positions → 8-bit mask from bitmap.
// Low nibble controls low 128-bit half, high nibble controls high half.
// Each half uses the same 16-entry pshufb LUT as SSE.

ABPFOR_INLINE __m256i avx2Scatter(const uint32_t*& pex, unsigned mask8, unsigned B)
{
    unsigned loNib = mask8 & 0xF;
    unsigned hiNib = (mask8 >> 4) & 0xF;

    // Load up to 8 contiguous residuals (may over-read, safe with padding)
    __m128i excLo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pex));
    unsigned loCount = __builtin_popcount(loNib);
    __m128i excHi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pex + loCount));

    // Shift residuals left by B bits
    __m128i excLoShifted = _mm_slli_epi32(excLo, B);
    __m128i excHiShifted = _mm_slli_epi32(excHi, B);

    // Scatter via LUT
    __m128i maskLo = _mm_load_si128(reinterpret_cast<const __m128i*>(detail::kScatterLUT8[loNib]));
    __m128i maskHi = _mm_load_si128(reinterpret_cast<const __m128i*>(detail::kScatterLUT8[hiNib]));

    __m128i scatLo = _mm_shuffle_epi8(excLoShifted, maskLo);
    __m128i scatHi = _mm_shuffle_epi8(excHiShifted, maskHi);

    // Combine into __m256i
    __m256i result = _mm256_inserti128_si256(_mm256_castsi128_si256(scatLo), scatHi, 1);

    pex += loCount + __builtin_popcount(hiNib);

    return result;
}

// ---------------------------------------------------------------------------
// FusedGroup8 — one template handles extract + optional patch + optional delta.
// ---------------------------------------------------------------------------

template <unsigned B, unsigned G, unsigned MaxG, int LoadedIdx, bool Delta, bool Patch, bool MinusOne = false> struct FusedGroup8
{
    static ABPFOR_INLINE void run(const __m256i*& ip, __m256i& iv, uint32_t* out, const __m256i& vmask, __m256i& sv,
                                  const __m256i& cv, const uint64_t* bitmap, const uint32_t*& pex)
    {
        constexpr unsigned bitOffset = G * B;
        constexpr int stripeIdx = static_cast<int>(bitOffset / 32u);
        constexpr unsigned shift = bitOffset % 32u;
        constexpr bool spans = (shift + B > 32u) && (B < 32u);

        // --- Extract ---
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

        // --- Patch ---
        if constexpr (Patch)
        {
            // 8 bits from bitmap for this group (256 elements → 4 uint64_t words)
            const uint64_t bmpWord = bitmap[G / 8];
            const unsigned mask8 = static_cast<unsigned>((bmpWord >> ((G % 8) * 8)) & 0xFFu);

            ov = _mm256_add_epi32(ov, avx2Scatter(pex, mask8, B));
        }

        // --- Delta ---
        if constexpr (Delta)
        {
            ov = avx2PrefixSum(ov, sv, cv);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + G * 8), ov);
        }
        else
        {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + G * 8), ov);
        }

        // --- Recurse ---
        constexpr int nextLoaded = spans ? stripeIdx + 1 : stripeIdx;
        FusedGroup8<B, G + 1, MaxG, nextLoaded, Delta, Patch, MinusOne>::run(ip, iv, out, vmask, sv, cv, bitmap, pex);
    }
};

template <unsigned B, unsigned MaxG, int LoadedIdx, bool Delta, bool Patch, bool MinusOne>
struct FusedGroup8<B, MaxG, MaxG, LoadedIdx, Delta, Patch, MinusOne>
{
    static ABPFOR_INLINE void run(const __m256i*&, __m256i&, uint32_t*, const __m256i&, __m256i&, const __m256i&,
                                  const uint64_t*, const uint32_t*&)
    {
    }
};

// ---------------------------------------------------------------------------
// fusedDecodeI8_impl<B, Delta, Patch>
// ---------------------------------------------------------------------------

template <unsigned B, bool Delta, bool Patch, bool MinusOne = false>
ABPFOR_INLINE void fusedDecodeI8_impl(const uint8_t* in, uint32_t* out, uint32_t& carry, const uint64_t* bitmap,
                                      const uint32_t* residuals)
{
    if constexpr (B == 0)
    {
        // shortcut: scalar fallback (rare)
        const uint32_t* pex = residuals;
        if constexpr (Delta)
        {
            for (unsigned i = 0; i < 256; ++i)
            {
                uint32_t val = 0;
                if constexpr (Patch)
                {
                    if (bitmap[i / 64] & (uint64_t(1) << (i % 64))) val = *pex++;
                }
                carry += (MinusOne ? 1u : 0u) + val;
                out[i] = carry;
            }
        }
        else
        {
            std::memset(out, 0, 256 * sizeof(uint32_t));
            if constexpr (Patch)
            {
                for (unsigned i = 0; i < 256; ++i)
                {
                    if (bitmap[i / 64] & (uint64_t(1) << (i % 64))) out[i] = *pex++;
                }
            }
        }
        return;
    }

    if constexpr (B == 32)
    {
        std::memcpy(out, in, 256 * sizeof(uint32_t));
        if constexpr (Delta)
        {
            __m256i sv = _mm256_set1_epi32(static_cast<int>(carry));
            const __m256i cv = MinusOne ? _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8) : _mm256_setzero_si256();
            for (unsigned g = 0; g < 32; ++g)
            {
                __m256i ov = _mm256_loadu_si256(reinterpret_cast<__m256i*>(out + g * 8));
                ov = avx2PrefixSum(ov, sv, cv);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + g * 8), ov);
            }
            carry = static_cast<uint32_t>(_mm256_extract_epi32(sv, 0));
        }
        return;
    }

    // General case
    const __m256i* ip = reinterpret_cast<const __m256i*>(in);
    __m256i iv = _mm256_setzero_si256();
    const __m256i vmask = avx2Mask<B>();

    __m256i sv = _mm256_set1_epi32(static_cast<int>(carry));
    const __m256i cv = (Delta && MinusOne) ? _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8) : _mm256_setzero_si256();

    const uint32_t* pex = residuals;

    FusedGroup8<B, 0, 32, -1, Delta, Patch, MinusOne>::run(ip, iv, out, vmask, sv, cv, bitmap, pex);

    if constexpr (Delta) carry = static_cast<uint32_t>(_mm256_extract_epi32(sv, 0));
}

// ---------------------------------------------------------------------------
// Runtime dispatch
// ---------------------------------------------------------------------------

template <bool Delta, bool Patch, bool MinusOne>
using FusedI8Fn = void (*)(const uint8_t*, uint32_t*, uint32_t&, const uint64_t*, const uint32_t*);

namespace detail
{

template <bool Delta, bool Patch, bool MinusOne = false> struct FusedI8Table
{
    static inline const FusedI8Fn<Delta, Patch, MinusOne> table[33] = {
        fusedDecodeI8_impl<0, Delta, Patch, MinusOne>,  fusedDecodeI8_impl<1, Delta, Patch, MinusOne>,
        fusedDecodeI8_impl<2, Delta, Patch, MinusOne>,  fusedDecodeI8_impl<3, Delta, Patch, MinusOne>,
        fusedDecodeI8_impl<4, Delta, Patch, MinusOne>,  fusedDecodeI8_impl<5, Delta, Patch, MinusOne>,
        fusedDecodeI8_impl<6, Delta, Patch, MinusOne>,  fusedDecodeI8_impl<7, Delta, Patch, MinusOne>,
        fusedDecodeI8_impl<8, Delta, Patch, MinusOne>,  fusedDecodeI8_impl<9, Delta, Patch, MinusOne>,
        fusedDecodeI8_impl<10, Delta, Patch, MinusOne>, fusedDecodeI8_impl<11, Delta, Patch, MinusOne>,
        fusedDecodeI8_impl<12, Delta, Patch, MinusOne>, fusedDecodeI8_impl<13, Delta, Patch, MinusOne>,
        fusedDecodeI8_impl<14, Delta, Patch, MinusOne>, fusedDecodeI8_impl<15, Delta, Patch, MinusOne>,
        fusedDecodeI8_impl<16, Delta, Patch, MinusOne>, fusedDecodeI8_impl<17, Delta, Patch, MinusOne>,
        fusedDecodeI8_impl<18, Delta, Patch, MinusOne>, fusedDecodeI8_impl<19, Delta, Patch, MinusOne>,
        fusedDecodeI8_impl<20, Delta, Patch, MinusOne>, fusedDecodeI8_impl<21, Delta, Patch, MinusOne>,
        fusedDecodeI8_impl<22, Delta, Patch, MinusOne>, fusedDecodeI8_impl<23, Delta, Patch, MinusOne>,
        fusedDecodeI8_impl<24, Delta, Patch, MinusOne>, fusedDecodeI8_impl<25, Delta, Patch, MinusOne>,
        fusedDecodeI8_impl<26, Delta, Patch, MinusOne>, fusedDecodeI8_impl<27, Delta, Patch, MinusOne>,
        fusedDecodeI8_impl<28, Delta, Patch, MinusOne>, fusedDecodeI8_impl<29, Delta, Patch, MinusOne>,
        fusedDecodeI8_impl<30, Delta, Patch, MinusOne>, fusedDecodeI8_impl<31, Delta, Patch, MinusOne>,
        fusedDecodeI8_impl<32, Delta, Patch, MinusOne>,
    };
};

} // namespace detail

// AVX2 is a hard compile-time baseline; see avx2_pack.h.
//
// UB: b must be <= 32. Larger values index past the end of the dispatch table.
template <bool Delta, bool Patch, bool MinusOne = false>
inline void fusedDecodeI8(const uint8_t* in, uint32_t* out, unsigned b, uint32_t& carry, const uint64_t* bitmap,
                          const uint32_t* residuals)
{
    detail::FusedI8Table<Delta, Patch, MinusOne>::table[b](in, out, carry, bitmap, residuals);
}

} // namespace abpfor
