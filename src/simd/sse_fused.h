#pragma once

// abpfor SSE fused decode — single-pass unpack + patch + prefixSum.
//
// Extends the UnpackGroup template with two optional fused operations:
//   1. Outlier patching: pshufb + LUT scatters residuals into correct lanes
//   2. Prefix sum (delta-1): running add across groups
//
// Both are controlled by compile-time bool template params, so the
// compiler eliminates dead code for modes that don't use them.

#include "../core/bits.h"
#include "sse_pack.h" // for sseMask, packedBytes

#include <immintrin.h>

namespace abpfor
{

// ---------------------------------------------------------------------------
// pshufb scatter LUT — maps a 4-bit mask to a shuffle control.
//
// For mask m (which 4-bit nibble says which of 4 lanes have outliers),
// the shuffle moves packed residuals from contiguous positions to the
// correct sparse lane positions. Unused lanes get 0xFF → zeroed by pshufb.
// ---------------------------------------------------------------------------

namespace detail
{

alignas(16) inline const int8_t kScatterLUT[16][16] = {
    // 0000: no outliers
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    // 0001: lane 0
    {0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    // 0010: lane 1
    {-1, -1, -1, -1, 0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1},
    // 0011: lanes 0,1
    {0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1},
    // 0100: lane 2
    {-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, -1, -1, -1, -1},
    // 0101: lanes 0,2
    {0, 1, 2, 3, -1, -1, -1, -1, 4, 5, 6, 7, -1, -1, -1, -1},
    // 0110: lanes 1,2
    {-1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1, -1},
    // 0111: lanes 0,1,2
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -1, -1, -1, -1},
    // 1000: lane 3
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3},
    // 1001: lanes 0,3
    {0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 6, 7},
    // 1010: lanes 1,3
    {-1, -1, -1, -1, 0, 1, 2, 3, -1, -1, -1, -1, 4, 5, 6, 7},
    // 1011: lanes 0,1,3
    {0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1, -1, 8, 9, 10, 11},
    // 1100: lanes 2,3
    {-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7},
    // 1101: lanes 0,2,3
    {0, 1, 2, 3, -1, -1, -1, -1, 4, 5, 6, 7, 8, 9, 10, 11},
    // 1110: lanes 1,2,3
    {-1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
    // 1111: all lanes
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
};

} // namespace detail

// ---------------------------------------------------------------------------
// FusedGroup — one template handles extract + optional patch + optional delta.
// ---------------------------------------------------------------------------
// Template params:
//   B          — bit-width
//   G          — current group (0..31)
//   MaxG       — total groups
//   LoadedIdx  — last loaded stripe index (-1 = none)
//   Delta      — fuse prefix sum
//   Patch      — fuse outlier scatter

template <unsigned B, unsigned G, unsigned MaxG, int LoadedIdx, bool Delta, bool Patch, bool MinusOne = false> struct FusedGroup
{
    static ABPFOR_INLINE void run(const __m128i*& ip, __m128i& iv, uint32_t* out, const __m128i& vmask, __m128i& sv,
                                  const __m128i& cv, const uint64_t* bitmap, const uint32_t*& pex)
    {
        constexpr unsigned bitOffset = G * B;
        constexpr int stripeIdx = static_cast<int>(bitOffset / 32u);
        constexpr unsigned shift = bitOffset % 32u;
        constexpr bool spans = (shift + B > 32u) && (B < 32u);

        // --- Step 1: Extract (identical to UnpackGroup) ---

        if constexpr (stripeIdx > LoadedIdx) iv = _mm_loadu_si128(ip + stripeIdx);

        __m128i ov;
        if constexpr (shift == 0)
            ov = iv;
        else
            ov = _mm_srli_epi32(iv, shift);

        if constexpr (spans)
        {
            __m128i next = _mm_loadu_si128(ip + stripeIdx + 1);
            iv = next;
            constexpr unsigned lo = 32u - shift;
            ov = _mm_or_si128(ov, _mm_slli_epi32(next, lo));
            ov = _mm_and_si128(ov, vmask);
        }
        else if constexpr (B < 32u)
        {
            ov = _mm_and_si128(ov, vmask);
        }

        // --- Step 2: Patch outliers via pshufb ---

        if constexpr (Patch)
        {
            // Read 4-bit nibble from bitmap for this group
            const uint64_t bmpWord = (G < 16) ? bitmap[0] : bitmap[1];
            const unsigned nibble = static_cast<unsigned>((bmpWord >> ((G % 16) * 4)) & 0xFu);

            // Load contiguous residuals, scatter to correct lanes via LUT
            __m128i exc = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pex));
            __m128i exc_shifted = _mm_slli_epi32(exc, B);
            __m128i shufMask = _mm_load_si128(reinterpret_cast<const __m128i*>(detail::kScatterLUT[nibble]));
            __m128i scattered = _mm_shuffle_epi8(exc_shifted, shufMask);

            ov = _mm_add_epi32(ov, scattered);

            pex += __builtin_popcount(nibble);
        }

        // --- Step 3: Prefix sum (delta-1 decode) ---

        if constexpr (Delta)
        {
            // In-register inclusive prefix sum of 4 lanes:
            //   ov = [a, b, c, d]
            //   → [a, a+b, a+b+c, a+b+c+d]
            ov = _mm_add_epi32(ov, _mm_slli_si128(ov, 4));
            ov = _mm_add_epi32(ov, _mm_slli_si128(ov, 8));

            // Add carry from previous group + per-lane offset [1,2,3,4]
            ov = _mm_add_epi32(ov, _mm_add_epi32(sv, cv));

            // Store
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out + G * 4), ov);

            // Update carry: broadcast last element
            sv = _mm_shuffle_epi32(ov, _MM_SHUFFLE(3, 3, 3, 3));
        }
        else
        {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out + G * 4), ov);
        }

        // --- Recurse ---
        constexpr int nextLoaded = spans ? stripeIdx + 1 : stripeIdx;
        FusedGroup<B, G + 1, MaxG, nextLoaded, Delta, Patch, MinusOne>::run(ip, iv, out, vmask, sv, cv, bitmap, pex);
    }
};

// Base case
template <unsigned B, unsigned MaxG, int LoadedIdx, bool Delta, bool Patch, bool MinusOne>
struct FusedGroup<B, MaxG, MaxG, LoadedIdx, Delta, Patch, MinusOne>
{
    static ABPFOR_INLINE void run(const __m128i*&, __m128i&, uint32_t*, const __m128i&, __m128i&, const __m128i&,
                                  const uint64_t*, const uint32_t*&)
    {
    }
};

// ---------------------------------------------------------------------------
// fusedDecodeI4_impl<B, Delta, Patch> — entry point
// ---------------------------------------------------------------------------

template <unsigned B, bool Delta, bool Patch, bool MinusOne = false>
ABPFOR_INLINE void fusedDecodeI4_impl(const uint8_t* in, uint32_t* out, uint32_t& carry, const uint64_t* bitmap,
                                      const uint32_t* residuals)
{
    if constexpr (B == 0)
    {
        if constexpr (Delta)
        {
            // b=0 + delta: out[i] = carry + (i+1) + patches (delta1) or carry + patches (delta0)
            // shortcut: scalar fallback for b=0 (rare case)
            const uint32_t* pex = residuals;
            for (unsigned g = 0; g < 32; ++g)
            {
                for (unsigned lane = 0; lane < 4; ++lane)
                {
                    uint32_t val = 0;
                    if constexpr (Patch)
                    {
                        const uint64_t bw = (g < 16) ? bitmap[0] : bitmap[1];
                        unsigned nib = static_cast<unsigned>((bw >> ((g % 16) * 4)) & 0xFu);
                        if (nib & (1u << lane)) val = *pex++;
                    }
                    carry += (MinusOne ? 1u : 0u) + val;
                    out[g * 4 + lane] = carry;
                }
            }
        }
        else
        {
            std::memset(out, 0, 128 * sizeof(uint32_t));
            if constexpr (Patch)
            {
                // b=0, no delta, but has patches — just scatter
                const uint32_t* pex = residuals;
                for (unsigned g = 0; g < 32; ++g)
                {
                    const uint64_t bw = (g < 16) ? bitmap[0] : bitmap[1];
                    unsigned nib = static_cast<unsigned>((bw >> ((g % 16) * 4)) & 0xFu);
                    for (unsigned lane = 0; lane < 4; ++lane)
                    {
                        if (nib & (1u << lane)) out[g * 4 + lane] = *pex++;
                    }
                }
            }
        }
        return;
    }

    if constexpr (B == 32)
    {
        std::memcpy(out, in, 128 * sizeof(uint32_t));
        // Patch and delta on already-unpacked data (full width, no masking needed)
        if constexpr (Patch)
        {
            // B=32 means no outliers are possible (max bits = 32),
            // so this branch is dead code. But keep for template completeness.
        }
        if constexpr (Delta)
        {
            __m128i sv = _mm_set1_epi32(static_cast<int>(carry));
            const __m128i cv = MinusOne ? _mm_setr_epi32(1, 2, 3, 4) : _mm_setzero_si128();
            for (unsigned g = 0; g < 32; ++g)
            {
                __m128i ov = _mm_loadu_si128(reinterpret_cast<__m128i*>(out + g * 4));
                ov = _mm_add_epi32(ov, _mm_slli_si128(ov, 4));
                ov = _mm_add_epi32(ov, _mm_slli_si128(ov, 8));
                ov = _mm_add_epi32(ov, _mm_add_epi32(sv, cv));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(out + g * 4), ov);
                sv = _mm_shuffle_epi32(ov, _MM_SHUFFLE(3, 3, 3, 3));
            }
            carry = static_cast<uint32_t>(_mm_cvtsi128_si32(sv));
        }
        return;
    }

    // General case: fused template recursion
    const __m128i* ip = reinterpret_cast<const __m128i*>(in);
    __m128i iv = _mm_setzero_si128();
    const __m128i vmask = sseMask<B>();

    __m128i sv = _mm_set1_epi32(static_cast<int>(carry));
    const __m128i cv = (Delta && MinusOne) ? _mm_setr_epi32(1, 2, 3, 4) : _mm_setzero_si128();

    const uint32_t* pex = residuals;

    FusedGroup<B, 0, 32, -1, Delta, Patch, MinusOne>::run(ip, iv, out, vmask, sv, cv, bitmap, pex);

    if constexpr (Delta) carry = static_cast<uint32_t>(_mm_cvtsi128_si32(sv));
}

// ---------------------------------------------------------------------------
// Runtime dispatch
// ---------------------------------------------------------------------------

template <bool Delta, bool Patch, bool MinusOne>
using FusedI4Fn = void (*)(const uint8_t*, uint32_t*, uint32_t&, const uint64_t*, const uint32_t*);

namespace detail
{

template <bool Delta, bool Patch, bool MinusOne = false> struct FusedI4Table
{
    static inline const FusedI4Fn<Delta, Patch, MinusOne> table[33] = {
        fusedDecodeI4_impl<0, Delta, Patch, MinusOne>,  fusedDecodeI4_impl<1, Delta, Patch, MinusOne>,
        fusedDecodeI4_impl<2, Delta, Patch, MinusOne>,  fusedDecodeI4_impl<3, Delta, Patch, MinusOne>,
        fusedDecodeI4_impl<4, Delta, Patch, MinusOne>,  fusedDecodeI4_impl<5, Delta, Patch, MinusOne>,
        fusedDecodeI4_impl<6, Delta, Patch, MinusOne>,  fusedDecodeI4_impl<7, Delta, Patch, MinusOne>,
        fusedDecodeI4_impl<8, Delta, Patch, MinusOne>,  fusedDecodeI4_impl<9, Delta, Patch, MinusOne>,
        fusedDecodeI4_impl<10, Delta, Patch, MinusOne>, fusedDecodeI4_impl<11, Delta, Patch, MinusOne>,
        fusedDecodeI4_impl<12, Delta, Patch, MinusOne>, fusedDecodeI4_impl<13, Delta, Patch, MinusOne>,
        fusedDecodeI4_impl<14, Delta, Patch, MinusOne>, fusedDecodeI4_impl<15, Delta, Patch, MinusOne>,
        fusedDecodeI4_impl<16, Delta, Patch, MinusOne>, fusedDecodeI4_impl<17, Delta, Patch, MinusOne>,
        fusedDecodeI4_impl<18, Delta, Patch, MinusOne>, fusedDecodeI4_impl<19, Delta, Patch, MinusOne>,
        fusedDecodeI4_impl<20, Delta, Patch, MinusOne>, fusedDecodeI4_impl<21, Delta, Patch, MinusOne>,
        fusedDecodeI4_impl<22, Delta, Patch, MinusOne>, fusedDecodeI4_impl<23, Delta, Patch, MinusOne>,
        fusedDecodeI4_impl<24, Delta, Patch, MinusOne>, fusedDecodeI4_impl<25, Delta, Patch, MinusOne>,
        fusedDecodeI4_impl<26, Delta, Patch, MinusOne>, fusedDecodeI4_impl<27, Delta, Patch, MinusOne>,
        fusedDecodeI4_impl<28, Delta, Patch, MinusOne>, fusedDecodeI4_impl<29, Delta, Patch, MinusOne>,
        fusedDecodeI4_impl<30, Delta, Patch, MinusOne>, fusedDecodeI4_impl<31, Delta, Patch, MinusOne>,
        fusedDecodeI4_impl<32, Delta, Patch, MinusOne>,
    };
};

} // namespace detail

// ---------------------------------------------------------------------------
// Public: fusedDecodeI4<Delta, Patch, MinusOne>(...)
// ---------------------------------------------------------------------------
// Decodes 128 interleaved values at runtime bit-width `b`.
// `carry` is the prefix-sum state (in/out, only used when Delta=true).
// `bitmap` and `residuals` are only read when Patch=true.
// MinusOne=true → delta1 (carry += val+1), MinusOne=false → delta0 (carry += val)

template <bool Delta, bool Patch, bool MinusOne = false>
inline void fusedDecodeI4(const uint8_t* in, uint32_t* out, unsigned b, uint32_t& carry, const uint64_t* bitmap,
                          const uint32_t* residuals)
{
    detail::FusedI4Table<Delta, Patch, MinusOne>::table[b](in, out, carry, bitmap, residuals);
}


} // namespace abpfor
