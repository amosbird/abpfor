#pragma once

// abpfor NEON fused decode — single-pass unpack + patch + prefixSum.
// Direct port of sse_fused.h using uint32x4_t instead of __m128i.

#include "../core/bits.h"
#include "neon_pack.h"

#include <arm_neon.h>

namespace abpfor
{

// ---------------------------------------------------------------------------
// NEON scatter helper — replaces SSE pshufb LUT approach.
// shortcut: scalar scatter per group; fine for ≤4 lanes. A vtbl-based
// approach would save ~1 cycle per group but adds complexity for 4 elements.
// ---------------------------------------------------------------------------

namespace detail
{

ABPFOR_INLINE uint32x4_t neonScatterPatch(unsigned nibble, const uint32_t*& pex, unsigned B)
{
    alignas(16) uint32_t tmp[4] = {0, 0, 0, 0};
    for (unsigned lane = 0; lane < 4; ++lane)
    {
        if (nibble & (1u << lane)) tmp[lane] = *pex++ << B;
    }
    return vld1q_u32(tmp);
}

} // namespace detail

// ---------------------------------------------------------------------------
// NeonFusedGroup
// ---------------------------------------------------------------------------

template <unsigned B, unsigned G, unsigned MaxG, int LoadedIdx, bool Delta, bool Patch, bool MinusOne = true> struct NeonFusedGroup
{
    static ABPFOR_INLINE void run(const uint32x4_t*& ip, uint32x4_t& iv, uint32_t* out, const uint32x4_t& vmask,
                                  uint32x4_t& sv, const uint32x4_t& cv, const uint64_t* bitmap, const uint32_t*& pex)
    {
        constexpr unsigned bitOffset = G * B;
        constexpr int stripeIdx = static_cast<int>(bitOffset / 32u);
        constexpr unsigned shift = bitOffset % 32u;
        constexpr bool spans = (shift + B > 32u) && (B < 32u);

        // --- Step 1: Extract (identical to NeonUnpackGroup) ---

        if constexpr (stripeIdx > LoadedIdx) iv = vld1q_u32(reinterpret_cast<const uint32_t*>(ip + stripeIdx));

        uint32x4_t ov;
        if constexpr (shift == 0)
            ov = iv;
        else
            ov = vshrq_n_u32(iv, shift);

        if constexpr (spans)
        {
            uint32x4_t next = vld1q_u32(reinterpret_cast<const uint32_t*>(ip + stripeIdx + 1));
            iv = next;
            constexpr unsigned lo = 32u - shift;
            ov = vorrq_u32(ov, vshlq_n_u32(next, lo));
            ov = vandq_u32(ov, vmask);
        }
        else if constexpr (B < 32u)
        {
            ov = vandq_u32(ov, vmask);
        }

        // --- Step 2: Patch outliers ---

        if constexpr (Patch)
        {
            const uint64_t bmpWord = (G < 16) ? bitmap[0] : bitmap[1];
            const unsigned nibble = static_cast<unsigned>((bmpWord >> ((G % 16) * 4)) & 0xFu);

            if (nibble != 0)
            {
                uint32x4_t scattered = detail::neonScatterPatch(nibble, pex, B);
                ov = vaddq_u32(ov, scattered);
            }
        }

        // --- Step 3: Prefix sum (delta-1 decode) ---

        if constexpr (Delta)
        {
            // In-register inclusive prefix sum of 4 lanes:
            //   ov = [a, b, c, d]
            //   → [a, a+b, a+b+c, a+b+c+d]
            // NEON: use vextq_u32 to shift lanes (equivalent to _mm_slli_si128 by 4/8 bytes)
            ov = vaddq_u32(ov, vextq_u32(vdupq_n_u32(0), ov, 3)); // shift left by 1 lane
            ov = vaddq_u32(ov, vextq_u32(vdupq_n_u32(0), ov, 2)); // shift left by 2 lanes

            // Add carry + per-lane offset
            ov = vaddq_u32(ov, vaddq_u32(sv, cv));

            vst1q_u32(out + G * 4, ov);

            // Update carry: broadcast last element
            sv = vdupq_lane_u32(vget_high_u32(ov), 1);
        }
        else
        {
            vst1q_u32(out + G * 4, ov);
        }

        // --- Recurse ---
        constexpr int nextLoaded = spans ? stripeIdx + 1 : stripeIdx;
        NeonFusedGroup<B, G + 1, MaxG, nextLoaded, Delta, Patch, MinusOne>::run(ip, iv, out, vmask, sv, cv, bitmap, pex);
    }
};

// Base case
template <unsigned B, unsigned MaxG, int LoadedIdx, bool Delta, bool Patch, bool MinusOne>
struct NeonFusedGroup<B, MaxG, MaxG, LoadedIdx, Delta, Patch, MinusOne>
{
    static ABPFOR_INLINE void run(const uint32x4_t*&, uint32x4_t&, uint32_t*, const uint32x4_t&, uint32x4_t&,
                                  const uint32x4_t&, const uint64_t*, const uint32_t*&)
    {
    }
};

// ---------------------------------------------------------------------------
// fusedDecodeI4_impl<B, Delta, Patch>
// ---------------------------------------------------------------------------

template <unsigned B, bool Delta, bool Patch, bool MinusOne = true>
ABPFOR_INLINE void fusedDecodeI4_impl(const uint8_t* in, uint32_t* out, uint32_t& carry, const uint64_t* bitmap,
                                      const uint32_t* residuals)
{
    if constexpr (B == 0)
    {
        if constexpr (Delta)
        {
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
        if constexpr (Delta)
        {
            uint32x4_t sv = vdupq_n_u32(carry);
            alignas(16) const uint32_t cv_data[4] = {1, 2, 3, 4};
            const uint32x4_t cv = MinusOne ? vld1q_u32(cv_data) : vdupq_n_u32(0);
            for (unsigned g = 0; g < 32; ++g)
            {
                uint32x4_t ov = vld1q_u32(out + g * 4);
                ov = vaddq_u32(ov, vextq_u32(vdupq_n_u32(0), ov, 3));
                ov = vaddq_u32(ov, vextq_u32(vdupq_n_u32(0), ov, 2));
                ov = vaddq_u32(ov, vaddq_u32(sv, cv));
                vst1q_u32(out + g * 4, ov);
                sv = vdupq_lane_u32(vget_high_u32(ov), 1);
            }
            carry = vgetq_lane_u32(sv, 0);
        }
        return;
    }

    // General case
    const uint32x4_t* ip = reinterpret_cast<const uint32x4_t*>(in);
    uint32x4_t iv = vdupq_n_u32(0);
    const uint32x4_t vmask = neonMask<B>();

    uint32x4_t sv = vdupq_n_u32(carry);
    alignas(16) const uint32_t cv_data[4] = {1, 2, 3, 4};
    const uint32x4_t cv = (Delta && MinusOne) ? vld1q_u32(cv_data) : vdupq_n_u32(0);

    const uint32_t* pex = residuals;

    NeonFusedGroup<B, 0, 32, -1, Delta, Patch, MinusOne>::run(ip, iv, out, vmask, sv, cv, bitmap, pex);

    if constexpr (Delta) carry = vgetq_lane_u32(sv, 0);
}

// ---------------------------------------------------------------------------
// Runtime dispatch
// ---------------------------------------------------------------------------

template <bool Delta, bool Patch, bool MinusOne>
using FusedI4Fn = void (*)(const uint8_t*, uint32_t*, uint32_t&, const uint64_t*, const uint32_t*);

namespace detail
{

template <bool Delta, bool Patch, bool MinusOne = true> struct FusedI4Table
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

template <bool Delta, bool Patch, bool MinusOne = true>
inline void fusedDecodeI4(const uint8_t* in, uint32_t* out, unsigned b, uint32_t& carry, const uint64_t* bitmap,
                          const uint32_t* residuals)
{
    detail::FusedI4Table<Delta, Patch, MinusOne>::table[b](in, out, carry, bitmap, residuals);
}

// ===========================================================================
// Interleave8 fused decode — 256 elements, 8-lane (two uint32x4_t halves)
// shortcut: processes each 8-lane stripe as lo+hi halves; same bit-layout
// as AVX2 but with 128-bit NEON ops. Fine for ARM64; no wider alternative.
// ===========================================================================

namespace detail
{

// shortcut: scalar scatter for 4-lane half, reuses same approach as I4
ABPFOR_INLINE uint32x4_t neonScatterPatch8(unsigned nibble, const uint32_t*& pex, unsigned B)
{
    alignas(16) uint32_t tmp[4] = {0, 0, 0, 0};
    for (unsigned lane = 0; lane < 4; ++lane)
    {
        if (nibble & (1u << lane)) tmp[lane] = *pex++ << B;
    }
    return vld1q_u32(tmp);
}

} // namespace detail

// ---------------------------------------------------------------------------
// NeonFusedGroup8
// ---------------------------------------------------------------------------

template <unsigned B, unsigned G, unsigned MaxG, int LoadedIdx, bool Delta, bool Patch, bool MinusOne = true> struct NeonFusedGroup8
{
    static ABPFOR_INLINE void run(const uint32_t*& ip, uint32x4_t& ivLo, uint32x4_t& ivHi, uint32_t* out,
                                  const uint32x4_t& vmask, uint32x4_t& sv, const uint32x4_t& cv4,
                                  const uint64_t* bitmap, const uint32_t*& pex)
    {
        constexpr unsigned bitOffset = G * B;
        constexpr int stripeIdx = static_cast<int>(bitOffset / 32u);
        constexpr unsigned shift = bitOffset % 32u;
        constexpr bool spans = (shift + B > 32u) && (B < 32u);

        // --- Step 1: Extract (same as NeonUnpackGroup8) ---

        if constexpr (stripeIdx > LoadedIdx)
        {
            ivLo = vld1q_u32(ip + stripeIdx * 8);
            ivHi = vld1q_u32(ip + stripeIdx * 8 + 4);
        }

        uint32x4_t ovLo, ovHi;
        if constexpr (shift == 0)
        {
            ovLo = ivLo;
            ovHi = ivHi;
        }
        else
        {
            ovLo = vshrq_n_u32(ivLo, shift);
            ovHi = vshrq_n_u32(ivHi, shift);
        }

        if constexpr (spans)
        {
            uint32x4_t nextLo = vld1q_u32(ip + (stripeIdx + 1) * 8);
            uint32x4_t nextHi = vld1q_u32(ip + (stripeIdx + 1) * 8 + 4);
            ivLo = nextLo;
            ivHi = nextHi;
            constexpr unsigned lo = 32u - shift;
            ovLo = vandq_u32(vorrq_u32(ovLo, vshlq_n_u32(nextLo, lo)), vmask);
            ovHi = vandq_u32(vorrq_u32(ovHi, vshlq_n_u32(nextHi, lo)), vmask);
        }
        else if constexpr (B < 32u)
        {
            ovLo = vandq_u32(ovLo, vmask);
            ovHi = vandq_u32(ovHi, vmask);
        }

        // --- Step 2: Patch outliers ---
        // 8 bits from bitmap per group (256 elements -> 4 uint64_t words)

        if constexpr (Patch)
        {
            const uint64_t bmpWord = bitmap[G / 8];
            const unsigned mask8 = static_cast<unsigned>((bmpWord >> ((G % 8) * 8)) & 0xFFu);
            const unsigned loNib = mask8 & 0xFu;
            const unsigned hiNib = (mask8 >> 4) & 0xFu;

            if (loNib != 0) ovLo = vaddq_u32(ovLo, detail::neonScatterPatch8(loNib, pex, B));
            if (hiNib != 0) ovHi = vaddq_u32(ovHi, detail::neonScatterPatch8(hiNib, pex, B));
        }

        // --- Step 3: Prefix sum (delta-1 decode) for 8 lanes ---

        if constexpr (Delta)
        {
            // Prefix-sum within lo (4 lanes)
            ovLo = vaddq_u32(ovLo, vextq_u32(vdupq_n_u32(0), ovLo, 3));
            ovLo = vaddq_u32(ovLo, vextq_u32(vdupq_n_u32(0), ovLo, 2));
            ovLo = vaddq_u32(ovLo, vaddq_u32(sv, cv4));

            // Cross-lane: broadcast lo's lane 3 -> add to all of hi before hi's prefix-sum
            uint32x4_t loCarry = vdupq_lane_u32(vget_high_u32(ovLo), 1);

            ovHi = vaddq_u32(ovHi, vextq_u32(vdupq_n_u32(0), ovHi, 3));
            ovHi = vaddq_u32(ovHi, vextq_u32(vdupq_n_u32(0), ovHi, 2));
            ovHi = vaddq_u32(ovHi, vaddq_u32(loCarry, cv4));

            vst1q_u32(out + G * 8, ovLo);
            vst1q_u32(out + G * 8 + 4, ovHi);

            // Update carry: broadcast hi's lane 3
            sv = vdupq_lane_u32(vget_high_u32(ovHi), 1);
        }
        else
        {
            vst1q_u32(out + G * 8, ovLo);
            vst1q_u32(out + G * 8 + 4, ovHi);
        }

        // --- Recurse ---
        constexpr int nextLoaded = spans ? stripeIdx + 1 : stripeIdx;
        NeonFusedGroup8<B, G + 1, MaxG, nextLoaded, Delta, Patch, MinusOne>::run(ip, ivLo, ivHi, out, vmask, sv, cv4, bitmap,
                                                                       pex);
    }
};

template <unsigned B, unsigned MaxG, int LoadedIdx, bool Delta, bool Patch, bool MinusOne>
struct NeonFusedGroup8<B, MaxG, MaxG, LoadedIdx, Delta, Patch, MinusOne>
{
    static ABPFOR_INLINE void run(const uint32_t*&, uint32x4_t&, uint32x4_t&, uint32_t*, const uint32x4_t&, uint32x4_t&,
                                  const uint32x4_t&, const uint64_t*, const uint32_t*&)
    {
    }
};

// ---------------------------------------------------------------------------
// fusedDecodeI8_impl<B, Delta, Patch>
// ---------------------------------------------------------------------------

template <unsigned B, bool Delta, bool Patch, bool MinusOne = true>
ABPFOR_INLINE void fusedDecodeI8_impl(const uint8_t* in, uint32_t* out, uint32_t& carry, const uint64_t* bitmap,
                                      const uint32_t* residuals)
{
    if constexpr (B == 0)
    {
        // shortcut: scalar fallback for b=0 (rare case)
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
            uint32x4_t sv = vdupq_n_u32(carry);
            alignas(16) const uint32_t cv_data[4] = {1, 2, 3, 4};
            const uint32x4_t cv4 = MinusOne ? vld1q_u32(cv_data) : vdupq_n_u32(0);
            for (unsigned g = 0; g < 32; ++g)
            {
                uint32x4_t ovLo = vld1q_u32(out + g * 8);
                uint32x4_t ovHi = vld1q_u32(out + g * 8 + 4);

                ovLo = vaddq_u32(ovLo, vextq_u32(vdupq_n_u32(0), ovLo, 3));
                ovLo = vaddq_u32(ovLo, vextq_u32(vdupq_n_u32(0), ovLo, 2));
                ovLo = vaddq_u32(ovLo, vaddq_u32(sv, cv4));

                uint32x4_t loCarry = vdupq_lane_u32(vget_high_u32(ovLo), 1);

                ovHi = vaddq_u32(ovHi, vextq_u32(vdupq_n_u32(0), ovHi, 3));
                ovHi = vaddq_u32(ovHi, vextq_u32(vdupq_n_u32(0), ovHi, 2));
                ovHi = vaddq_u32(ovHi, vaddq_u32(loCarry, cv4));

                vst1q_u32(out + g * 8, ovLo);
                vst1q_u32(out + g * 8 + 4, ovHi);

                sv = vdupq_lane_u32(vget_high_u32(ovHi), 1);
            }
            carry = vgetq_lane_u32(sv, 0);
        }
        return;
    }

    // General case
    const uint32_t* ip = reinterpret_cast<const uint32_t*>(in);
    uint32x4_t ivLo = vdupq_n_u32(0);
    uint32x4_t ivHi = vdupq_n_u32(0);
    const uint32x4_t vmask = neonMask<B>();

    uint32x4_t sv = vdupq_n_u32(carry);
    alignas(16) const uint32_t cv_data[4] = {1, 2, 3, 4};
    const uint32x4_t cv4 = (Delta && MinusOne) ? vld1q_u32(cv_data) : vdupq_n_u32(0);

    const uint32_t* pex = residuals;

    NeonFusedGroup8<B, 0, 32, -1, Delta, Patch, MinusOne>::run(ip, ivLo, ivHi, out, vmask, sv, cv4, bitmap, pex);

    if constexpr (Delta) carry = vgetq_lane_u32(sv, 0);
}

// ---------------------------------------------------------------------------
// Runtime dispatch - I8
// ---------------------------------------------------------------------------

template <bool Delta, bool Patch, bool MinusOne>
using FusedI8Fn = void (*)(const uint8_t*, uint32_t*, uint32_t&, const uint64_t*, const uint32_t*);

namespace detail
{

template <bool Delta, bool Patch, bool MinusOne = true> struct FusedI8Table
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

template <bool Delta, bool Patch, bool MinusOne = true>
inline void fusedDecodeI8(const uint8_t* in, uint32_t* out, unsigned b, uint32_t& carry, const uint64_t* bitmap,
                          const uint32_t* residuals)
{
    detail::FusedI8Table<Delta, Patch, MinusOne>::table[b](in, out, carry, bitmap, residuals);
}

} // namespace abpfor
