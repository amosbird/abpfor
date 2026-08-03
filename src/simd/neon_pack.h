#pragma once

// abpfor NEON interleaved (Interleave4) pack/unpack — 128 elements, 4-lane.
// Direct port of sse_pack.h using uint32x4_t instead of __m128i.

#include "../core/bits.h"

#include <arm_neon.h>
#include <cstring>

namespace abpfor
{

// ---------------------------------------------------------------------------
// Compile-time mask for NEON lanes
// ---------------------------------------------------------------------------

template <unsigned B> ABPFOR_INLINE uint32x4_t neonMask()
{
    if constexpr (B == 32)
        return vdupq_n_u32(0xFFFFFFFFu);
    else
        return vdupq_n_u32((1u << B) - 1u);
}

// ---------------------------------------------------------------------------
// NeonUnpackGroup — extract group G's 4 values from the stripe array.
// ---------------------------------------------------------------------------

template <unsigned B, unsigned G, unsigned MaxG, int LoadedIdx> struct NeonUnpackGroup
{
    static ABPFOR_INLINE void run(const uint32x4_t*& ip, uint32x4_t& iv, uint32_t* out, const uint32x4_t& vmask)
    {
        constexpr unsigned bitOffset = G * B;
        constexpr int stripeIdx = static_cast<int>(bitOffset / 32u);
        constexpr unsigned shift = bitOffset % 32u;
        constexpr bool spans = (shift + B > 32u) && (B < 32u);

        if constexpr (stripeIdx > LoadedIdx) iv = vld1q_u32(reinterpret_cast<const uint32_t*>(ip + stripeIdx));

        uint32x4_t ov;
        if constexpr (shift == 0)
            ov = iv;
        else
            ov = vshrq_n_u32(iv, shift); // shortcut: shift is always 1-31 here since shift!=0 and B<32 implies shift<32

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

        vst1q_u32(out + G * 4, ov);

        constexpr int nextLoaded = spans ? stripeIdx + 1 : stripeIdx;
        NeonUnpackGroup<B, G + 1, MaxG, nextLoaded>::run(ip, iv, out, vmask);
    }
};

template <unsigned B, unsigned MaxG, int LoadedIdx> struct NeonUnpackGroup<B, MaxG, MaxG, LoadedIdx>
{
    static ABPFOR_INLINE void run(const uint32x4_t*&, uint32x4_t&, uint32_t*, const uint32x4_t&) {}
};

// ---------------------------------------------------------------------------
// unpackI4_impl<B>
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
        const uint32x4_t* ip = reinterpret_cast<const uint32x4_t*>(in);
        uint32x4_t iv = vdupq_n_u32(0);
        const uint32x4_t vmask = neonMask<B>();

        NeonUnpackGroup<B, 0, 32, -1>::run(ip, iv, out, vmask);

        return in + packedBytes(128, B);
    }
}

// ---------------------------------------------------------------------------
// NeonPackGroup — pack group G's 4 values into the stripe array.
// ---------------------------------------------------------------------------

template <unsigned B, unsigned G, unsigned MaxG> struct NeonPackGroup
{
    static ABPFOR_INLINE void run(const uint32_t* in, uint32x4_t* stripes)
    {
        constexpr unsigned bitOffset = G * B;
        constexpr unsigned stripeIdx = bitOffset / 32u;
        constexpr unsigned shift = bitOffset % 32u;
        constexpr bool spans = (shift + B > 32u) && (B < 32u);

        uint32x4_t v = vld1q_u32(in + G * 4);

        if constexpr (shift == 0)
            stripes[stripeIdx] = vorrq_u32(stripes[stripeIdx], v);
        else
            stripes[stripeIdx] = vorrq_u32(stripes[stripeIdx], vshlq_n_u32(v, shift));

        if constexpr (spans)
        {
            constexpr unsigned lo = 32u - shift;
            stripes[stripeIdx + 1] = vshrq_n_u32(v, lo);
        }

        NeonPackGroup<B, G + 1, MaxG>::run(in, stripes);
    }
};

template <unsigned B, unsigned MaxG> struct NeonPackGroup<B, MaxG, MaxG>
{
    static ABPFOR_INLINE void run(const uint32_t*, uint32x4_t*) {}
};

// ---------------------------------------------------------------------------
// packI4_impl<B>
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
        constexpr unsigned totalBits = 32u * B;
        constexpr unsigned numStripes = (totalBits + 31u) / 32u;

        uint32x4_t stripes[numStripes];
        for (unsigned i = 0; i < numStripes; ++i) stripes[i] = vdupq_n_u32(0);

        NeonPackGroup<B, 0, 32>::run(in, stripes);

        unsigned totalBytes = packedBytes(128, B);
        std::memcpy(out, stripes, totalBytes);
        return out + totalBytes;
    }
}

// ---------------------------------------------------------------------------
// Runtime dispatch
// ---------------------------------------------------------------------------

using UnpackI4Fn = const uint8_t* (*)(const uint8_t*, uint32_t*);
using PackI4Fn = uint8_t* (*)(const uint32_t*, uint8_t*);

namespace detail
{

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
// Public API
// ---------------------------------------------------------------------------

inline const uint8_t* unpackI4(const uint8_t* in, uint32_t* out, unsigned b)
{
    return detail::kUnpackI4Table[b](in, out);
}

inline uint8_t* packI4(const uint32_t* in, uint8_t* out, unsigned b)
{
    return detail::kPackI4Table[b](in, out);
}

// ===========================================================================
// Interleave8 — 256 elements, 8-lane (two uint32x4_t per stripe)
// shortcut: processes each 8-lane stripe as lo+hi halves; same bit-layout
// as AVX2 but with 128-bit NEON ops. Fine for ARM64; no wider alternative.
// ===========================================================================

// ---------------------------------------------------------------------------
// NeonUnpackGroup8 — extract group G's 8 values (lo 4 + hi 4).
// ---------------------------------------------------------------------------

template <unsigned B, unsigned G, unsigned MaxG, int LoadedIdx> struct NeonUnpackGroup8
{
    static ABPFOR_INLINE void run(const uint32_t*& ip, uint32x4_t& ivLo, uint32x4_t& ivHi, uint32_t* out,
                                  const uint32x4_t& vmask)
    {
        constexpr unsigned bitOffset = G * B;
        constexpr int stripeIdx = static_cast<int>(bitOffset / 32u);
        constexpr unsigned shift = bitOffset % 32u;
        constexpr bool spans = (shift + B > 32u) && (B < 32u);

        if constexpr (stripeIdx > LoadedIdx)
        {
            // Each stripe is 8 uint32_t wide (lo 4 + hi 4)
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

        vst1q_u32(out + G * 8, ovLo);
        vst1q_u32(out + G * 8 + 4, ovHi);

        constexpr int nextLoaded = spans ? stripeIdx + 1 : stripeIdx;
        NeonUnpackGroup8<B, G + 1, MaxG, nextLoaded>::run(ip, ivLo, ivHi, out, vmask);
    }
};

template <unsigned B, unsigned MaxG, int LoadedIdx> struct NeonUnpackGroup8<B, MaxG, MaxG, LoadedIdx>
{
    static ABPFOR_INLINE void run(const uint32_t*&, uint32x4_t&, uint32x4_t&, uint32_t*, const uint32x4_t&) {}
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
        const uint32_t* ip = reinterpret_cast<const uint32_t*>(in);
        uint32x4_t ivLo = vdupq_n_u32(0);
        uint32x4_t ivHi = vdupq_n_u32(0);
        const uint32x4_t vmask = neonMask<B>();

        NeonUnpackGroup8<B, 0, 32, -1>::run(ip, ivLo, ivHi, out, vmask);

        return in + packedBytes(256, B);
    }
}

// ---------------------------------------------------------------------------
// NeonPackGroup8 — pack group G's 8 values into the stripe array.
// ---------------------------------------------------------------------------

template <unsigned B, unsigned G, unsigned MaxG> struct NeonPackGroup8
{
    static ABPFOR_INLINE void run(const uint32_t* in, uint32x4_t* stripesLo, uint32x4_t* stripesHi)
    {
        constexpr unsigned bitOffset = G * B;
        constexpr unsigned stripeIdx = bitOffset / 32u;
        constexpr unsigned shift = bitOffset % 32u;
        constexpr bool spans = (shift + B > 32u) && (B < 32u);

        uint32x4_t vLo = vld1q_u32(in + G * 8);
        uint32x4_t vHi = vld1q_u32(in + G * 8 + 4);

        if constexpr (shift == 0)
        {
            stripesLo[stripeIdx] = vorrq_u32(stripesLo[stripeIdx], vLo);
            stripesHi[stripeIdx] = vorrq_u32(stripesHi[stripeIdx], vHi);
        }
        else
        {
            stripesLo[stripeIdx] = vorrq_u32(stripesLo[stripeIdx], vshlq_n_u32(vLo, shift));
            stripesHi[stripeIdx] = vorrq_u32(stripesHi[stripeIdx], vshlq_n_u32(vHi, shift));
        }

        if constexpr (spans)
        {
            constexpr unsigned lo = 32u - shift;
            stripesLo[stripeIdx + 1] = vshrq_n_u32(vLo, lo);
            stripesHi[stripeIdx + 1] = vshrq_n_u32(vHi, lo);
        }

        NeonPackGroup8<B, G + 1, MaxG>::run(in, stripesLo, stripesHi);
    }
};

template <unsigned B, unsigned MaxG> struct NeonPackGroup8<B, MaxG, MaxG>
{
    static ABPFOR_INLINE void run(const uint32_t*, uint32x4_t*, uint32x4_t*) {}
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

        uint32x4_t stripesLo[numStripes];
        uint32x4_t stripesHi[numStripes];
        for (unsigned i = 0; i < numStripes; ++i)
        {
            stripesLo[i] = vdupq_n_u32(0);
            stripesHi[i] = vdupq_n_u32(0);
        }

        NeonPackGroup8<B, 0, 32>::run(in, stripesLo, stripesHi);

        // Interleave lo/hi into output: each stripe is [lo4, hi4] = 32 bytes
        unsigned totalBytes = packedBytes(256, B);
        uint8_t* op = out;
        for (unsigned i = 0; i < numStripes; ++i)
        {
            unsigned remain = totalBytes - i * 32;
            if (remain >= 32)
            {
                vst1q_u32(reinterpret_cast<uint32_t*>(op), stripesLo[i]);
                vst1q_u32(reinterpret_cast<uint32_t*>(op + 16), stripesHi[i]);
                op += 32;
            }
            else
            {
                // Last partial stripe
                if (remain >= 16)
                {
                    vst1q_u32(reinterpret_cast<uint32_t*>(op), stripesLo[i]);
                    std::memcpy(op + 16, &stripesHi[i], remain - 16);
                    op += remain;
                }
                else
                {
                    std::memcpy(op, &stripesLo[i], remain);
                    op += remain;
                }
            }
        }
        return out + totalBytes;
    }
}

// ---------------------------------------------------------------------------
// Runtime dispatch — I8
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

inline const uint8_t* unpackI8(const uint8_t* in, uint32_t* out, unsigned b)
{
    return detail::kUnpackI8Table[b](in, out);
}

inline uint8_t* packI8(const uint32_t* in, uint8_t* out, unsigned b)
{
    return detail::kPackI8Table[b](in, out);
}

} // namespace abpfor
