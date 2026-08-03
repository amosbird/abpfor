#pragma once

// abpfor scalar fallback for interleaved (I4/I8) pack/unpack/fused decode.
//
// Produces the EXACT same binary layout as the SSE/AVX2 versions:
//   N values → 32 groups × LANES lanes.
//   At bit-width B, group g's values sit at bit offset g*B within each
//   32-bit lane of consecutive "stripes" (uint32_t[LANES]).
//
// shortcut: pure scalar loops; fine for correctness on non-x86, not optimised

#include "../core/bits.h"
#include "../core/endian.h"

#include <cstring>

namespace abpfor
{

// ---------------------------------------------------------------------------
// Internal: interleaved pack/unpack for arbitrary lane count
// ---------------------------------------------------------------------------

namespace detail
{

// Unpack: read interleaved bitstream → flat output array
// Layout: stripes are uint32_t[lanes], one stripe per 32-bit word across all lanes.
// Group g's B-bit value for lane L is at bit offset (g*B)%32 in stripe[(g*B)/32][L].
template <unsigned Lanes>
inline const uint8_t* unpackInterleaved(const uint8_t* in, uint32_t* out, unsigned b, unsigned count)
{
    if (b == 0)
    {
        std::memset(out, 0, count * sizeof(uint32_t));
        return in;
    }
    if (b == 32)
    {
        detail::copyU32ArrayFromLe(out, in, count);
        return in + count * sizeof(uint32_t);
    }

    // shortcut: load all stripe words once with LE conversion; max 32 stripes × Lanes words
    const uint32_t* raw = reinterpret_cast<const uint32_t*>(in);
    unsigned totalBits = 32u * b;
    unsigned numStripes = (totalBits + 31u) / 32u;
    unsigned numWords = numStripes * Lanes;
    uint32_t stripes[32 * Lanes];
    for (unsigned i = 0; i < numWords; ++i) stripes[i] = detail::leToNative(raw[i]);

    const uint32_t vmask = (1u << b) - 1u;

    for (unsigned g = 0; g < 32; ++g)
    {
        unsigned bitOff = g * b;
        unsigned stripeIdx = bitOff / 32u;
        unsigned shift = bitOff % 32u;
        bool spans = (shift + b > 32u) && (b < 32u);

        for (unsigned lane = 0; lane < Lanes; ++lane)
        {
            uint32_t val = stripes[stripeIdx * Lanes + lane] >> shift;
            if (spans) val |= stripes[(stripeIdx + 1) * Lanes + lane] << (32u - shift);
            out[g * Lanes + lane] = val & vmask;
        }
    }

    return in + packedBytes(count, b);
}

// Pack: flat input array → interleaved bitstream
template <unsigned Lanes> inline uint8_t* packInterleaved(const uint32_t* in, uint8_t* out, unsigned b, unsigned count)
{
    if (b == 0) return out;
    if (b == 32)
    {
        detail::copyU32ArrayToLe(out, in, count);
        return out + count * sizeof(uint32_t);
    }

    unsigned totalBits = 32u * b; // 32 groups × B bits
    unsigned numStripes = (totalBits + 31u) / 32u;
    unsigned numWords = numStripes * Lanes;

    // Zero-init stripe buffer
    // shortcut: VLA-like, max 32*Lanes words (32*8=256 uint32_t = 1KB); fine on stack
    uint32_t stripeBuf[32 * Lanes]; // max numStripes is 32
    std::memset(stripeBuf, 0, numWords * sizeof(uint32_t));

    for (unsigned g = 0; g < 32; ++g)
    {
        unsigned bitOff = g * b;
        unsigned stripeIdx = bitOff / 32u;
        unsigned shift = bitOff % 32u;
        bool spans = (shift + b > 32u) && (b < 32u);

        for (unsigned lane = 0; lane < Lanes; ++lane)
        {
            uint32_t v = in[g * Lanes + lane];
            stripeBuf[stripeIdx * Lanes + lane] |= v << shift;
            if (spans) stripeBuf[(stripeIdx + 1) * Lanes + lane] |= v >> (32u - shift);
        }
    }

    // Convert stripe words to LE before writing
    for (unsigned i = 0; i < numWords; ++i) stripeBuf[i] = detail::nativeToLe(stripeBuf[i]);

    unsigned totalBytes = packedBytes(count, b);
    std::memcpy(out, stripeBuf, totalBytes);
    return out + totalBytes;
}

// Fused decode: unpack interleaved + optional patch + optional delta
template <unsigned Lanes, bool Delta, bool Patch, bool MinusOne = false>
inline void fusedDecodeInterleaved(const uint8_t* in, uint32_t* out, unsigned b, uint32_t& carry,
                                   const uint64_t* bitmap, const uint32_t* residuals, unsigned count)
{
    const uint32_t* pex = residuals;

    if (b == 0)
    {
        if constexpr (Delta)
        {
            for (unsigned g = 0; g < 32; ++g)
            {
                for (unsigned lane = 0; lane < Lanes; ++lane)
                {
                    uint32_t val = 0;
                    if constexpr (Patch)
                    {
                        unsigned idx = g * Lanes + lane;
                        if (bitmap[idx / 64] & (uint64_t(1) << (idx % 64))) val = *pex++;
                    }
                    carry += (MinusOne ? 1u : 0u) + val;
                    out[g * Lanes + lane] = carry;
                }
            }
        }
        else
        {
            std::memset(out, 0, count * sizeof(uint32_t));
            if constexpr (Patch)
            {
                for (unsigned i = 0; i < count; ++i)
                {
                    if (bitmap[i / 64] & (uint64_t(1) << (i % 64))) out[i] = *pex++;
                }
            }
        }
        return;
    }

    if (b == 32)
    {
        detail::copyU32ArrayFromLe(out, in, count);
        if constexpr (Delta)
        {
            for (unsigned i = 0; i < count; ++i)
            {
                carry += out[i] + (MinusOne ? 1u : 0u);
                out[i] = carry;
            }
        }
        return;
    }

    // General case: unpack then fuse — load stripe words with LE conversion
    const uint32_t* raw = reinterpret_cast<const uint32_t*>(in);
    unsigned totalBits = 32u * b;
    unsigned numStripes = (totalBits + 31u) / 32u;
    unsigned numWords = numStripes * Lanes;
    uint32_t stripes[32 * Lanes];
    for (unsigned i = 0; i < numWords; ++i) stripes[i] = detail::leToNative(raw[i]);

    const uint32_t vmask = (1u << b) - 1u;

    for (unsigned g = 0; g < 32; ++g)
    {
        unsigned bitOff = g * b;
        unsigned stripeIdx = bitOff / 32u;
        unsigned shift = bitOff % 32u;
        bool spans = (shift + b > 32u) && (b < 32u);

        for (unsigned lane = 0; lane < Lanes; ++lane)
        {
            uint32_t val = stripes[stripeIdx * Lanes + lane] >> shift;
            if (spans) val |= stripes[(stripeIdx + 1) * Lanes + lane] << (32u - shift);
            val &= vmask;

            if constexpr (Patch)
            {
                unsigned idx = g * Lanes + lane;
                if (bitmap[idx / 64] & (uint64_t(1) << (idx % 64))) val += (*pex++) << b;
            }

            if constexpr (Delta)
            {
                carry += val + (MinusOne ? 1u : 0u);
                out[g * Lanes + lane] = carry;
            }
            else
            {
                out[g * Lanes + lane] = val;
            }
        }
    }
}

} // namespace detail

// ---------------------------------------------------------------------------
// Interleave4 (128 elements, 4 lanes) — scalar fallback
// ---------------------------------------------------------------------------

inline const uint8_t* unpackI4(const uint8_t* in, uint32_t* out, unsigned b)
{
    return detail::unpackInterleaved<4>(in, out, b, 128);
}

inline uint8_t* packI4(const uint32_t* in, uint8_t* out, unsigned b)
{
    return detail::packInterleaved<4>(in, out, b, 128);
}

template <bool Delta, bool Patch, bool MinusOne = false>
inline void fusedDecodeI4(const uint8_t* in, uint32_t* out, unsigned b, uint32_t& carry, const uint64_t* bitmap,
                          const uint32_t* residuals)
{
    detail::fusedDecodeInterleaved<4, Delta, Patch, MinusOne>(in, out, b, carry, bitmap, residuals, 128);
}

// ---------------------------------------------------------------------------
// Interleave8 (256 elements, 8 lanes) — scalar fallback
// ---------------------------------------------------------------------------

inline const uint8_t* unpackI8(const uint8_t* in, uint32_t* out, unsigned b)
{
    return detail::unpackInterleaved<8>(in, out, b, 256);
}

inline uint8_t* packI8(const uint32_t* in, uint8_t* out, unsigned b)
{
    return detail::packInterleaved<8>(in, out, b, 256);
}

template <bool Delta, bool Patch, bool MinusOne = false>
inline void fusedDecodeI8(const uint8_t* in, uint32_t* out, unsigned b, uint32_t& carry, const uint64_t* bitmap,
                          const uint32_t* residuals)
{
    detail::fusedDecodeInterleaved<8, Delta, Patch, MinusOne>(in, out, b, carry, bitmap, residuals, 256);
}

} // namespace abpfor
