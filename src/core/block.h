#pragma once

// abpfor::encodeBlock / abpfor::decodeBlock — single-block PFor codec.
//
// Wire format:
//
//   Byte 0: [type:2][b:6]
//     type=00  bitpack only
//     type=01  bitmap outliers (byte 1 = pb)
//     type=10  sparse outliers (byte 1 = pb, byte 2 = outlierCount)
//     type=11  special: b=0 constant, b=1 raw, b=2 all-zeros

#include "bits.h"
#include "cost.h"
#include "delta.h"
#include "pack.h"

#include <algorithm>
#include <cstring>

namespace abpfor
{

// ---------------------------------------------------------------------------
// Header byte layout
// ---------------------------------------------------------------------------

namespace hdr
{
constexpr uint8_t kBitpackOnly = 0x00;   // type=00
constexpr uint8_t kBitmapOutlier = 0x40; // type=01
constexpr uint8_t kSparseOutlier = 0x80; // type=10
constexpr uint8_t kSpecial = 0xC0;       // type=11

constexpr uint8_t kTypeMask = 0xC0;
constexpr uint8_t kBitsMask = 0x3F;

// Special sub-types (in the b field when type=11)
constexpr uint8_t kAllZero = 0; // type=11 b=0: all values are zero
constexpr uint8_t kRaw = 63;    // type=11 b=63: uncompressed raw data
// type=11 b=1..62: constant block, b = bitwidth of the repeated value
//   followed by ceil(b/8) bytes of the value in little-endian
} // namespace hdr

// ---------------------------------------------------------------------------
// encodeBlock — encode n values into a compressed block
// ---------------------------------------------------------------------------
// Forward declarations (delta variants call the non-delta overloads)
template <typename T> size_t encodeBlock(const T* in, unsigned n, uint8_t* out);
template <typename T> size_t decodeBlock(const uint8_t* in, unsigned n, T* out);

// Returns bytes written.  Delta variant: pass `start` to pre-apply delta.

template <typename T, unsigned MaxN = 256> size_t encodeBlockDelta1(const T* in, unsigned n, uint8_t* out, T start)
{
    T tmp[MaxN];
    delta(in, n, tmp, start);
    return encodeBlock(tmp, n, out);
}

// shortcut: delta0 variant — same as delta1 but uses delta0 (no -1)
template <typename T, unsigned MaxN = 256> size_t encodeBlockDelta0(const T* in, unsigned n, uint8_t* out, T start)
{
    T tmp[MaxN];
    delta0(in, n, tmp, start);
    return encodeBlock(tmp, n, out);
}

template <typename T> size_t encodeBlock(const T* in, unsigned n, uint8_t* out)
{
    static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>);
    constexpr unsigned W = kMaxBits<T>;
    uint8_t* op = out;

    // Inline fast scan: detect allzero/constant without calling optimalWidth
    T orAll = 0;
    const T first = in[0];
    unsigned eqCount = 0;
    for (unsigned i = 0; i < n; ++i)
    {
        orAll |= in[i];
        eqCount += (in[i] == first);
    }
    unsigned maxBits = bitwidth(orAll);

    if (maxBits == 0)
    {
        *op++ = hdr::kSpecial | hdr::kAllZero;
        return static_cast<size_t>(op - out);
    }
    if (eqCount == n)
    {
        unsigned b = maxBits;
        if (b >= hdr::kRaw)
        {
            *op++ = hdr::kSpecial | hdr::kRaw;
            // shortcut: kRaw uses native-endian memcpy; not portable across endiannesses
            std::memcpy(op, in, n * sizeof(T));
            op += n * sizeof(T);
        }
        else
        {
            *op++ = hdr::kSpecial | static_cast<uint8_t>(b);
            if constexpr (std::is_same_v<T, uint32_t>)
                storeu<uint32_t>(op, in[0]);
            else
                storeu<uint64_t>(op, in[0]);
            op += (b + 7u) >> 3;
        }
        return static_cast<size_t>(op - out);
    }

    // General path: full cost model
    unsigned pbx;
    unsigned b = optimalWidth(in, n, &pbx, maxBits);

    // --- All zeros ---
    if (b == 0 && pbx == 0)
    {
        *op++ = hdr::kSpecial | hdr::kAllZero;
        return static_cast<size_t>(op - out);
    }

    // --- Constant ---
    if (pbx == W + 2)
    {
        if (b >= hdr::kRaw)
        {
            *op++ = hdr::kSpecial | hdr::kRaw;
            // shortcut: kRaw uses native-endian memcpy; not portable across endiannesses
            std::memcpy(op, in, n * sizeof(T));
            op += n * sizeof(T);
        }
        else
        {
            *op++ = hdr::kSpecial | static_cast<uint8_t>(b);
            // shortcut: always write 4/8 bytes (safe, buffer has room), advance by needed
            if constexpr (std::is_same_v<T, uint32_t>)
                storeu<uint32_t>(op, in[0]);
            else
                storeu<uint64_t>(op, in[0]);
            op += (b + 7u) >> 3;
        }
        return static_cast<size_t>(op - out);
    }

    // --- Bitpack only (no outliers) ---
    if (pbx == 0) [[likely]]
    {
        // shortcut: b=64 can't fit in 6-bit header field; emit raw block
        if (b >= W)
        {
            *op++ = hdr::kSpecial | hdr::kRaw;
            // shortcut: kRaw uses native-endian memcpy; not portable across endiannesses
            std::memcpy(op, in, n * sizeof(T));
            op += n * sizeof(T);
            return static_cast<size_t>(op - out);
        }
        *op++ = hdr::kBitpackOnly | static_cast<uint8_t>(b);
        op = pack(in, n, op, b);
        return static_cast<size_t>(op - out);
    }

    // --- Outlier path: split values into base and residuals in one pass ---
    // For bitmap (pbx == W+1), compute outlierBits from data scan.
    // For sparse, pbx IS the outlierBits.

    T baseMask = mask<T>(b);
    T base[256];
    T residuals[256];
    uint8_t positions[256];
    unsigned oc = 0;

    for (unsigned i = 0; i < n; ++i)
    {
        T v = in[i];
        base[i] = v & baseMask;
        if (v > baseMask)
        {
            positions[oc] = static_cast<uint8_t>(i);
            residuals[oc] = v >> b;
            ++oc;
        }
    }

    unsigned pb = (pbx == W + 1) ? (maxBits - b) : pbx;

    if (pbx == W + 1) // Bitmap
    {
        *op++ = hdr::kBitmapOutlier | static_cast<uint8_t>(b);
        *op++ = static_cast<uint8_t>(pb);

        unsigned bitmapBytes = divCeil(n, 8u);
        uint64_t bitmapWords[4] = {}; // max n=256 → 4 uint64_t
        for (unsigned i = 0; i < oc; ++i) bitmapWords[positions[i] >> 6] |= uint64_t(1) << (positions[i] & 63);
#if ABPFOR_BIG_ENDIAN
        for (unsigned i = 0; i < 4; ++i) bitmapWords[i] = detail::bswap64(bitmapWords[i]);
#endif
        std::memcpy(op, bitmapWords, bitmapBytes);
        op += bitmapBytes;

        op = pack(residuals, oc, op, pb);
        op = pack(base, n, op, b);
    }
    else // Sparse
    {
        *op++ = hdr::kSparseOutlier | static_cast<uint8_t>(b);
        *op++ = static_cast<uint8_t>(pb);
        *op++ = static_cast<uint8_t>(oc);

        std::memcpy(op, positions, oc);
        op += oc;

        op = pack(residuals, oc, op, pb);
        op = pack(base, n, op, b);
    }

    return static_cast<size_t>(op - out);
}

// ---------------------------------------------------------------------------
// decodeBlock — decode n values from a compressed block
// ---------------------------------------------------------------------------
// Returns bytes consumed.  Delta-1 variant: pass `start` to post-apply undelta.
//
// shortcut: split into fast path (decodeBlock) + slow path (decodeBlockOutliers)
// to minimize register pressure on the common bitpack-only case; the outlier
// path needs large stack arrays (residuals, positions, bitmap) that would
// pollute the fast path's prologue if kept in one function.

// --- Slow path: bitmap and sparse outlier decoding (NOINLINE to keep
//     register/stack pressure out of the fast path) ---
template <typename T> size_t decodeBlockOutliers(const uint8_t* in, unsigned n, T* out, uint8_t type, unsigned b)
{
    const uint8_t* ip = in + 1; // skip header byte already parsed by caller

    // --- Bitmap outliers ---
    if (type == hdr::kBitmapOutlier)
    {
        unsigned pb = *ip++;
        unsigned bitmapBytes = divCeil(n, 8u);
        const uint8_t* bitmap = ip;
        ip += bitmapBytes;

        // Count outliers via 64-bit popcount (faster than byte-by-byte)
        unsigned oc = 0;
        {
            const uint64_t* bmp64 = reinterpret_cast<const uint64_t*>(bitmap);
            unsigned words = bitmapBytes / 8;
            for (unsigned i = 0; i < words; ++i) oc += static_cast<unsigned>(__builtin_popcountll(bmp64[i]));
            // Handle remaining bytes
            for (unsigned i = words * 8; i < bitmapBytes; ++i)
                oc += static_cast<unsigned>(__builtin_popcount(bitmap[i]));
        }

        // Unpack outlier residuals
        T residuals[256];
        ip = unpack(ip, oc, residuals, pb);

        // Unpack base values
        ip = unpack(ip, n, out, b);

        // Merge: scan bitmap with 64-bit words for fast bit extraction
        unsigned ri = 0;
        for (unsigned wi = 0; wi < bitmapBytes / 8; ++wi)
        {
            uint64_t word;
            std::memcpy(&word, bitmap + wi * 8, 8);
#if ABPFOR_BIG_ENDIAN
            word = detail::bswap64(word);
#endif
            while (word)
            {
                unsigned bit = static_cast<unsigned>(__builtin_ctzll(word));
                unsigned pos = wi * 64 + bit;
                out[pos] |= residuals[ri++] << b;
                word &= word - 1; // clear lowest set bit
            }
        }
        // Handle remaining bytes
        for (unsigned i = (bitmapBytes / 8) * 8; i < bitmapBytes; ++i)
        {
            uint8_t byte = bitmap[i];
            while (byte)
            {
                unsigned bit = static_cast<unsigned>(__builtin_ctz(byte));
                unsigned pos = i * 8 + bit;
                out[pos] |= residuals[ri++] << b;
                byte &= byte - 1;
            }
        }

        return static_cast<size_t>(ip - in);
    }

    // --- Sparse outliers ---
    if (type == hdr::kSparseOutlier)
    {
        unsigned pb = *ip++;
        unsigned oc = *ip++;

        const uint8_t* positions = ip;
        ip += oc;

        // Save residuals start, skip past them
        const uint8_t* resPtr = ip;
        ip += packedBytes(oc, pb);

        // Unpack base values first (hot in L1 for merge)
        ip = unpack(ip, n, out, b);

        // Unpack residuals via dispatch table, then merge at positions
        if (pb > 0 && oc > 0)
        {
            // shortcut: fused extract — loadU64 per element, avoids dispatch table
            // overhead for small oc. Safe: base data follows residuals in buffer.
            T pmask = (pb >= sizeof(T) * 8) ? ~T(0) : ((T(1) << pb) - 1);
            for (unsigned i = 0; i < oc; ++i)
            {
                unsigned bitpos = i * pb;
                uint64_t w = detail::bitops::loadU64Fast(resPtr + (bitpos >> 3));
                T residual = static_cast<T>(w >> (bitpos & 7)) & pmask;
                out[positions[i]] |= residual << b;
            }
        }

        return static_cast<size_t>(ip - in);
    }

    return 0; // unreachable
}

// --- Fast path entry point: parse header, handle common cases inline,
//     tail-call to decodeBlockOutliers for outlier paths ---

template <typename T> size_t decodeBlockDelta1(const uint8_t* in, unsigned n, T* out, T start)
{
    // Peek at header to handle all-zeros and constant cases without
    // the separate undelta pass (these are hot paths for sorted data).
    uint8_t h = in[0];
    uint8_t type = h & hdr::kTypeMask;
    unsigned b = h & hdr::kBitsMask;

    // Bitpack-only first: it is by far the most common block type, and it is
    // the only one whose work is large enough for a mispredicted branch here to
    // matter. Testing the two rare kSpecial cases ahead of it put two
    // dependent compare/branch pairs on the critical path before the kernel
    // could start. Measured at n=128: b=4 -13.8% -> -4%, b=16 -12.1% -> -3%
    // against TurboPFor, which orders its dispatch the same way.
    if (type == hdr::kBitpackOnly) [[likely]]
    {
        unpack_delta(in + 1, n, out, b, start);
        return 1 + packedBytes(n, b);
    }

    if (type == hdr::kSpecial && b == hdr::kAllZero)
    {
        // All deltas are zero → output is start+1, start+2, ...
        for (unsigned i = 0; i < n; ++i) out[i] = start + T(i + 1);
        return 1;
    }

    // Constant: b = bitwidth, followed by ceil(b/8) value bytes.
    // kRaw (b=63) must be excluded: it stores n*sizeof(T) raw bytes, not a
    // constant, so it falls through to decodeBlock below (memcpy + undelta).
    if (type == hdr::kSpecial && b != hdr::kAllZero && b != hdr::kRaw)
    {
        T val = loadu<T>(reinterpret_cast<const uint8_t*>(in + 1)) & mask<T>(b); // reads LE
        unsigned valBytes = (b + 7u) >> 3;
        T inc = val + 1;
        for (unsigned i = 0; i < n; ++i) out[i] = start + inc * T(i + 1);
        return 1 + valBytes;
    }

    // Outlier path: decode base+exceptions, then apply delta.
    // Also handles kRaw blocks via decodeBlock's memcpy path.
    size_t consumed = decodeBlock(in, n, out);
    for (unsigned i = 0; i < n; ++i) out[i] = (start += out[i]) + T(i + 1u);
    return consumed;
}

// shortcut: delta0 variant — unpack then undelta0 (no fused path, no +1)
template <typename T> size_t decodeBlockDelta0(const uint8_t* in, unsigned n, T* out, T& carry)
{
    size_t consumed = decodeBlock(in, n, out);
    undelta0(out, n, carry);
    return consumed;
}

template <typename T> size_t decodeBlock(const uint8_t* in, unsigned n, T* out)
{
    static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>);

    const uint8_t* ip = in;
    uint8_t h = *ip++;
    uint8_t type = h & hdr::kTypeMask;
    unsigned b = h & hdr::kBitsMask;

    // --- Bitpack only (most common path, check first) ---
    if (type == hdr::kBitpackOnly) [[likely]]
    {
        ip = unpack(ip, n, out, b);
        return static_cast<size_t>(ip - in);
    }

    // --- Special types (small, no extra registers needed) ---
    if (type == hdr::kSpecial)
    {
        if (b == hdr::kAllZero)
        {
            std::memset(out, 0, n * sizeof(T));
            return static_cast<size_t>(ip - in);
        }

        if (b == hdr::kRaw)
        {
            // shortcut: kRaw uses native-endian memcpy; round-trips correctly on same arch
            // but wire format is not portable across endiannesses
            std::memcpy(out, ip, n * sizeof(T));
            ip += n * sizeof(T);
            return static_cast<size_t>(ip - in);
        }

        // Constant: b = bitwidth of repeated value, ceil(b/8) value bytes follow
        {
            // shortcut: always load full T (safe, buffer has padding), advance by needed; reads LE via loadu
            T val;
            if constexpr (std::is_same_v<T, uint32_t>)
                val = loadu<uint32_t>(ip) & mask<T>(b);
            else
                val = loadu<uint64_t>(ip) & mask<T>(b);
            ip += (b + 7u) >> 3;
            std::fill_n(out, n, val);
            return static_cast<size_t>(ip - in);
        }
    }

    // --- Outlier paths: tail-call to NOINLINE slow path ---
    return decodeBlockOutliers(in, n, out, type, b);
}

} // namespace abpfor
