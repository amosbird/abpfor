#pragma once

// Internal: interleaved block encode/decode templates.
// These are templates (instantiated for uint32_t/uint64_t) so they live in a header,
// but this header is NOT part of the public API.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <algorithm>
#include <type_traits>

#include "core/arch.h"
#include "core/block.h"
#include "core/delta.h"

// The scalar interleaved path is correct but ~3-4x slower on decode. Selecting
// it by accident (e.g. because core/arch.h was not included, or because -mavx2
// was left off an x86 build) is a performance cliff with no compile error, so
// it must be opted into explicitly.
#if !ABPFOR_ARCH_X86 && !ABPFOR_ARCH_ARM64 && !defined(ABPFOR_ALLOW_SCALAR_FALLBACK)
#error "abpfor: no SIMD architecture detected. Define ABPFOR_ALLOW_SCALAR_FALLBACK to accept the slow scalar codec."
#endif

#if ABPFOR_ARCH_X86
#include "simd/sse_pack.h"
#include "simd/sse_fused.h"
#include "simd/avx2_pack.h"
#include "simd/avx2_fused.h"
#elif ABPFOR_ARCH_ARM64
#include "simd/neon_pack.h"
#include "simd/neon_fused.h"
#else
#include "simd/scalar_interleaved.h"
#endif

namespace abpfor::detail
{

// shortcut: helper to pack interleaved data — narrows uint64_t→uint32_t for b<=32,
// falls back to scalar horizontal pack<uint64_t> for b>32
template <typename T, unsigned BlockSize>
uint8_t* packInterleaved(const T* in, uint8_t* out, unsigned b, uint8_t* (*packI)(const uint32_t*, uint8_t*, unsigned))
{
    if constexpr (std::is_same_v<T, uint32_t>)
    {
        return packI(in, out, b);
    }
    else
    {
        if (b <= 32)
        {
            alignas(32) uint32_t tmp[BlockSize];
            for (unsigned i = 0; i < BlockSize; ++i) tmp[i] = static_cast<uint32_t>(in[i]);
            return packI(tmp, out, b);
        }
        else
        {
            return pack(in, BlockSize, out, b);
        }
    }
}

// shortcut: helper to unpack interleaved data — widens uint32_t→uint64_t for b<=32,
// falls back to scalar horizontal unpack<uint64_t> for b>32
template <typename T, unsigned BlockSize>
const uint8_t* unpackInterleaved(const uint8_t* in, T* out, unsigned b,
                                 const uint8_t* (*unpackI)(const uint8_t*, uint32_t*, unsigned))
{
    if constexpr (std::is_same_v<T, uint32_t>)
    {
        return unpackI(in, out, b);
    }
    else
    {
        if (b <= 32)
        {
            alignas(32) uint32_t tmp[BlockSize];
            const uint8_t* ret = unpackI(in, tmp, b);
            for (unsigned i = 0; i < BlockSize; ++i) out[i] = tmp[i];
            return ret;
        }
        else
        {
            return unpack(in, BlockSize, out, b);
        }
    }
}

// Collect the outliers (values that do not fit in b bits) into positions[] and
// residuals[], returning the count and OR-ing the outlier values into orOutliers
// so the caller can derive maxBits with a single bitwidth().
//
// `bitwidth(v) > b` is exactly `v > baseMask` (baseMask has the low b bits set):
// a compare against a loop-invariant limit, no lzcnt per element.
//
// The scalar form of this loop carries a data-dependent branch and a loop-carried
// `oc`, so it runs at ~1 element/cycle and mispredicts badly at moderate outlier
// rates. The SIMD form splits it: one branch-free pass builds a 1-bit-per-element
// mask (and OR-accumulates the outliers), then a ctz loop visits only the set
// bits, so the expensive part scales with the outlier count rather than n.
//
// Measured per 256-element block (scalar / SIMD):
//   exc  0%: 166 / 29 ns (5.8x)      exc 30%: 243 / 122 ns (2.0x)
//   exc 10%: 192 / 53 ns (3.6x)      exc 50%: 245 / 197 ns (1.2x)
template <typename T>
ABPFOR_INLINE unsigned collectOutliers(const T* in, unsigned n, unsigned b, T baseMask, uint8_t* positions,
                                       T* residuals, T& orOutliers)
{
    unsigned oc = 0;
    T orAcc = 0;

#if ABPFOR_ARCH_X86
    // The SIMD path requires n to be a multiple of 32 (it has no scalar tail) and
    // n <= 256 (maskWords holds n/32 words, positions[] is uint8_t). Both hold for
    // every caller: encodeBlockI4/I8 are only ever invoked with n == 128 or 256,
    // and ragged tails go to the scalar block codec in core/block.h instead.
    // Fall back to the scalar loop rather than silently corrupting if that changes.
    if constexpr (std::is_same_v<T, uint32_t>)
        if (n % 32 == 0 && n <= 256)
    {
        const __m256i sign = _mm256_set1_epi32(static_cast<int>(0x80000000u));
        const __m256i lim = _mm256_set1_epi32(static_cast<int>(baseMask ^ 0x80000000u));
        const unsigned nw = n / 32;

        uint32_t maskWords[8];
        __m256i acc = _mm256_setzero_si256();
        for (unsigned w = 0; w < nw; ++w)
        {
            uint32_t m = 0;
            for (unsigned j = 0; j < 4; ++j)
            {
                __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + w * 32 + j * 8));
                // Unsigned compare via bias: cmpgt_epi32 is signed only.
                __m256i gt = _mm256_cmpgt_epi32(_mm256_xor_si256(v, sign), lim);
                acc = _mm256_or_si256(acc, _mm256_and_si256(v, gt));
                m |= static_cast<uint32_t>(_mm256_movemask_ps(_mm256_castsi256_ps(gt))) << (j * 8);
            }
            maskWords[w] = m;
        }

        alignas(32) uint32_t lanes[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), acc);
        for (unsigned k = 0; k < 8; ++k) orAcc |= lanes[k];

        for (unsigned w = 0; w < nw; ++w)
        {
            uint32_t m = maskWords[w];
            while (m)
            {
                unsigned idx = w * 32 + static_cast<unsigned>(__builtin_ctz(m));
                positions[oc] = static_cast<uint8_t>(idx);
                residuals[oc] = in[idx] >> b;
                ++oc;
                m &= m - 1;
            }
        }
        orOutliers = orAcc;
        return oc;
    }
#endif

    for (unsigned i = 0; i < n; ++i)
    {
        T v = in[i];
        if (v > baseMask)
        {
            positions[oc] = static_cast<uint8_t>(i);
            residuals[oc] = v >> b;
            orAcc |= v;
            ++oc;
        }
    }
    orOutliers = orAcc;
    return oc;
}

#if ABPFOR_ARCH_X86
#define ABPFOR_HAS_FUSED_I8_BITMAP_ENCODE 1
ABPFOR_NOINLINE inline size_t encodeBitmapI8Full32(const uint32_t* in, uint8_t* out, unsigned b)
{
    constexpr unsigned N = 256;
    const uint32_t baseMask = mask<uint32_t>(b);
    alignas(32) uint32_t base[N], residuals[N], orLanes[8];
    uint8_t bitmap[N / 8];
    const __m256i maskVector = _mm256_set1_epi32(static_cast<int>(baseMask));
    __m256i orVector = _mm256_setzero_si256();
    unsigned count = 0;

    for (unsigned i = 0; i < N; i += 8)
    {
        const __m256i values = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i));
        _mm256_store_si256(reinterpret_cast<__m256i*>(base + i), _mm256_and_si256(values, maskVector));
        const __m256i high = _mm256_andnot_si256(maskVector, values);
        orVector = _mm256_or_si256(orVector, high);
        unsigned bits = static_cast<unsigned>(~_mm256_movemask_ps(_mm256_castsi256_ps(
            _mm256_cmpeq_epi32(high, _mm256_setzero_si256())))) & 0xffu;
        bitmap[i / 8] = static_cast<uint8_t>(bits);
        while (bits)
        {
            const unsigned lane = static_cast<unsigned>(__builtin_ctz(bits));
            residuals[count++] = in[i + lane] >> b;
            bits &= bits - 1;
        }
    }

    _mm256_store_si256(reinterpret_cast<__m256i*>(orLanes), orVector);
    uint32_t outlierOr = 0;
    for (uint32_t lane : orLanes) outlierOr |= lane;
    const unsigned maxBits = bitwidth(outlierOr);
    const unsigned pb = maxBits - b;

    uint8_t* op = out;
    *op++ = hdr::kBitmapOutlier | static_cast<uint8_t>(b);
    *op++ = static_cast<uint8_t>(pb);
    std::memcpy(op, bitmap, sizeof(bitmap));
    op += sizeof(bitmap);
    op = pack(residuals, count, op, pb);
    op = packI8(base, op, b);
    return static_cast<size_t>(op - out);
}
#endif

template <typename T> size_t encodeBlockI4(const T* in, unsigned n, uint8_t* out)
{
    static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>);
    constexpr unsigned W = kMaxBits<T>;

    unsigned pbx;
    unsigned b = optimalWidth(in, n, &pbx);
    uint8_t* op = out;

    if (b == 0 && pbx == 0)
    {
        *op++ = hdr::kSpecial | hdr::kAllZero;
        return static_cast<size_t>(op - out);
    }

    if (pbx == W + 2)
    {
        if (b >= hdr::kRaw)
        {
            *op++ = hdr::kSpecial | hdr::kRaw;
            std::memcpy(op, in, n * sizeof(T));
            op += n * sizeof(T);
        }
        else
        {
            *op++ = hdr::kSpecial | static_cast<uint8_t>(b);
            unsigned vb = (b + 7u) >> 3;
            storeu<T>(op, in[0]);
            op += vb;
        }
        return static_cast<size_t>(op - out);
    }

    if (pbx == 0)
    {
        if (b >= W)
        {
            *op++ = hdr::kSpecial | hdr::kRaw;
            std::memcpy(op, in, n * sizeof(T));
            op += n * sizeof(T);
            return static_cast<size_t>(op - out);
        }
        *op++ = hdr::kBitpackOnly | static_cast<uint8_t>(b);
        op = packInterleaved<T, 128>(in, op, b, packI4);
        return static_cast<size_t>(op - out);
    }

    T baseMask = mask<T>(b);
    uint8_t positions[128];
    T residuals[128];
    T orOutliers;
    unsigned oc = collectOutliers(in, n, b, baseMask, positions, residuals, orOutliers);
    unsigned maxBits = bitwidth(orOutliers);

    unsigned pb = (pbx == W + 1) ? (maxBits - b) : pbx;

    alignas(16) T masked[128];
    for (unsigned i = 0; i < n; ++i) masked[i] = in[i] & baseMask;

    if (pbx == W + 1) // Bitmap
    {
        *op++ = hdr::kBitmapOutlier | static_cast<uint8_t>(b);
        *op++ = static_cast<uint8_t>(pb);

        unsigned bitmapBytes = n / 8;
        std::memset(op, 0, bitmapBytes);
        for (unsigned i = 0; i < oc; ++i) op[positions[i] >> 3] |= uint8_t(1) << (positions[i] & 7);
        op += bitmapBytes;

        op = pack(residuals, oc, op, pb);
        op = packInterleaved<T, 128>(masked, op, b, packI4);
    }
    else // Sparse
    {
        *op++ = hdr::kSparseOutlier | static_cast<uint8_t>(b);
        *op++ = static_cast<uint8_t>(pb);
        *op++ = static_cast<uint8_t>(oc);

        std::memcpy(op, positions, oc);
        op += oc;

        op = pack(residuals, oc, op, pb);
        op = packInterleaved<T, 128>(masked, op, b, packI4);
    }

    return static_cast<size_t>(op - out);
}

template <typename T>
void applyBitmapPatch(T* out, const uint8_t* bitmapPtr, unsigned bitmapBytes, const T* residuals, unsigned b)
{
    unsigned ri = 0;
    for (unsigned wi = 0; wi < bitmapBytes / 8; ++wi)
    {
        uint64_t word;
        std::memcpy(&word, bitmapPtr + wi * 8, 8);
#if ABPFOR_BIG_ENDIAN
        word = bswap64(word);
#endif
        while (word)
        {
            unsigned bit = static_cast<unsigned>(__builtin_ctzll(word));
            unsigned pos = wi * 64 + bit;
            out[pos] |= residuals[ri++] << b;
            word &= word - 1;
        }
    }
}

// out[i] = start + i + 1, the delta1 expansion of an all-zero (constant-stride)
// block. This is the hottest path for sorted input with a stride of 1, where a
// whole block collapses to a single header byte and decode is pure output
// bandwidth. The compiler vectorises the scalar loop but keeps a scalar
// prologue/epilogue; writing the ramp directly is measurably faster.
// Measured on a 256-element block: 16.2 -> 13.2 ns (63 -> 77 GB/s).
template <typename T> ABPFOR_INLINE void fillRamp(T* out, unsigned n, T start)
{
#if ABPFOR_ARCH_X86
    if constexpr (std::is_same_v<T, uint32_t>)
        if (n % 8 == 0)
        {
            __m256i v = _mm256_add_epi32(_mm256_set1_epi32(static_cast<int>(start)),
                                         _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8));
            const __m256i step = _mm256_set1_epi32(8);
            for (unsigned i = 0; i < n; i += 8)
            {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), v);
                v = _mm256_add_epi32(v, step);
            }
            return;
        }
#endif
    for (unsigned i = 0; i < n; ++i) out[i] = start + T(i + 1);
}

template <typename T> void applySparsePatch(T* out, const uint8_t* posPtr, unsigned oc, const T* residuals, unsigned b)
{
    for (unsigned i = 0; i < oc; ++i) out[posPtr[i]] |= residuals[i] << b;
}

// The outlier paths need ~1KB of scratch (residuals + bitmap). Keeping them in
// the main decode body makes every call pay for that frame -- the prologue
// allocates it and sets the stack canary before the header byte is even read,
// so an all-zero block that decodes to a single ramp still pays full price.
//
// Out-of-lining them moves the buffers into a frame that only the outlier path
// enters. TurboPFor splits its exception handling the same way
// (p4Dec128Exceptions is __attribute__((noinline))).
//
// Measured on a 256-element all-zero delta1 block: 23.8 -> 21.3 ns.
template <typename T, unsigned BS, bool UseDelta, bool MinusOne>
ABPFOR_NOINLINE const uint8_t* decodeBitmapOutliers(const uint8_t* ip, unsigned n, T* out, T& carry, unsigned b)
{
    constexpr unsigned Words = BS / 64;
    unsigned pb = *ip++;
    unsigned bitmapBytes = n / 8;
    const uint8_t* bitmapPtr = ip;
    ip += bitmapBytes;

    alignas(32) uint64_t bmp64[Words];
    unsigned oc = 0;
    for (unsigned w = 0; w < bitmapBytes / 8; ++w)
    {
        uint64_t word = loadu<uint64_t>(bitmapPtr + w * 8);
        bmp64[w] = word;
        oc += static_cast<unsigned>(__builtin_popcountll(word));
    }

    alignas(32) uint32_t residuals[BS];
    ip = unpack(ip, oc, residuals, pb);

    if constexpr (UseDelta)
    {
        uint32_t c = carry;
        if constexpr (BS == 128) fusedDecodeI4<true, true, MinusOne>(ip, out, b, c, bmp64, residuals);
        else fusedDecodeI8<true, true, MinusOne>(ip, out, b, c, bmp64, residuals);
        carry = c;
    }
    else
    {
        uint32_t d = 0;
        if constexpr (BS == 128) fusedDecodeI4<false, true, MinusOne>(ip, out, b, d, bmp64, residuals);
        else fusedDecodeI8<false, true, MinusOne>(ip, out, b, d, bmp64, residuals);
    }
    return ip + packedBytes(n, b);
}

template <bool UseDelta, bool MinusOne>
ABPFOR_NOINLINE const uint8_t* decodeBitmapOutliersI4Full(const uint8_t* ip, uint32_t* out, uint32_t& carry,
                                                          unsigned b)
{
    // shortcut: full I4 is always 128 values and exactly two bitmap words; partial/I8 stay generic.
    unsigned pb = *ip++;
    alignas(32) uint64_t bmp64[2];
    bmp64[0] = loadu<uint64_t>(ip);
    bmp64[1] = loadu<uint64_t>(ip + 8);
    ip += 16;
    unsigned oc = static_cast<unsigned>(__builtin_popcountll(bmp64[0]) + __builtin_popcountll(bmp64[1]));

    alignas(32) uint32_t residuals[128];
    ip = unpack(ip, oc, residuals, pb);

    uint32_t c = carry;
    fusedDecodeI4<UseDelta, true, MinusOne>(ip, out, b, c, bmp64, residuals);
    if constexpr (UseDelta) carry = c;
    return ip + packedBytes(128, b);
}

template <typename T, unsigned BS, bool UseDelta, bool MinusOne>
ABPFOR_NOINLINE const uint8_t* decodeSparseOutliers(const uint8_t* ip, unsigned n, T* out, T& carry, unsigned b)
{
    constexpr unsigned Words = BS / 64;
    unsigned pb = *ip++;
    unsigned oc = *ip++;
    const uint8_t* posPtr = ip;
    ip += oc;

    alignas(32) uint32_t residuals[BS];
    ip = unpack(ip, oc, residuals, pb);

    if constexpr (UseDelta)
    {
        // Patch must precede the prefix sum, so it has to go through the fused
        // (bitmap-driven) kernel. Materialise the bitmap from the position list.
        alignas(32) uint64_t bmp64[Words] = {};
        for (unsigned i = 0; i < oc; ++i) bmp64[posPtr[i] / 64] |= uint64_t(1) << (posPtr[i] % 64);
        uint32_t c = carry;
        if constexpr (BS == 128) fusedDecodeI4<true, true, MinusOne>(ip, out, b, c, bmp64, residuals);
        else fusedDecodeI8<true, true, MinusOne>(ip, out, b, c, bmp64, residuals);
        carry = c;
    }
    else
    {
        // Without delta the patch commutes with the unpack, so scatter straight
        // through the position list: no bitmap to build and no per-lane patch
        // work in the kernel.
        uint32_t d = 0;
        if constexpr (BS == 128) fusedDecodeI4<false, false, MinusOne>(ip, out, b, d, nullptr, nullptr);
        else fusedDecodeI8<false, false, MinusOne>(ip, out, b, d, nullptr, nullptr);
        applySparsePatch(out, posPtr, oc, residuals, b);
    }
    return ip + packedBytes(n, b);
}

template <typename T, bool UseDelta, bool MinusOne>
size_t decodeBlockI4_impl(const uint8_t* in, unsigned n, T* out, T& carry)
{
    static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>);

    const uint8_t* ip = in;
    uint8_t h = *ip++;
    uint8_t type = h & hdr::kTypeMask;
    unsigned b = h & hdr::kBitsMask;

    if (type == hdr::kSpecial)
    {
        if (b == hdr::kAllZero)
        {
            if constexpr (UseDelta)
            {
                if constexpr (MinusOne)
                {
                    fillRamp(out, n, carry);
                    carry += T(n);
                }
                else
                {
                    std::fill_n(out, n, carry);
                }
            }
            else
                std::memset(out, 0, n * sizeof(T));
            return static_cast<size_t>(ip - in);
        }
        if (b != hdr::kAllZero && b != hdr::kRaw)
        {
            T val = loadu<T>(ip) & mask<T>(b);
            unsigned valBytes = (b + 7u) >> 3;
            ip += valBytes;
            if constexpr (UseDelta)
            {
                T inc = MinusOne ? (val + 1) : val;
                for (unsigned i = 0; i < n; ++i) out[i] = carry + inc * T(i + 1);
                carry += inc * T(n);
            }
            else
                std::fill_n(out, n, val);
            return static_cast<size_t>(ip - in);
        }
        if (b == hdr::kRaw)
        {
            std::memcpy(out, ip, n * sizeof(T));
            ip += n * sizeof(T);
            if constexpr (UseDelta)
            {
                if constexpr (MinusOne) { undelta(out, n, carry); carry = out[n - 1]; }
                else undelta0(out, n, carry);
            }
            return static_cast<size_t>(ip - in);
        }
        return 0;
    }

    if constexpr (std::is_same_v<T, uint32_t>)
    {
        if (type == hdr::kBitpackOnly)
        {
            if constexpr (UseDelta)
            {
                uint32_t c = carry;
                fusedDecodeI4<true, false, MinusOne>(ip, out, b, c, nullptr, nullptr);
                carry = c;
            }
            else
                unpackI4(ip, out, b);
            ip += packedBytes(n, b);
            return static_cast<size_t>(ip - in);
        }

        if (type == hdr::kBitmapOutlier)
        {
            if (n == 128)
                ip = decodeBitmapOutliersI4Full<UseDelta, MinusOne>(ip, out, carry, b);
            else
                ip = decodeBitmapOutliers<T, 128, UseDelta, MinusOne>(ip, n, out, carry, b);
            return static_cast<size_t>(ip - in);
        }

        if (type == hdr::kSparseOutlier)
        {
            ip = decodeSparseOutliers<T, 128, UseDelta, MinusOne>(ip, n, out, carry, b);
            return static_cast<size_t>(ip - in);
        }
    }
    else // uint64_t
    {
        if (type == hdr::kBitpackOnly)
        {
            unpackInterleaved<T, 128>(ip, out, b, unpackI4);
            ip += packedBytes(n, b);
            if constexpr (UseDelta)
            {
                if constexpr (MinusOne) { undelta(out, n, carry); carry = out[n - 1]; }
                else undelta0(out, n, carry);
            }
            return static_cast<size_t>(ip - in);
        }

        if (type == hdr::kBitmapOutlier)
        {
            unsigned pb = *ip++;
            unsigned bitmapBytes = n / 8;
            const uint8_t* bitmapPtr = ip;
            ip += bitmapBytes;
            unsigned oc = 0;
            for (unsigned i = 0; i < bitmapBytes; ++i) oc += static_cast<unsigned>(__builtin_popcount(bitmapPtr[i]));

            alignas(16) T residuals[128];
            ip = unpack(ip, oc, residuals, pb);

            unpackInterleaved<T, 128>(ip, out, b, unpackI4);
            ip += packedBytes(n, b);

            applyBitmapPatch(out, bitmapPtr, bitmapBytes, residuals, b);
            if constexpr (UseDelta)
            {
                if constexpr (MinusOne) { undelta(out, n, carry); carry = out[n - 1]; }
                else undelta0(out, n, carry);
            }
            return static_cast<size_t>(ip - in);
        }

        if (type == hdr::kSparseOutlier)
        {
            unsigned pb = *ip++;
            unsigned oc = *ip++;
            const uint8_t* posPtr = ip;
            ip += oc;

            alignas(16) T residuals[128];
            ip = unpack(ip, oc, residuals, pb);

            unpackInterleaved<T, 128>(ip, out, b, unpackI4);
            ip += packedBytes(n, b);

            applySparsePatch(out, posPtr, oc, residuals, b);
            if constexpr (UseDelta)
            {
                if constexpr (MinusOne) { undelta(out, n, carry); carry = out[n - 1]; }
                else undelta0(out, n, carry);
            }
            return static_cast<size_t>(ip - in);
        }
    }

    return 0;
}

// Interleave8 encode
template <typename T> size_t encodeBlockI8(const T* in, unsigned n, uint8_t* out)
{
    static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>);
    constexpr unsigned W = kMaxBits<T>;

    unsigned pbx;
    unsigned b = optimalWidth(in, n, &pbx);
    uint8_t* op = out;

    if (b == 0 && pbx == 0)
    {
        *op++ = hdr::kSpecial | hdr::kAllZero;
        return static_cast<size_t>(op - out);
    }
    if (pbx == W + 2)
    {
        if (b >= hdr::kRaw)
        {
            *op++ = hdr::kSpecial | hdr::kRaw;
            std::memcpy(op, in, n * sizeof(T));
            op += n * sizeof(T);
        }
        else
        {
            *op++ = hdr::kSpecial | static_cast<uint8_t>(b);
            unsigned vb = (b + 7u) >> 3;
            storeu<T>(op, in[0]);
            op += vb;
        }
        return static_cast<size_t>(op - out);
    }

    if (pbx == 0)
    {
        if (b >= W)
        {
            *op++ = hdr::kSpecial | hdr::kRaw;
            std::memcpy(op, in, n * sizeof(T));
            op += n * sizeof(T);
            return static_cast<size_t>(op - out);
        }
        *op++ = hdr::kBitpackOnly | static_cast<uint8_t>(b);
        op = packInterleaved<T, 256>(in, op, b, packI8);
        return static_cast<size_t>(op - out);
    }

#if ABPFOR_ARCH_X86
    // shortcut: only the measured full uint32_t I8 bitmap case is fused; partial, uint64_t, and non-x86 paths stay generic.
    if constexpr (std::is_same_v<T, uint32_t>)
        if (n == 256 && pbx == W + 1) return encodeBitmapI8Full32(in, out, b);
#endif

    T baseMask = mask<T>(b);
    uint8_t positions[256];
    T residuals[256];
    T orOutliers;
    unsigned oc = collectOutliers(in, n, b, baseMask, positions, residuals, orOutliers);
    unsigned maxBits = bitwidth(orOutliers);

    unsigned pb = (pbx == W + 1) ? (maxBits - b) : pbx;

    alignas(32) T masked[256];
    for (unsigned i = 0; i < n; ++i) masked[i] = in[i] & baseMask;

    if (pbx == W + 1) // Bitmap
    {
        *op++ = hdr::kBitmapOutlier | static_cast<uint8_t>(b);
        *op++ = static_cast<uint8_t>(pb);
        unsigned bitmapBytes = n / 8;
        std::memset(op, 0, bitmapBytes);
        for (unsigned i = 0; i < oc; ++i) op[positions[i] >> 3] |= uint8_t(1) << (positions[i] & 7);
        op += bitmapBytes;
        op = pack(residuals, oc, op, pb);
        op = packInterleaved<T, 256>(masked, op, b, packI8);
    }
    else // Sparse
    {
        *op++ = hdr::kSparseOutlier | static_cast<uint8_t>(b);
        *op++ = static_cast<uint8_t>(pb);
        *op++ = static_cast<uint8_t>(oc);
        std::memcpy(op, positions, oc);
        op += oc;
        op = pack(residuals, oc, op, pb);
        op = packInterleaved<T, 256>(masked, op, b, packI8);
    }

    return static_cast<size_t>(op - out);
}

template <typename T, bool MinusOne = false>
size_t decodeBlockI4(const uint8_t* in, unsigned n, T* out, T& carry)
{
    return decodeBlockI4_impl<T, true, MinusOne>(in, n, out, carry);
}

template <typename T>
size_t decodeBlockI4(const uint8_t* in, unsigned n, T* out)
{
    T unused{};
    return decodeBlockI4_impl<T, false, true>(in, n, out, unused);
}

template <typename T, bool UseDelta, bool MinusOne>
size_t decodeBlockI8_impl(const uint8_t* in, unsigned n, T* out, T& carry)
{
    static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>);

    const uint8_t* ip = in;
    uint8_t h = *ip++;
    uint8_t type = h & hdr::kTypeMask;
    unsigned b = h & hdr::kBitsMask;

    if (type == hdr::kSpecial)
    {
        if (b == hdr::kAllZero)
        {
            if constexpr (UseDelta)
            {
                if constexpr (MinusOne)
                {
                    fillRamp(out, n, carry);
                    carry += T(n);
                }
                else
                {
                    std::fill_n(out, n, carry);
                }
            }
            else
                std::memset(out, 0, n * sizeof(T));
            return static_cast<size_t>(ip - in);
        }
        if (b != hdr::kAllZero && b != hdr::kRaw)
        {
            T val = loadu<T>(ip) & mask<T>(b);
            unsigned vb = (b + 7u) >> 3;
            ip += vb;
            if constexpr (UseDelta)
            {
                T inc = MinusOne ? (val + 1) : val;
                for (unsigned i = 0; i < n; ++i) out[i] = carry + inc * T(i + 1);
                carry += inc * T(n);
            }
            else
                std::fill_n(out, n, val);
            return static_cast<size_t>(ip - in);
        }
        if (b == hdr::kRaw)
        {
            std::memcpy(out, ip, n * sizeof(T));
            ip += n * sizeof(T);
            if constexpr (UseDelta)
            {
                if constexpr (MinusOne) { undelta(out, n, carry); carry = out[n - 1]; }
                else undelta0(out, n, carry);
            }
            return static_cast<size_t>(ip - in);
        }
        return 0;
    }

    if constexpr (std::is_same_v<T, uint32_t>)
    {
        if (type == hdr::kBitpackOnly)
        {
            if constexpr (UseDelta)
            {
                uint32_t c = carry;
                fusedDecodeI8<true, false, MinusOne>(ip, out, b, c, nullptr, nullptr);
                carry = c;
            }
            else
                unpackI8(ip, out, b);
            ip += packedBytes(n, b);
            return static_cast<size_t>(ip - in);
        }

        if (type == hdr::kBitmapOutlier)
        {
            ip = decodeBitmapOutliers<T, 256, UseDelta, MinusOne>(ip, n, out, carry, b);
            return static_cast<size_t>(ip - in);
        }

        if (type == hdr::kSparseOutlier)
        {
            ip = decodeSparseOutliers<T, 256, UseDelta, MinusOne>(ip, n, out, carry, b);
            return static_cast<size_t>(ip - in);
        }
    }
    else // uint64_t
    {
        if (type == hdr::kBitpackOnly)
        {
            unpackInterleaved<T, 256>(ip, out, b, unpackI8);
            ip += packedBytes(n, b);
            if constexpr (UseDelta)
            {
                if constexpr (MinusOne) { undelta(out, n, carry); carry = out[n - 1]; }
                else undelta0(out, n, carry);
            }
            return static_cast<size_t>(ip - in);
        }

        if (type == hdr::kBitmapOutlier)
        {
            unsigned pb = *ip++;
            unsigned bitmapBytes = n / 8;
            const uint8_t* bitmapPtr = ip;
            ip += bitmapBytes;
            unsigned oc = 0;
            for (unsigned i = 0; i < bitmapBytes; ++i) oc += static_cast<unsigned>(__builtin_popcount(bitmapPtr[i]));

            alignas(32) T residuals[256];
            ip = unpack(ip, oc, residuals, pb);

            unpackInterleaved<T, 256>(ip, out, b, unpackI8);
            ip += packedBytes(n, b);

            applyBitmapPatch(out, bitmapPtr, bitmapBytes, residuals, b);
            if constexpr (UseDelta)
            {
                if constexpr (MinusOne) { undelta(out, n, carry); carry = out[n - 1]; }
                else undelta0(out, n, carry);
            }
            return static_cast<size_t>(ip - in);
        }

        if (type == hdr::kSparseOutlier)
        {
            unsigned pb = *ip++;
            unsigned oc = *ip++;
            const uint8_t* posPtr = ip;
            ip += oc;

            alignas(32) T residuals[256];
            ip = unpack(ip, oc, residuals, pb);

            unpackInterleaved<T, 256>(ip, out, b, unpackI8);
            ip += packedBytes(n, b);

            applySparsePatch(out, posPtr, oc, residuals, b);
            if constexpr (UseDelta)
            {
                if constexpr (MinusOne) { undelta(out, n, carry); carry = out[n - 1]; }
                else undelta0(out, n, carry);
            }
            return static_cast<size_t>(ip - in);
        }
    }

    return 0;
}

template <typename T, bool MinusOne = false>
size_t decodeBlockI8(const uint8_t* in, unsigned n, T* out, T& carry)
{
    return decodeBlockI8_impl<T, true, MinusOne>(in, n, out, carry);
}

template <typename T>
size_t decodeBlockI8(const uint8_t* in, unsigned n, T* out)
{
    T unused{};
    return decodeBlockI8_impl<T, false, true>(in, n, out, unused);
}

} // namespace abpfor::detail

// ---------------------------------------------------------------------------
// Internal only — old template API for benchmarks/tests
// ---------------------------------------------------------------------------
namespace abpfor {

template <typename T> constexpr size_t maxCompressedSize(size_t n)
{
    return n * sizeof(T) + (n / 128 + 2) * 4;
}

enum class Layout : uint8_t { Scalar, Interleave4, Interleave8 };
enum class Delta : uint8_t { None, Delta0, Delta1 };

// ---------------------------------------------------------------------------
// encode<Layout, Delta>
// ---------------------------------------------------------------------------

template <Layout L, Delta D, typename T>
size_t encode(std::span<const T> in, uint8_t* out, T start, unsigned blockSize = 0)
{
    const unsigned n = static_cast<unsigned>(in.size());
    const T* ptr = in.data();
    uint8_t* op = out;

    if constexpr (L == Layout::Scalar)
    {
        // shortcut: blockSize param only effective for Scalar; I4/I8 have structural sizes
        const unsigned BS = blockSize > 0 ? blockSize : 128;
        unsigned pos = 0;
        T s = start;
        while (pos + BS <= n)
        {
            if constexpr (D == Delta::Delta1)
            {
                op += encodeBlockDelta1(ptr + pos, BS, op, s);
                s = ptr[pos + BS - 1];
            }
            else if constexpr (D == Delta::Delta0)
            {
                op += encodeBlockDelta0(ptr + pos, BS, op, s);
                s = ptr[pos + BS - 1];
            }
            else
            {
                op += encodeBlock(ptr + pos, BS, op);
            }
            pos += BS;
        }
        if (unsigned tail = n - pos; tail > 0)
        {
            if constexpr (D == Delta::Delta1)
                op += encodeBlockDelta1(ptr + pos, tail, op, s);
            else if constexpr (D == Delta::Delta0)
                op += encodeBlockDelta0(ptr + pos, tail, op, s);
            else
                op += encodeBlock(ptr + pos, tail, op);
        }
    }
    else if constexpr (L == Layout::Interleave4)
    {
        constexpr unsigned BS = 128;
        T tmp[BS];
        unsigned pos = 0;
        T s = start;

        while (pos + BS <= n)
        {
            if constexpr (D == Delta::Delta1)
            {
                delta(ptr + pos, BS, tmp, s);
                s = ptr[pos + BS - 1];
                op += detail::encodeBlockI4(tmp, BS, op);
            }
            else if constexpr (D == Delta::Delta0)
            {
                delta0(ptr + pos, BS, tmp, s);
                s = ptr[pos + BS - 1];
                op += detail::encodeBlockI4(tmp, BS, op);
            }
            else
            {
                op += detail::encodeBlockI4(ptr + pos, BS, op);
            }
            pos += BS;
        }

        // Tail: use scalar block
        if (pos < n)
        {
            unsigned tail = n - pos;
            if constexpr (D == Delta::Delta1)
                op += encodeBlockDelta1(ptr + pos, tail, op, s);
            else if constexpr (D == Delta::Delta0)
                op += encodeBlockDelta0(ptr + pos, tail, op, s);
            else
                op += encodeBlock(ptr + pos, tail, op);
        }
    }
    else if constexpr (L == Layout::Interleave8)
    {
        constexpr unsigned BS = 256;
        T tmp[BS];
        unsigned pos = 0;
        T s = start;

        while (pos + BS <= n)
        {
            if constexpr (D == Delta::Delta1)
            {
                delta(ptr + pos, BS, tmp, s);
                s = ptr[pos + BS - 1];
                op += detail::encodeBlockI8(tmp, BS, op);
            }
            else if constexpr (D == Delta::Delta0)
            {
                delta0(ptr + pos, BS, tmp, s);
                s = ptr[pos + BS - 1];
                op += detail::encodeBlockI8(tmp, BS, op);
            }
            else
            {
                op += detail::encodeBlockI8(ptr + pos, BS, op);
            }
            pos += BS;
        }

        if (pos < n)
        {
            unsigned tail = n - pos;
            if constexpr (D == Delta::Delta1)
                op += encodeBlockDelta1(ptr + pos, tail, op, s);
            else if constexpr (D == Delta::Delta0)
                op += encodeBlockDelta0(ptr + pos, tail, op, s);
            else
                op += encodeBlock(ptr + pos, tail, op);
        }
    }

    return static_cast<size_t>(op - out);
}

// ---------------------------------------------------------------------------
// decode<Layout, Delta>
// ---------------------------------------------------------------------------

template <Layout L, Delta D, typename T>
size_t decode(const uint8_t* in, std::span<T> out, T start, unsigned blockSize = 0)
{
    const unsigned n = static_cast<unsigned>(out.size());
    T* ptr = out.data();
    const uint8_t* ip = in;

    if constexpr (L == Layout::Scalar)
    {
        const unsigned BS = blockSize > 0 ? blockSize : 128;
        unsigned pos = 0;
        T carry = start;
        while (pos + BS <= n)
        {
            if constexpr (D == Delta::Delta1)
            {
                ip += decodeBlockDelta1(ip, BS, ptr + pos, carry);
                carry = ptr[pos + BS - 1];
            }
            else if constexpr (D == Delta::Delta0)
            {
                ip += decodeBlockDelta0(ip, BS, ptr + pos, carry);
            }
            else
            {
                ip += decodeBlock(ip, BS, ptr + pos);
            }
            pos += BS;
        }
        unsigned tail = n - pos;
        if (tail > 0)
        {
            if constexpr (D == Delta::Delta1)
                ip += decodeBlockDelta1(ip, tail, ptr + pos, carry);
            else if constexpr (D == Delta::Delta0)
                ip += decodeBlockDelta0(ip, tail, ptr + pos, carry);
            else
                ip += decodeBlock(ip, tail, ptr + pos);
        }
    }
    else if constexpr (L == Layout::Interleave4)
    {
        constexpr unsigned BS = 128;
        unsigned pos = 0;
        T carry = start;

        while (pos + BS <= n)
        {
            if constexpr (D == Delta::Delta1)
            {
                ip += detail::decodeBlockI4<T, true>(ip, BS, ptr + pos, carry);
            }
            else if constexpr (D == Delta::Delta0)
            {
                ip += detail::decodeBlockI4<T>(ip, BS, ptr + pos, carry);
            }
            else
            {
                ip += detail::decodeBlockI4<T>(ip, BS, ptr + pos);
            }
            pos += BS;
        }

        if (pos < n)
        {
            unsigned tail = n - pos;
            if constexpr (D == Delta::Delta1)
                ip += decodeBlockDelta1(ip, tail, ptr + pos, carry);
            else if constexpr (D == Delta::Delta0)
                ip += decodeBlockDelta0(ip, tail, ptr + pos, carry);
            else
                ip += decodeBlock(ip, tail, ptr + pos);
        }
    }
    else if constexpr (L == Layout::Interleave8)
    {
        constexpr unsigned BS = 256;
        unsigned pos = 0;
        T carry = start;

        while (pos + BS <= n)
        {
            if constexpr (D == Delta::Delta1)
            {
                ip += detail::decodeBlockI8<T, true>(ip, BS, ptr + pos, carry);
            }
            else if constexpr (D == Delta::Delta0)
            {
                ip += detail::decodeBlockI8<T>(ip, BS, ptr + pos, carry);
            }
            else
            {
                ip += detail::decodeBlockI8<T>(ip, BS, ptr + pos);
            }
            pos += BS;
        }

        if (pos < n)
        {
            unsigned tail = n - pos;
            if constexpr (D == Delta::Delta1)
                ip += decodeBlockDelta1(ip, tail, ptr + pos, carry);
            else if constexpr (D == Delta::Delta0)
                ip += decodeBlockDelta0(ip, tail, ptr + pos, carry);
            else
                ip += decodeBlock(ip, tail, ptr + pos);
        }
    }

    return static_cast<size_t>(ip - in);
}

} // namespace abpfor
