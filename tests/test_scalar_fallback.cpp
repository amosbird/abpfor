// Test: exercises the scalar_interleaved fallback path through the full
// encode/decode API (block header + payload), not just raw pack/unpack.
//
// Strategy: re-implement decodeBlockI4/I8 using scalar_ref:: primitives,
// then verify that data encoded by the normal (SSE/AVX2) encode<> API
// decodes identically via this scalar block decoder. This simulates what
// happens when data is encoded on x86 and decoded on ARM (scalar path).

#include "abpfor.h"
#include "detail/codec.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <random>

#include <vector>

// --- Scalar reference primitives (copied from test_cross_platform.cpp) ---
namespace scalar_ref {
    using abpfor::packedBytes;
    using abpfor::detail::leToNative;
    using abpfor::detail::copyU32ArrayFromLe;
    namespace detail {
        template <unsigned Lanes>
        inline const uint8_t* unpackInterleaved(const uint8_t* in, uint32_t* out, unsigned b, unsigned count)
        {
            if (b == 0) { std::memset(out, 0, count * sizeof(uint32_t)); return in; }
            if (b == 32) {
                copyU32ArrayFromLe(out, in, count);
                return in + count * sizeof(uint32_t);
            }
            const uint32_t* raw = reinterpret_cast<const uint32_t*>(in);
            unsigned totalBits = 32u * b;
            unsigned numStripes = (totalBits + 31u) / 32u;
            unsigned numWords = numStripes * Lanes;
            uint32_t stripes[32 * Lanes];
            for (unsigned i = 0; i < numWords; ++i)
                stripes[i] = leToNative(raw[i]);
            const uint32_t vmask = (1u << b) - 1u;
            for (unsigned g = 0; g < 32; ++g) {
                unsigned bitOff = g * b;
                unsigned stripeIdx = bitOff / 32u;
                unsigned shift = bitOff % 32u;
                bool spans = (shift + b > 32u) && (b < 32u);
                for (unsigned lane = 0; lane < Lanes; ++lane) {
                    uint32_t val = stripes[stripeIdx * Lanes + lane] >> shift;
                    if (spans) val |= stripes[(stripeIdx + 1) * Lanes + lane] << (32u - shift);
                    out[g * Lanes + lane] = val & vmask;
                }
            }
            return in + packedBytes(count, b);
        }

        template <unsigned Lanes, bool Delta, bool Patch>
        inline void fusedDecodeInterleaved(const uint8_t* in, uint32_t* out, unsigned b, uint32_t& carry,
                                           const uint64_t* bitmap, const uint32_t* residuals, unsigned count)
        {
            const uint32_t* pex = residuals;
            if (b == 0) {
                if constexpr (Delta) {
                    for (unsigned g = 0; g < 32; ++g)
                        for (unsigned lane = 0; lane < Lanes; ++lane) {
                            uint32_t val = 0;
                            if constexpr (Patch) {
                                unsigned idx = g * Lanes + lane;
                                if (bitmap[idx / 64] & (uint64_t(1) << (idx % 64))) val = *pex++;
                            }
                            carry += 1u + val;
                            out[g * Lanes + lane] = carry;
                        }
                } else {
                    std::memset(out, 0, count * sizeof(uint32_t));
                    if constexpr (Patch) {
                        for (unsigned i = 0; i < count; ++i)
                            if (bitmap[i / 64] & (uint64_t(1) << (i % 64))) out[i] = *pex++;
                    }
                }
                return;
            }
            if (b == 32) {
                copyU32ArrayFromLe(out, in, count);
                if constexpr (Delta) {
                    for (unsigned i = 0; i < count; ++i) { carry += out[i] + 1u; out[i] = carry; }
                }
                return;
            }
            const uint32_t* raw2 = reinterpret_cast<const uint32_t*>(in);
            unsigned totalBits2 = 32u * b;
            unsigned numStripes2 = (totalBits2 + 31u) / 32u;
            unsigned numWords2 = numStripes2 * Lanes;
            uint32_t stripes[32 * Lanes];
            for (unsigned i = 0; i < numWords2; ++i)
                stripes[i] = leToNative(raw2[i]);
            const uint32_t vmask = (1u << b) - 1u;
            for (unsigned g = 0; g < 32; ++g) {
                unsigned bitOff = g * b;
                unsigned stripeIdx = bitOff / 32u;
                unsigned shift = bitOff % 32u;
                bool spans = (shift + b > 32u) && (b < 32u);
                for (unsigned lane = 0; lane < Lanes; ++lane) {
                    uint32_t val = stripes[stripeIdx * Lanes + lane] >> shift;
                    if (spans) val |= stripes[(stripeIdx + 1) * Lanes + lane] << (32u - shift);
                    val &= vmask;
                    if constexpr (Patch) {
                        unsigned idx = g * Lanes + lane;
                        if (bitmap[idx / 64] & (uint64_t(1) << (idx % 64))) val += (*pex++) << b;
                    }
                    if constexpr (Delta) { carry += val + 1u; out[g * Lanes + lane] = carry; }
                    else { out[g * Lanes + lane] = val; }
                }
            }
        }
    } // namespace detail

    inline const uint8_t* unpackI4(const uint8_t* in, uint32_t* out, unsigned b)
    { return detail::unpackInterleaved<4>(in, out, b, 128); }
    template <bool Delta, bool Patch>
    inline void fusedDecodeI4(const uint8_t* in, uint32_t* out, unsigned b, uint32_t& carry,
                              const uint64_t* bitmap, const uint32_t* residuals)
    { detail::fusedDecodeInterleaved<4, Delta, Patch>(in, out, b, carry, bitmap, residuals, 128); }

    inline const uint8_t* unpackI8(const uint8_t* in, uint32_t* out, unsigned b)
    { return detail::unpackInterleaved<8>(in, out, b, 256); }
    template <bool Delta, bool Patch>
    inline void fusedDecodeI8(const uint8_t* in, uint32_t* out, unsigned b, uint32_t& carry,
                              const uint64_t* bitmap, const uint32_t* residuals)
    { detail::fusedDecodeInterleaved<8, Delta, Patch>(in, out, b, carry, bitmap, residuals, 256); }
} // namespace scalar_ref

// --- Scalar block decoder (mirrors abpfor::detail::decodeBlockI4 but uses scalar_ref) ---
// This is the key: it re-implements the full block decode (header parsing + payload)
// using scalar primitives, simulating what the ARM/scalar path would do.

namespace scalar_block {

using namespace abpfor;
using namespace abpfor::hdr;

template <unsigned BlockSize>
size_t decodeBlock(const uint8_t* in, unsigned n, uint32_t* out, bool useDelta, uint32_t& carry)
{
    static_assert(BlockSize == 128 || BlockSize == 256);
    constexpr unsigned Lanes = BlockSize / 32;

    const uint8_t* ip = in;
    uint8_t h = *ip++;
    uint8_t type = h & hdr::kTypeMask;
    unsigned b = h & hdr::kBitsMask;

    if (type == hdr::kSpecial)
    {
        if (b == hdr::kAllZero)
        {
            if (useDelta) {
                for (unsigned i = 0; i < n; ++i) out[i] = carry + uint32_t(i + 1);
                carry += uint32_t(n);
            } else
                std::memset(out, 0, n * sizeof(uint32_t));
            return size_t(ip - in);
        }
        if (b != hdr::kAllZero && b != hdr::kRaw)
        {
            uint32_t val = abpfor::loadu<uint32_t>(ip) & abpfor::mask<uint32_t>(b);
            unsigned valBytes = (b + 7u) >> 3;
            ip += valBytes;
            if (useDelta) {
                uint32_t inc = val + 1;
                for (unsigned i = 0; i < n; ++i) out[i] = carry + inc * uint32_t(i + 1);
                carry += inc * uint32_t(n);
            } else
                std::fill_n(out, n, val);
            return size_t(ip - in);
        }
        if (b == hdr::kRaw)
        {
            std::memcpy(out, ip, n * sizeof(uint32_t));
            ip += n * sizeof(uint32_t);
            if (useDelta) abpfor::undelta(out, n, carry);
            return size_t(ip - in);
        }
        return 0;
    }

    if (type == hdr::kBitpackOnly)
    {
        if (useDelta) {
            uint32_t c = carry;
            if constexpr (BlockSize == 128)
                scalar_ref::fusedDecodeI4<true, false>(ip, out, b, c, nullptr, nullptr);
            else
                scalar_ref::fusedDecodeI8<true, false>(ip, out, b, c, nullptr, nullptr);
            carry = c;
        } else {
            if constexpr (BlockSize == 128)
                scalar_ref::unpackI4(ip, out, b);
            else
                scalar_ref::unpackI8(ip, out, b);
        }
        ip += packedBytes(n, b);
        return size_t(ip - in);
    }

    if (type == hdr::kBitmapOutlier)
    {
        unsigned pb = *ip++;
        unsigned bitmapBytes = n / 8;
        const uint8_t* bitmapPtr = ip;
        ip += bitmapBytes;

        unsigned oc = 0;
        for (unsigned i = 0; i < bitmapBytes; ++i) oc += unsigned(__builtin_popcount(bitmapPtr[i]));

        alignas(16) uint32_t residuals[256];
        ip = unpack(ip, oc, residuals, pb);

        alignas(16) uint64_t bmp64[4] = {};
        std::memcpy(bmp64, bitmapPtr, bitmapBytes);

        if (useDelta) {
            uint32_t c = carry;
            if constexpr (BlockSize == 128)
                scalar_ref::fusedDecodeI4<true, true>(ip, out, b, c, bmp64, residuals);
            else
                scalar_ref::fusedDecodeI8<true, true>(ip, out, b, c, bmp64, residuals);
            carry = c;
        } else {
            uint32_t dummy = 0;
            if constexpr (BlockSize == 128)
                scalar_ref::fusedDecodeI4<false, true>(ip, out, b, dummy, bmp64, residuals);
            else
                scalar_ref::fusedDecodeI8<false, true>(ip, out, b, dummy, bmp64, residuals);
        }
        ip += packedBytes(n, b);
        return size_t(ip - in);
    }

    if (type == hdr::kSparseOutlier)
    {
        unsigned pb = *ip++;
        unsigned oc = *ip++;

        const uint8_t* posPtr = ip;
        ip += oc;

        alignas(16) uint32_t residuals[256];
        ip = unpack(ip, oc, residuals, pb);

        alignas(16) uint64_t bmp64[4] = {};
        for (unsigned i = 0; i < oc; ++i) bmp64[posPtr[i] / 64] |= uint64_t(1) << (posPtr[i] % 64);

        if (useDelta) {
            uint32_t c = carry;
            if constexpr (BlockSize == 128)
                scalar_ref::fusedDecodeI4<true, true>(ip, out, b, c, bmp64, residuals);
            else
                scalar_ref::fusedDecodeI8<true, true>(ip, out, b, c, bmp64, residuals);
            carry = c;
        } else {
            uint32_t dummy = 0;
            if constexpr (BlockSize == 128)
                scalar_ref::fusedDecodeI4<false, true>(ip, out, b, dummy, bmp64, residuals);
            else
                scalar_ref::fusedDecodeI8<false, true>(ip, out, b, dummy, bmp64, residuals);
        }
        ip += packedBytes(n, b);
        return size_t(ip - in);
    }

    return 0;
}

// Full decode using scalar block decoder (mirrors abpfor::decode<Interleave4/8, Delta>)
template <unsigned BS, bool UseDelta>
size_t decode(const uint8_t* in, unsigned n, uint32_t* ptr, uint32_t start)
{
    const uint8_t* ip = in;
    uint32_t carry = start;
    unsigned pos = 0;

    while (pos + BS <= n) {
        ip += decodeBlock<BS>(ip, BS, ptr + pos, UseDelta, carry);
        pos += BS;
    }
    // Tail handled by scalar path in real impl; for this test we only
    // feed multiples of BS to the interleaved path.
    return size_t(ip - in);
}

} // namespace scalar_block

// ---------------------------------------------------------------------------
// Test infrastructure
// ---------------------------------------------------------------------------

static std::mt19937 rng(123);
static int failures = 0;

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); ++failures; } } while(0)

static std::vector<uint32_t> randomValues(unsigned count, unsigned maxBits)
{
    std::vector<uint32_t> v(count);
    uint32_t mask = maxBits == 32 ? ~0u : (1u << maxBits) - 1u;
    for (auto& x : v) x = rng() & mask;
    return v;
}

static std::vector<uint32_t> sortedValues(unsigned count, unsigned maxBits)
{
    auto v = randomValues(count, maxBits);
    std::sort(v.begin(), v.end());
    // Ensure strictly increasing for delta encoding
    for (unsigned i = 1; i < count; ++i)
        if (v[i] <= v[i-1]) v[i] = v[i-1] + 1;
    return v;
}

// ---------------------------------------------------------------------------
// Test: Full encode→scalar decode roundtrip (Interleave4, no delta)
// ---------------------------------------------------------------------------
static void testI4NoDelta()
{
    printf("  I4 full encode → scalar decode (no delta)...\n");
    constexpr unsigned N = 128 * 4; // multiple blocks

    for (unsigned bits : {1u, 5u, 8u, 13u, 17u, 24u, 31u, 32u}) {
        auto input = randomValues(N, bits);
        std::vector<uint8_t> buf(N * 5); // generous
        std::vector<uint32_t> out(N);

        size_t enc = abpfor::b128::encode(input.data(), N, buf.data());

        size_t dec = scalar_block::decode<128, false>(
            buf.data(), N, out.data(), uint32_t(0));

        CHECK(enc == dec, "I4 noDelta bits=%u: enc=%zu dec=%zu", bits, enc, dec);
        for (unsigned i = 0; i < N; ++i)
            CHECK(out[i] == input[i], "I4 noDelta bits=%u i=%u: got %u want %u", bits, i, out[i], input[i]);
    }
}

// ---------------------------------------------------------------------------
// Test: Full encode→scalar decode roundtrip (Interleave4, Delta1)
// ---------------------------------------------------------------------------
static void testI4Delta()
{
    printf("  I4 full encode → scalar decode (delta)...\n");
    constexpr unsigned N = 128 * 4;

    for (unsigned bits : {1u, 5u, 10u, 16u, 20u, 28u}) {
        auto input = sortedValues(N, bits);
        std::vector<uint8_t> buf(N * 5);
        std::vector<uint32_t> out(N);

        size_t enc = abpfor::b128::encodeDelta1(input.data(), N, buf.data(), uint32_t(0));

        size_t dec = scalar_block::decode<128, true>(
            buf.data(), N, out.data(), uint32_t(0));

        CHECK(enc == dec, "I4 delta bits=%u: enc=%zu dec=%zu", bits, enc, dec);
        for (unsigned i = 0; i < N; ++i)
            CHECK(out[i] == input[i], "I4 delta bits=%u i=%u: got %u want %u", bits, i, out[i], input[i]);
    }
}

// ---------------------------------------------------------------------------
// Test: Full encode→scalar decode roundtrip (Interleave8, no delta)
// ---------------------------------------------------------------------------
static void testI8NoDelta()
{
    printf("  I8 full encode → scalar decode (no delta)...\n");
    constexpr unsigned N = 256 * 3;

    for (unsigned bits : {1u, 7u, 12u, 19u, 25u, 32u}) {
        auto input = randomValues(N, bits);
        std::vector<uint8_t> buf(N * 5);
        std::vector<uint32_t> out(N);

        size_t enc = abpfor::b256::encode(input.data(), N, buf.data());

        size_t dec = scalar_block::decode<256, false>(
            buf.data(), N, out.data(), uint32_t(0));

        CHECK(enc == dec, "I8 noDelta bits=%u: enc=%zu dec=%zu", bits, enc, dec);
        for (unsigned i = 0; i < N; ++i)
            CHECK(out[i] == input[i], "I8 noDelta bits=%u i=%u: got %u want %u", bits, i, out[i], input[i]);
    }
}

// ---------------------------------------------------------------------------
// Test: Full encode→scalar decode roundtrip (Interleave8, Delta1)
// ---------------------------------------------------------------------------
static void testI8Delta()
{
    printf("  I8 full encode → scalar decode (delta)...\n");
    constexpr unsigned N = 256 * 3;

    for (unsigned bits : {1u, 6u, 11u, 18u, 24u}) {
        auto input = sortedValues(N, bits);
        std::vector<uint8_t> buf(N * 5);
        std::vector<uint32_t> out(N);

        size_t enc = abpfor::b256::encodeDelta1(input.data(), N, buf.data(), uint32_t(0));

        size_t dec = scalar_block::decode<256, true>(
            buf.data(), N, out.data(), uint32_t(0));

        CHECK(enc == dec, "I8 delta bits=%u: enc=%zu dec=%zu", bits, enc, dec);
        for (unsigned i = 0; i < N; ++i)
            CHECK(out[i] == input[i], "I8 delta bits=%u i=%u: got %u want %u", bits, i, out[i], input[i]);
    }
}

// ---------------------------------------------------------------------------
// Test: Outlier-heavy data (forces bitmap/sparse outlier paths)
// ---------------------------------------------------------------------------
static void testOutlierPaths()
{
    printf("  Outlier paths (bitmap + sparse) through scalar decode...\n");
    constexpr unsigned N = 128 * 2;

    // Sparse outliers: most values small, a few large
    {
        auto input = randomValues(N, 4); // base: 4 bits
        // Inject a few outliers
        for (unsigned i = 0; i < N; i += 17)
            input[i] = rng() | (1u << 20); // needs >20 bits
        std::vector<uint8_t> buf(N * 5);
        std::vector<uint32_t> out(N);

        size_t enc = abpfor::b128::encode(input.data(), N, buf.data());
        size_t dec = scalar_block::decode<128, false>(
            buf.data(), N, out.data(), uint32_t(0));

        CHECK(enc == dec, "sparse outlier: enc=%zu dec=%zu", enc, dec);
        for (unsigned i = 0; i < N; ++i)
            CHECK(out[i] == input[i], "sparse outlier i=%u: got %u want %u", i, out[i], input[i]);
    }

    // Bitmap outliers: many values exceed base width
    {
        auto input = randomValues(N, 8); // base: 8 bits
        // Make ~40% of values outliers
        for (unsigned i = 0; i < N; i += 3)
            input[i] |= (rng() & 0xFFu) << 8;
        std::vector<uint8_t> buf(N * 5);
        std::vector<uint32_t> out(N);

        size_t enc = abpfor::b128::encode(input.data(), N, buf.data());
        size_t dec = scalar_block::decode<128, false>(
            buf.data(), N, out.data(), uint32_t(0));

        CHECK(enc == dec, "bitmap outlier: enc=%zu dec=%zu", enc, dec);
        for (unsigned i = 0; i < N; ++i)
            CHECK(out[i] == input[i], "bitmap outlier i=%u: got %u want %u", i, out[i], input[i]);
    }
}

// ---------------------------------------------------------------------------
// Test: Edge cases (all zeros, all same value, all max)
// ---------------------------------------------------------------------------
static void testEdgeCases()
{
    printf("  Edge cases through scalar decode...\n");
    constexpr unsigned N = 128;

    // All zeros
    {
        std::vector<uint32_t> input(N, 0);
        std::vector<uint8_t> buf(N * 5);
        std::vector<uint32_t> out(N, 0xDEADBEEF);

        abpfor::b128::encode(input.data(), N, buf.data());
        scalar_block::decode<128, false>(
            buf.data(), N, out.data(), uint32_t(0));
        for (unsigned i = 0; i < N; ++i)
            CHECK(out[i] == 0, "all-zero i=%u: got %u", i, out[i]);
    }

    // All same non-zero value
    {
        std::vector<uint32_t> input(N, 42);
        std::vector<uint8_t> buf(N * 5);
        std::vector<uint32_t> out(N, 0);

        abpfor::b128::encode(input.data(), N, buf.data());
        scalar_block::decode<128, false>(
            buf.data(), N, out.data(), uint32_t(0));
        for (unsigned i = 0; i < N; ++i)
            CHECK(out[i] == 42, "all-42 i=%u: got %u", i, out[i]);
    }

    // Strictly increasing (delta → all-zero gaps)
    {
        std::vector<uint32_t> input(N);
        for (unsigned i = 0; i < N; ++i) input[i] = i + 1;
        std::vector<uint8_t> buf(N * 5);
        std::vector<uint32_t> out(N, 0);

        abpfor::b128::encodeDelta1(input.data(), N, buf.data(), uint32_t(0));
        scalar_block::decode<128, true>(
            buf.data(), N, out.data(), uint32_t(0));
        for (unsigned i = 0; i < N; ++i)
            CHECK(out[i] == input[i], "sequential delta i=%u: got %u want %u", i, out[i], input[i]);
    }
}

// ---------------------------------------------------------------------------
int main()
{
    printf("Scalar fallback full-path tests:\n");
    testI4NoDelta();
    testI4Delta();
    testI8NoDelta();
    testI8Delta();
    testOutlierPaths();
    testEdgeCases();

    if (failures == 0)
        printf("ALL PASSED\n");
    else
        printf("FAILURES: %d\n", failures);
    return failures ? 1 : 0;
}
