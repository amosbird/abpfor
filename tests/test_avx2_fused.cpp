// Layer 5 tests: AVX2 fused decode (unpack + patch + prefixSum, 256 elements)

#include "simd/avx2_fused.h"
#include "simd/avx2_pack.h"
#include "core/delta.h"
#include "core/bits.h"

#include <cstdio>
#include <cstring>
#include <random>

static int failures = 0;

#define CHECK(cond, ...)                                    \
    do {                                                    \
        if (!(cond)) {                                      \
            printf("  FAIL %s:%d: ", __FILE__, __LINE__);   \
            printf(__VA_ARGS__);                             \
            printf("\n");                                    \
            ++failures;                                     \
        }                                                   \
    } while (0)

struct TestBlock8 {
    alignas(32) uint32_t values[256];
    alignas(32) uint8_t packed[2048];
    alignas(32) uint64_t bitmap[4]; // 256 bits
    alignas(32) uint32_t residuals[256];
    unsigned b, pb, outlierCount;
};

static void makeBlock8(TestBlock8& blk, unsigned b, const uint32_t* data)
{
    blk.b = b;
    std::memcpy(blk.values, data, 256 * sizeof(uint32_t));

    unsigned maxBits = 0;
    for (unsigned i = 0; i < 256; ++i)
    {
        unsigned bw = abpfor::bitwidth(data[i]);
        if (bw > maxBits) maxBits = bw;
    }
    blk.pb = (maxBits > b) ? maxBits - b : 0;

    std::memset(blk.bitmap, 0, sizeof(blk.bitmap));
    blk.outlierCount = 0;
    uint32_t baseMask = abpfor::mask<uint32_t>(b);
    alignas(32) uint32_t masked[256];

    for (unsigned i = 0; i < 256; ++i)
    {
        masked[i] = data[i] & baseMask;
        if (abpfor::bitwidth(data[i]) > b)
        {
            blk.bitmap[i / 64] |= uint64_t(1) << (i % 64);
            blk.residuals[blk.outlierCount++] = data[i] >> b;
        }
    }
    abpfor::packI8(masked, blk.packed, b);
}

// --- delta only ---

static void test_fused8_delta_only()
{
    printf("test_fused8_delta_only...\n");
    std::mt19937 rng(42);

    for (unsigned b : {1u, 4u, 8u, 12u, 16u, 24u, 32u})
    {
        uint32_t m = abpfor::mask<uint32_t>(b);
        uint32_t data[256];
        for (auto& v : data) v = rng() & m;

        alignas(32) uint32_t ref[256];
        alignas(32) uint8_t packed[2048];
        abpfor::packI8(data, packed, b);
        abpfor::unpackI8(packed, ref, b);
        abpfor::undelta(ref, 256, uint32_t(0));

        alignas(32) uint32_t fused[256];
        uint32_t carry = 0;
        abpfor::fusedDecodeI8<true, false, true>(packed, fused, b, carry, nullptr, nullptr);

        for (unsigned i = 0; i < 256; ++i)
        {
            CHECK(fused[i] == ref[i], "d1 b=%u i=%u: %u != %u", b, i, fused[i], ref[i]);
            if (fused[i] != ref[i]) break;
        }
    }
}

// --- patch only ---

static void test_fused8_patch_only()
{
    printf("test_fused8_patch_only...\n");
    std::mt19937 rng(7);

    uint32_t data[256];
    for (auto& v : data) v = rng() & 0xF;
    data[5] = 0xABCD;
    data[30] = 0x1234;
    data[90] = 0xFFFF;
    data[100] = 0x5678;
    data[200] = 0x9ABC;

    TestBlock8 blk;
    makeBlock8(blk, 4, data);

    alignas(32) uint32_t ref[256];
    abpfor::unpackI8(blk.packed, ref, blk.b);
    unsigned ri = 0;
    for (unsigned i = 0; i < 256; ++i)
    {
        if (blk.bitmap[i / 64] & (uint64_t(1) << (i % 64)))
            ref[i] |= blk.residuals[ri++] << blk.b;
    }

    alignas(32) uint32_t fused[256];
    uint32_t carry = 0;
    abpfor::fusedDecodeI8<false, true, true>(blk.packed, fused, blk.b, carry,
                                       blk.bitmap, blk.residuals);

    for (unsigned i = 0; i < 256; ++i)
    {
        CHECK(fused[i] == ref[i], "patch i=%u: %u != %u", i, fused[i], ref[i]);
        if (fused[i] != ref[i]) break;
    }
}

// --- full fused ---

static void test_fused8_full()
{
    printf("test_fused8_full...\n");
    std::mt19937 rng(99);

    for (unsigned excPct : {3u, 10u, 25u})
    {
        for (unsigned baseBw : {4u, 8u, 12u, 16u})
        {
            uint32_t data[256];
            uint32_t baseMask = abpfor::mask<uint32_t>(baseBw);
            for (unsigned i = 0; i < 256; ++i)
            {
                if ((rng() % 100) < excPct)
                    data[i] = (1u << baseBw) + (rng() & 0xFFFF);
                else
                    data[i] = rng() & baseMask;
            }

            TestBlock8 blk;
            makeBlock8(blk, baseBw, data);

            alignas(32) uint32_t ref[256];
            abpfor::unpackI8(blk.packed, ref, blk.b);
            unsigned ri = 0;
            for (unsigned i = 0; i < 256; ++i)
            {
                if (blk.bitmap[i / 64] & (uint64_t(1) << (i % 64)))
                    ref[i] |= blk.residuals[ri++] << blk.b;
            }
            abpfor::undelta(ref, 256, uint32_t(0));

            alignas(32) uint32_t fused[256];
            uint32_t carry = 0;
            abpfor::fusedDecodeI8<true, true, true>(blk.packed, fused, blk.b, carry,
                                               blk.bitmap, blk.residuals);

            for (unsigned i = 0; i < 256; ++i)
            {
                CHECK(fused[i] == ref[i],
                      "full b=%u exc=%u%% i=%u: %u != %u",
                      baseBw, excPct, i, fused[i], ref[i]);
                if (fused[i] != ref[i]) break;
            }
            CHECK(carry == ref[255], "carry=%u ref[255]=%u", carry, ref[255]);
        }
    }
}

// --- carry chain ---

static void test_fused8_carry()
{
    printf("test_fused8_carry...\n");
    std::mt19937 rng(55);

    uint32_t d1[256], d2[256];
    for (auto& v : d1) v = rng() & 0xF;
    for (auto& v : d2) v = rng() & 0xF;

    alignas(32) uint8_t p1[2048], p2[2048];
    abpfor::packI8(d1, p1, 4);
    abpfor::packI8(d2, p2, 4);

    alignas(32) uint32_t ref1[256], ref2[256];
    abpfor::unpackI8(p1, ref1, 4);
    abpfor::unpackI8(p2, ref2, 4);
    abpfor::undelta(ref1, 256, uint32_t(0));
    abpfor::undelta(ref2, 256, ref1[255]);

    alignas(32) uint32_t f1[256], f2[256];
    uint32_t carry = 0;
    abpfor::fusedDecodeI8<true, false, true>(p1, f1, 4, carry, nullptr, nullptr);
    abpfor::fusedDecodeI8<true, false, true>(p2, f2, 4, carry, nullptr, nullptr);

    for (unsigned i = 0; i < 256; ++i)
        CHECK(f1[i] == ref1[i], "chain b1 i=%u: %u != %u", i, f1[i], ref1[i]);
    for (unsigned i = 0; i < 256; ++i)
        CHECK(f2[i] == ref2[i], "chain b2 i=%u: %u != %u", i, f2[i], ref2[i]);
}

// The carry (`sv`) is the loop-carried value in the prefix sum, and every test
// above drives it with MinusOne=true only. The two settings update the carry
// differently, so MinusOne=false chained across blocks is its own case: an
// error in how the per-lane offset folds into the carry is invisible at
// MinusOne=true and invisible within a single block.
static void test_fused8_carry_chain_minusone_false()
{
    printf("test_fused8_carry_chain_minusone_false...\n");
    std::mt19937 rng(97);

    for (unsigned b : {1u, 4u, 8u, 12u, 16u, 24u, 32u})
    {
        const uint32_t m = abpfor::mask<uint32_t>(b);

        // Four consecutive blocks sharing one carry: a carry that is wrong by
        // a constant shows up as a growing error, not a single bad element.
        uint32_t data[4][256];
        for (auto& blk : data)
            for (auto& v : blk) v = rng() & m;

        alignas(32) uint32_t ref[4][256];
        alignas(32) uint8_t packed[4][2048];
        uint32_t refCarry = 0;
        for (unsigned k = 0; k < 4; ++k)
        {
            abpfor::packI8(data[k], packed[k], b);
            abpfor::unpackI8(packed[k], ref[k], b);
            // MinusOne=false is a plain prefix sum: out[i] = in[i] + prev.
            // (abpfor::undelta is the +1 variant and is the wrong model here.)
            for (unsigned i = 0; i < 256; ++i)
            {
                refCarry += ref[k][i];
                ref[k][i] = refCarry;
            }
        }

        alignas(32) uint32_t fused[4][256];
        uint32_t carry = 0;
        for (unsigned k = 0; k < 4; ++k)
            abpfor::fusedDecodeI8<true, false, false>(packed[k], fused[k], b, carry,
                                                     nullptr, nullptr);

        for (unsigned k = 0; k < 4; ++k)
            for (unsigned i = 0; i < 256; ++i)
            {
                CHECK(fused[k][i] == ref[k][i], "m0 b=%u blk=%u i=%u: %u != %u",
                      b, k, i, fused[k][i], ref[k][i]);
                if (fused[k][i] != ref[k][i]) return;
            }

        CHECK(carry == refCarry, "m0 b=%u final carry: %u != %u", b, carry, refCarry);
    }
}

int main()
{
    test_fused8_delta_only();
    test_fused8_patch_only();
    test_fused8_full();
    test_fused8_carry();
    test_fused8_carry_chain_minusone_false();

    if (failures == 0)
        printf("All Layer 5 AVX2 fused tests passed.\n");
    else
        printf("%d Layer 5 AVX2 fused test(s) FAILED.\n", failures);

    return failures > 0 ? 1 : 0;
}
