// Layer 5 tests: SSE fused decode (unpack + patch + prefixSum)
//
// Verifies that the fused single-pass decode produces identical results
// to the separate steps (unpackI4 → scalar merge → scalar undelta).

#include "simd/sse_fused.h"
#include "simd/sse_pack.h"
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

// --- Helper: prepare packed data + bitmap + residuals for a given block ---

struct TestBlock {
    alignas(16) uint32_t values[128];    // original values (before delta)
    alignas(16) uint8_t packed[1024];    // interleaved-packed base values (b bits each)
    alignas(16) uint64_t bitmap[2];      // outlier bitmap
    alignas(16) uint32_t residuals[128]; // unpacked outlier residuals (contiguous)
    unsigned b;                          // base bit-width
    unsigned pb;                         // outlier residual bits
    unsigned outlierCount;
};

static void makeBlock(TestBlock& blk, unsigned b, const uint32_t* data, unsigned n = 128)
{
    blk.b = b;
    std::memcpy(blk.values, data, n * sizeof(uint32_t));

    // Find max bits and separate base/outlier
    unsigned maxBits = 0;
    for (unsigned i = 0; i < n; ++i)
    {
        unsigned bw = abpfor::bitwidth(data[i]);
        if (bw > maxBits) maxBits = bw;
    }
    blk.pb = (maxBits > b) ? maxBits - b : 0;

    // Build bitmap and residuals
    blk.bitmap[0] = blk.bitmap[1] = 0;
    blk.outlierCount = 0;
    uint32_t baseMask = abpfor::mask<uint32_t>(b);
    alignas(16) uint32_t masked[128];

    for (unsigned i = 0; i < n; ++i)
    {
        masked[i] = data[i] & baseMask;
        if (abpfor::bitwidth(data[i]) > b)
        {
            blk.bitmap[i / 64] |= uint64_t(1) << (i % 64);
            blk.residuals[blk.outlierCount++] = data[i] >> b;
        }
    }

    // Pack base values in interleaved format
    abpfor::packI4(masked, blk.packed, b);
}

// --- Test 1: fused unpack + delta (no outliers) ---

static void test_fused_delta_only()
{
    printf("test_fused_delta_only...\n");
    std::mt19937 rng(42);

    for (unsigned b : {1u, 4u, 8u, 12u, 16u, 24u, 32u})
    {
        uint32_t m = abpfor::mask<uint32_t>(b);
        uint32_t data[128];
        for (auto& v : data) v = static_cast<uint32_t>(rng() & m);

        // Reference: unpack then scalar undelta
        alignas(16) uint32_t ref[128];
        alignas(16) uint8_t packed[1024];
        abpfor::packI4(data, packed, b);
        abpfor::unpackI4(packed, ref, b);
        abpfor::undelta(ref, 128, uint32_t(0));

        // Fused
        alignas(16) uint32_t fused[128];
        uint32_t carry = 0;
        abpfor::fusedDecodeI4<true, false, true>(packed, fused, b, carry, nullptr, nullptr);

        for (unsigned i = 0; i < 128; ++i)
        {
            CHECK(fused[i] == ref[i],
                  "d1 b=%u i=%u: fused=%u ref=%u", b, i, fused[i], ref[i]);
            if (fused[i] != ref[i]) break;
        }
    }
}

// --- Test 2: fused unpack + patch (no delta) ---

static void test_fused_patch_only()
{
    printf("test_fused_patch_only...\n");
    std::mt19937 rng(7);

    uint32_t data[128];
    for (auto& v : data) v = rng() & 0xF;
    // Add outliers
    data[5] = 0xABCD;
    data[30] = 0x1234;
    data[90] = 0xFFFF;
    data[100] = 0x5678;

    TestBlock blk;
    makeBlock(blk, 4, data);

    // Reference: separate unpack + merge
    alignas(16) uint32_t ref[128];
    abpfor::unpackI4(blk.packed, ref, blk.b);
    unsigned ri = 0;
    for (unsigned i = 0; i < 128; ++i)
    {
        if (blk.bitmap[i / 64] & (uint64_t(1) << (i % 64)))
            ref[i] |= blk.residuals[ri++] << blk.b;
    }

    // Fused
    alignas(16) uint32_t fused[128];
    uint32_t carry = 0;
    abpfor::fusedDecodeI4<false, true, true>(blk.packed, fused, blk.b, carry,
                                       blk.bitmap, blk.residuals);

    for (unsigned i = 0; i < 128; ++i)
    {
        CHECK(fused[i] == ref[i],
              "patch b=%u i=%u: fused=%u ref=%u", blk.b, i, fused[i], ref[i]);
        if (fused[i] != ref[i]) break;
    }
}

// --- Test 3: full fused (unpack + patch + delta) ---

static void test_fused_full()
{
    printf("test_fused_full...\n");
    std::mt19937 rng(99);

    for (unsigned excPct : {3u, 10u, 25u})
    {
        for (unsigned baseBw : {4u, 8u, 12u, 16u})
        {
            uint32_t data[128];
            uint32_t baseMask = abpfor::mask<uint32_t>(baseBw);
            for (unsigned i = 0; i < 128; ++i)
            {
                if ((rng() % 100) < excPct)
                    data[i] = (1u << baseBw) + (rng() & 0xFFFF);
                else
                    data[i] = static_cast<uint32_t>(rng() & baseMask);
            }

            TestBlock blk;
            makeBlock(blk, baseBw, data);

            // Reference: unpack + merge + undelta
            alignas(16) uint32_t ref[128];
            abpfor::unpackI4(blk.packed, ref, blk.b);
            unsigned ri = 0;
            for (unsigned i = 0; i < 128; ++i)
            {
                if (blk.bitmap[i / 64] & (uint64_t(1) << (i % 64)))
                    ref[i] |= blk.residuals[ri++] << blk.b;
            }
            abpfor::undelta(ref, 128, uint32_t(0));

            // Fused
            alignas(16) uint32_t fused[128];
            uint32_t carry = 0;
            abpfor::fusedDecodeI4<true, true, true>(blk.packed, fused, blk.b, carry,
                                               blk.bitmap, blk.residuals);

            for (unsigned i = 0; i < 128; ++i)
            {
                CHECK(fused[i] == ref[i],
                      "full b=%u exc=%u%% i=%u: fused=%u ref=%u",
                      baseBw, excPct, i, fused[i], ref[i]);
                if (fused[i] != ref[i]) break;
            }

            // Verify carry equals last element
            CHECK(carry == ref[127],
                  "carry=%u ref[127]=%u", carry, ref[127]);
        }
    }
}

// --- Test 4: carry propagation across blocks ---

static void test_carry_chain()
{
    printf("test_carry_chain...\n");
    std::mt19937 rng(55);

    // Two blocks, carry from first feeds into second
    uint32_t data1[128], data2[128];
    for (auto& v : data1) v = rng() & 0xF;
    for (auto& v : data2) v = rng() & 0xF;

    alignas(16) uint8_t p1[1024], p2[1024];
    abpfor::packI4(data1, p1, 4);
    abpfor::packI4(data2, p2, 4);

    // Reference
    alignas(16) uint32_t ref1[128], ref2[128];
    abpfor::unpackI4(p1, ref1, 4);
    abpfor::unpackI4(p2, ref2, 4);
    abpfor::undelta(ref1, 128, uint32_t(0));
    abpfor::undelta(ref2, 128, ref1[127]);

    // Fused
    alignas(16) uint32_t f1[128], f2[128];
    uint32_t carry = 0;
    abpfor::fusedDecodeI4<true, false, true>(p1, f1, 4, carry, nullptr, nullptr);
    abpfor::fusedDecodeI4<true, false, true>(p2, f2, 4, carry, nullptr, nullptr);

    for (unsigned i = 0; i < 128; ++i)
        CHECK(f1[i] == ref1[i], "chain blk1 i=%u: %u != %u", i, f1[i], ref1[i]);
    for (unsigned i = 0; i < 128; ++i)
        CHECK(f2[i] == ref2[i], "chain blk2 i=%u: %u != %u", i, f2[i], ref2[i]);
}

int main()
{
    test_fused_delta_only();
    test_fused_patch_only();
    test_fused_full();
    test_carry_chain();

    if (failures == 0)
        printf("All Layer 5 fused decode tests passed.\n");
    else
        printf("%d Layer 5 fused decode test(s) FAILED.\n", failures);

    return failures > 0 ? 1 : 0;
}
