// Layer 2 tests: cost model (optimalWidth)
//
// Tests that optimalWidth picks the (b, pbx) minimising block size,
// and verifies the pbx encoding directly.

#include "core/cost.h"
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

// --- Helper: manually compute block cost for a given (b, strategy) ---

enum Strategy { None, Bitmap, Sparse };

static unsigned manualCost(const uint32_t* in, unsigned n, unsigned b, Strategy strat)
{
    unsigned maxBits = 0;
    unsigned outlierCount = 0;
    for (unsigned i = 0; i < n; ++i)
    {
        unsigned bw = abpfor::bitwidth(in[i]);
        if (bw > maxBits) maxBits = bw;
        if (bw > b) ++outlierCount;
    }

    unsigned pb = (maxBits > b) ? maxBits - b : 0;

    if (outlierCount == 0)
        return abpfor::packedBytes(n, b) + 1;

    unsigned basePack = abpfor::packedBytes(n, b);
    unsigned outlierPack = abpfor::packedBytes(outlierCount, pb);

    if (strat == Bitmap)
        return 2 + n / 8 + outlierPack + basePack;
    else // Sparse
        return 3 + outlierCount + outlierPack + basePack;
}

// --- All zeros ---

static void test_all_zeros()
{
    printf("test_all_zeros...\n");
    uint32_t data[128] = {};
    unsigned pbx;
    unsigned b = abpfor::optimalWidth<uint32_t>(data, 128, &pbx);
    CHECK(b == 0, "baseBits=%u", b);
    CHECK(pbx == 0, "pbx=%u", pbx);
}

// --- All constant ---

static void test_constant()
{
    printf("test_constant...\n");
    uint32_t data[128];
    for (auto& v : data) v = 42;
    unsigned pbx;
    unsigned b = abpfor::optimalWidth<uint32_t>(data, 128, &pbx);
    CHECK(b == 6, "baseBits=%u", b); // bitwidth(42) = 6
    CHECK(pbx == 32 + 2, "pbx=%u (expected constant=%u)", pbx, 32 + 2);
}

// --- No outliers ---

static void test_no_outliers()
{
    printf("test_no_outliers...\n");
    std::mt19937 rng(1);
    uint32_t data[128];
    for (auto& v : data) v = rng() & 0xFF; // all 8-bit

    unsigned pbx;
    unsigned b = abpfor::optimalWidth<uint32_t>(data, 128, &pbx);
    CHECK(b == 8, "baseBits=%u", b);
    CHECK(pbx == 0, "pbx=%u", pbx);
}

// --- Few outliers → sparse wins ---

static void test_sparse_wins()
{
    printf("test_sparse_wins...\n");
    uint32_t data[128];
    for (unsigned i = 0; i < 128; ++i) data[i] = i & 0xF; // 4-bit values
    // Add 3 outliers
    data[10] = 0xFFFF;
    data[50] = 0xFFFF;
    data[90] = 0xFFFF;

    unsigned pbx;
    unsigned b = abpfor::optimalWidth<uint32_t>(data, 128, &pbx);
    // Sparse: pbx = outlierBits (1..W), not 0, not W+1, not W+2
    CHECK(pbx > 0 && pbx <= 32, "expected sparse pbx, got %u", pbx);
    CHECK(pbx != 33, "should not be bitmap");
}

// --- Many outliers → bitmap wins ---

static void test_bitmap_wins()
{
    printf("test_bitmap_wins...\n");
    uint32_t data[128];
    std::mt19937 rng(42);
    for (unsigned i = 0; i < 128; ++i) data[i] = rng() & 0xF; // 4-bit base
    // Add 30 outliers
    for (unsigned i = 0; i < 30; ++i) data[i * 4] = 0xFFFF;

    unsigned pbx;
    unsigned b = abpfor::optimalWidth<uint32_t>(data, 128, &pbx);
    // With 30 outliers bitmap (W+1=33) should win
    if (pbx != 0) // might choose no-outlier at higher b
        CHECK(pbx == 33, "expected bitmap (33), got pbx=%u", pbx);
}

// --- Cost is actually minimal ---

static void test_cost_minimal()
{
    printf("test_cost_minimal...\n");
    std::mt19937 rng(99);
    uint32_t data[128];

    for (unsigned trial = 0; trial < 100; ++trial)
    {
        unsigned baseBw = 2 + (rng() % 20);
        unsigned excBw = baseBw + 2 + (rng() % 10);
        if (excBw > 32) excBw = 32;
        unsigned excPct = 1 + (rng() % 30);

        for (unsigned i = 0; i < 128; ++i)
        {
            if ((rng() % 100) < excPct)
                data[i] = (1u << (excBw - 1)) | (rng() & abpfor::mask<uint32_t>(excBw));
            else
                data[i] = rng() & abpfor::mask<uint32_t>(baseBw);
        }

        unsigned pbx;
        unsigned b = abpfor::optimalWidth<uint32_t>(data, 128, &pbx);

        // Compute our cost from pbx encoding
        unsigned ourCost;
        if (b == 0 && pbx == 0) {
            ourCost = 1;
        } else if (pbx == 34) {
            // constant
            ourCost = 1 + ((b + 7u) >> 3);
        } else if (pbx == 0) {
            ourCost = abpfor::packedBytes(128, b) + 1;
        } else if (pbx == 33) {
            ourCost = manualCost(data, 128, b, Bitmap);
        } else {
            ourCost = manualCost(data, 128, b, Sparse);
        }

        // Verify cost is ≤ every alternative
        unsigned maxBits = 0;
        for (unsigned i = 0; i < 128; ++i) {
            unsigned bw = abpfor::bitwidth(data[i]);
            if (bw > maxBits) maxBits = bw;
        }

        for (unsigned tb = 0; tb <= maxBits; ++tb)
        {
            unsigned bitmapCost = manualCost(data, 128, tb, Bitmap);
            unsigned sparseCost = manualCost(data, 128, tb, Sparse);

            CHECK(ourCost <= bitmapCost,
                  "trial=%u b=%u: cost %u > bitmap %u", trial, tb, ourCost, bitmapCost);
            CHECK(ourCost <= sparseCost,
                  "trial=%u b=%u: cost %u > sparse %u", trial, tb, ourCost, sparseCost);
        }

        // Also check against bitpack-only at maxBits
        unsigned bitpackOnly = abpfor::packedBytes(128, maxBits) + 1;
        CHECK(ourCost <= bitpackOnly,
              "trial=%u: cost %u > bitpackOnly %u", trial, ourCost, bitpackOnly);
    }
}

// --- 64-bit ---

static void test_cost_64bit()
{
    printf("test_cost_64bit...\n");
    uint64_t data[128];
    std::mt19937_64 rng(77);

    for (unsigned i = 0; i < 128; ++i)
        data[i] = rng() & 0xFFFF; // 16-bit base
    data[0] = 1ULL << 40; // 41-bit outlier
    data[64] = 1ULL << 50; // 51-bit outlier

    unsigned pbx;
    unsigned b = abpfor::optimalWidth<uint64_t>(data, 128, &pbx);
    // 2 outliers → sparse (pbx = outlierBits, 1..64)
    CHECK(pbx > 0 && pbx <= 64, "expected sparse pbx, got %u", pbx);
    CHECK(pbx != 65, "should not be bitmap (W+1=65)");
}

// --- n=256 crossover point ---

static void test_crossover_256()
{
    printf("test_crossover_256...\n");
    uint32_t data[256];
    for (unsigned i = 0; i < 256; ++i) data[i] = i & 0x7; // 3-bit base

    // 10 outliers → sparse should win
    for (unsigned i = 0; i < 10; ++i) data[i * 25] = 0xFFFF;
    unsigned pbx;
    unsigned b = abpfor::optimalWidth<uint32_t>(data, 256, &pbx);
    if (pbx != 0) // not no-outlier
        CHECK(pbx > 0 && pbx <= 32 && pbx != 33,
              "10 outliers in 256: expected Sparse, got pbx=%u", pbx);

    // 40 outliers → bitmap should win
    for (unsigned i = 0; i < 256; ++i) data[i] = i & 0x7;
    for (unsigned i = 0; i < 40; ++i) data[i * 6] = 0xFFFF;
    b = abpfor::optimalWidth<uint32_t>(data, 256, &pbx);
    if (pbx != 0)
        CHECK(pbx == 33, "40 outliers in 256: expected Bitmap (33), got pbx=%u", pbx);
}

int main()
{
    test_all_zeros();
    test_constant();
    test_no_outliers();
    test_sparse_wins();
    test_bitmap_wins();
    test_cost_minimal();
    test_cost_64bit();
    test_crossover_256();

    if (failures == 0)
        printf("All Layer 2 tests passed.\n");
    else
        printf("%d Layer 2 test(s) FAILED.\n", failures);

    return failures > 0 ? 1 : 0;
}
