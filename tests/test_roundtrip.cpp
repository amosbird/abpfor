// Layer 3 tests: block encode/decode roundtrip
//
// Tests the full scalar pipeline: encode → decode with the new wire format.
// Covers all block types: bitpack-only, bitmap outliers, sparse outliers,
// constant, all-zeros, raw.  Also tests delta-1 encode/decode.

#include "core/block.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <numeric>
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

template <typename T>
static void roundtrip(const T* orig, unsigned n, const char* label, T start = 0, bool useDelta = false)
{
    uint8_t buf[4096];
    T decoded[512];

    std::memset(buf, 0xCC, sizeof(buf));
    std::memset(decoded, 0xDD, sizeof(decoded));

    size_t written;
    if (useDelta)
        written = abpfor::encodeBlockDelta1(orig, n, buf, start);
    else
        written = abpfor::encodeBlock(orig, n, buf);

    CHECK(written > 0 && written < sizeof(buf), "%s: written=%zu", label, written);

    size_t consumed;
    if (useDelta)
        consumed = abpfor::decodeBlockDelta1(buf, n, decoded, start);
    else
        consumed = abpfor::decodeBlock(buf, n, decoded);

    CHECK(consumed == written, "%s: consumed=%zu != written=%zu", label, consumed, written);

    for (unsigned i = 0; i < n; ++i)
    {
        if (decoded[i] != orig[i])
        {
            CHECK(false, "%s n=%u i=%u: decoded=%llu expected=%llu",
                  label, n, i, static_cast<unsigned long long>(decoded[i]), static_cast<unsigned long long>(orig[i]));
            break;
        }
    }
}

// --- All zeros ---

static void test_zeros()
{
    printf("test_zeros...\n");
    uint32_t data[128] = {};
    roundtrip(data, 128, "zeros-128");

    uint32_t data2[256] = {};
    roundtrip(data2, 256, "zeros-256");
}

// --- Constant ---

static void test_constant()
{
    printf("test_constant...\n");
    uint32_t data[128];
    std::fill_n(data, 128, 42u);
    roundtrip(data, 128, "const-42");

    uint32_t data2[128];
    std::fill_n(data2, 128, 0xDEADBEEFu);
    roundtrip(data2, 128, "const-large");
}

// --- Bitpack only (no outliers) ---

static void test_bitpack_only()
{
    printf("test_bitpack_only...\n");
    std::mt19937 rng(1);

    for (unsigned bw : {1u, 2u, 4u, 8u, 12u, 16u, 20u, 24u, 28u, 32u})
    {
        uint32_t data[128];
        uint32_t m = abpfor::mask<uint32_t>(bw);
        for (auto& v : data) v = static_cast<uint32_t>(rng() & m);

        char label[64];
        snprintf(label, sizeof(label), "bitpack-b%u", bw);
        roundtrip(data, 128, label);
    }
}

// --- Sparse outliers (few exceptions) ---

static void test_sparse_outliers()
{
    printf("test_sparse_outliers...\n");
    uint32_t data[128];
    for (unsigned i = 0; i < 128; ++i) data[i] = i & 0xF;
    data[10] = 0xABCD;
    data[50] = 0x1234;
    data[90] = 0xFFFF;
    roundtrip(data, 128, "sparse-3exc");
}

// --- Bitmap outliers (many exceptions) ---

static void test_bitmap_outliers()
{
    printf("test_bitmap_outliers...\n");
    std::mt19937 rng(42);
    uint32_t data[128];
    for (auto& v : data) v = rng() & 0xF;
    for (unsigned i = 0; i < 25; ++i) data[i * 5] = 0xFFFF;
    roundtrip(data, 128, "bitmap-25exc");
}

// --- Delta-1 roundtrip ---

static void test_delta()
{
    printf("test_delta...\n");
    std::mt19937 rng(7);

    // Sorted sequence
    uint32_t data[128];
    uint32_t v = 100;
    for (unsigned i = 0; i < 128; ++i)
    {
        v += 1 + static_cast<uint32_t>(rng() % 10);
        data[i] = v;
    }
    roundtrip(data, 128, "delta-sorted", uint32_t(100), true);

    // Tighter gaps
    v = 0;
    for (unsigned i = 0; i < 128; ++i)
    {
        v += 1 + static_cast<uint32_t>(rng() % 3);
        data[i] = v;
    }
    roundtrip(data, 128, "delta-tight", uint32_t(0), true);

    // Sorted with occasional large gap
    v = 0;
    for (unsigned i = 0; i < 128; ++i)
    {
        v += (rng() % 100 < 5) ? static_cast<uint32_t>(1000 + rng() % 5000) : static_cast<uint32_t>(1 + rng() % 5);
        data[i] = v;
    }
    roundtrip(data, 128, "delta-sparse-gaps", uint32_t(0), true);
}

// --- Various block sizes ---

static void test_various_n()
{
    printf("test_various_n...\n");
    std::mt19937 rng(99);

    for (unsigned n : {1u, 2u, 7u, 15u, 16u, 31u, 32u, 63u, 64u, 100u, 127u, 128u, 200u, 255u, 256u})
    {
        uint32_t data[256];
        for (unsigned i = 0; i < n; ++i) data[i] = rng() & 0xFF;

        char label[64];
        snprintf(label, sizeof(label), "n=%u", n);
        roundtrip(data, n, label);
    }
}

// --- 64-bit ---

static void test_64bit()
{
    printf("test_64bit...\n");
    std::mt19937_64 rng(123);

    // Small values
    {
        uint64_t data[128];
        for (auto& v : data) v = rng() & 0xFFFF;
        roundtrip(data, 128, "u64-16bit");
    }

    // Mix with large outliers
    {
        uint64_t data[128];
        for (auto& v : data) v = rng() & 0xFFFF;
        data[0] = 1ULL << 50;
        data[64] = 1ULL << 60;
        roundtrip(data, 128, "u64-outliers");
    }

    // 64-bit delta
    {
        uint64_t data[128];
        uint64_t v = 1000000;
        for (unsigned i = 0; i < 128; ++i)
        {
            v += 1 + (rng() % 100);
            data[i] = v;
        }
        roundtrip(data, 128, "u64-delta", uint64_t(1000000), true);
    }
}

// --- Sweep: all bit-widths × exception rates ---

static void test_sweep()
{
    printf("test_sweep...\n");
    std::mt19937 rng(42);
    uint32_t data[128];

    for (unsigned baseBw : {2u, 4u, 8u, 12u, 16u, 24u})
    {
        for (unsigned excPct : {0u, 3u, 10u, 25u, 50u})
        {
            uint32_t baseMask = abpfor::mask<uint32_t>(baseBw);

            for (unsigned i = 0; i < 128; ++i)
            {
                if (excPct > 0 && (rng() % 100) < excPct)
                    data[i] = (1u << baseBw) + (rng() & 0xFFFF);
                else
                    data[i] = static_cast<uint32_t>(rng() & baseMask);
            }

            char label[64];
            snprintf(label, sizeof(label), "sweep-b%u-exc%u%%", baseBw, excPct);
            roundtrip(data, 128, label);
        }
    }
}

// --- Raw block via delta1 (regression: kRaw header 0xFF must not be
//     mistaken for a constant block) ---
static void test_raw_delta()
{
    printf("test_raw_delta...\n");

    // uint64_t sorted ascending with a 64-bit stride forces encodeBlock to
    // pick the kRaw path (constant value has bitwidth 64 >= kRaw=63).
    {
        uint64_t data[16];
        const uint64_t stride = (1ULL << 63) + 1;
        uint64_t v = 5;
        for (unsigned i = 0; i < 16; ++i) { v += stride; data[i] = v; }
        roundtrip(data, 16, "raw-delta-u64", uint64_t(5), true);
    }

    // uint32_t: a stride whose delta bitwidth is 32 (>= kRaw=63 triggers raw).
    {
        uint32_t data[16];
        const uint32_t stride = (1u << 31) + 1;
        uint32_t v = 3;
        for (unsigned i = 0; i < 16; ++i) { v += stride; data[i] = v; }
        roundtrip(data, 16, "raw-delta-u32", uint32_t(3), true);
    }

    // Non-delta raw block (bitpack-only with b >= W) roundtrips too.
    {
        uint32_t data[128];
        std::mt19937 rng(2024);
        // mt19937::result_type is uint_fast32_t (64-bit here), hence the cast.
        for (auto& x : data) x = static_cast<uint32_t>(rng());   // full-width -> kRaw
        roundtrip(data, 128, "raw-nodelta-u32");
    }
}

int main()
{
    test_zeros();
    test_constant();
    test_bitpack_only();
    test_sparse_outliers();
    test_bitmap_outliers();
    test_delta();
    test_various_n();
    test_64bit();
    test_sweep();
    test_raw_delta();

    if (failures == 0)
        printf("All Layer 3 tests passed.\n");
    else
        printf("%d Layer 3 test(s) FAILED.\n", failures);

    return failures > 0 ? 1 : 0;
}
