// Comprehensive functional test — pure roundtrip verification.
// Covers: all bit-widths, b128/b256, delta/non-delta, outliers, edge cases, stress.

#include <abpfor.h>
#include "detail/codec.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int failures = 0;

#define CHECK(cond, ...)                                  \
    do {                                                  \
        if (!(cond)) {                                    \
            printf("  FAIL %s:%d: ", __FILE__, __LINE__); \
            printf(__VA_ARGS__);                           \
            printf("\n");                                  \
            ++failures;                                   \
        }                                                 \
    } while (0)

// ---------------------------------------------------------------------------
// Data generators
// ---------------------------------------------------------------------------

template <typename T>
static void makeRandom(T* data, unsigned n, unsigned bw, uint32_t seed)
{
    std::mt19937_64 rng(seed);
    T m = (bw >= sizeof(T)*8) ? ~T(0) : ((T(1) << bw) - 1);
    for (unsigned i = 0; i < n; ++i)
        data[i] = static_cast<T>(rng()) & m;
}

template <typename T>
static void makeRandomWithOutliers(T* data, unsigned n, unsigned baseBw, unsigned excPct, uint32_t seed)
{
    std::mt19937_64 rng(seed);
    T baseMask = (baseBw >= sizeof(T)*8) ? ~T(0) : ((T(1) << baseBw) - 1);
    for (unsigned i = 0; i < n; ++i)
    {
        if (excPct > 0 && (rng() % 100) < excPct)
            data[i] = static_cast<T>(rng());
        else
            data[i] = static_cast<T>(rng()) & baseMask;
    }
}

template <typename T>
static void makeSorted(T* data, unsigned n, T maxDelta, uint32_t seed)
{
    std::mt19937_64 rng(seed);
    T v = 0;
    for (unsigned i = 0; i < n; ++i)
    {
        v += T(1) + static_cast<T>(rng() % static_cast<uint64_t>(maxDelta));
        data[i] = v;
    }
}

// ---------------------------------------------------------------------------
// Generic roundtrip helpers
// ---------------------------------------------------------------------------

template <typename T>
using EncFn = size_t(*)(const T*, unsigned, uint8_t*);
template <typename T>
using DecFn = size_t(*)(const uint8_t*, unsigned, T*);
template <typename T>
using EncDFn = size_t(*)(const T*, unsigned, uint8_t*, T);
template <typename T>
using DecDFn = size_t(*)(const uint8_t*, unsigned, T*, T);

template <typename T>
static void rt(EncFn<T> enc, DecFn<T> dec, const T* data, unsigned n, const char* label)
{
    std::vector<uint8_t> buf(abpfor::maxCompressedSize<T>(n));
    std::vector<T> out(n);
    size_t written = enc(data, n, buf.data());
    CHECK(written > 0 && written <= buf.size(), "%s: written=%zu", label, written);
    size_t consumed = dec(buf.data(), n, out.data());
    CHECK(consumed == written, "%s: consumed=%zu != written=%zu", label, consumed, written);
    for (unsigned i = 0; i < n; ++i)
    {
        if (out[i] != data[i])
        {
            CHECK(false, "%s i=%u: got=%llu expected=%llu",
                  label, i, (unsigned long long)out[i], (unsigned long long)data[i]);
            break;
        }
    }
}

template <typename T>
static void rtd(EncDFn<T> enc, DecDFn<T> dec, const T* data, unsigned n, const char* label, T start)
{
    std::vector<uint8_t> buf(abpfor::maxCompressedSize<T>(n));
    std::vector<T> out(n);
    size_t written = enc(data, n, buf.data(), start);
    CHECK(written > 0 && written <= buf.size(), "%s: written=%zu", label, written);
    size_t consumed = dec(buf.data(), n, out.data(), start);
    CHECK(consumed == written, "%s: consumed=%zu != written=%zu", label, consumed, written);
    for (unsigned i = 0; i < n; ++i)
    {
        if (out[i] != data[i])
        {
            CHECK(false, "%s i=%u: got=%llu expected=%llu",
                  label, i, (unsigned long long)out[i], (unsigned long long)data[i]);
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// 1. b128 — all bit-widths
// ---------------------------------------------------------------------------

static void test_b128_all_bitwidths()
{
    printf("test_b128_all_bitwidths...\n");

    constexpr unsigned sizes[] = {1, 7, 31, 32, 63, 64, 100, 127, 128, 200, 255, 256};

    for (unsigned bw = 0; bw <= 32; ++bw)
    {
        for (unsigned n : sizes)
        {
            uint32_t data[256];
            makeRandom<uint32_t>(data, n, bw, 42 + bw * 1000 + n);
            char label[64];
            snprintf(label, sizeof(label), "b128-u32-b%u-n%u", bw, n);
            rt<uint32_t>(abpfor::b128::encode, abpfor::b128::decode, data, n, label);
        }
    }

    for (unsigned bw = 0; bw <= 64; ++bw)
    {
        for (unsigned n : sizes)
        {
            uint64_t data[256];
            makeRandom<uint64_t>(data, n, bw, 77 + bw * 1000 + n);
            char label[64];
            snprintf(label, sizeof(label), "b128-u64-b%u-n%u", bw, n);
            rt<uint64_t>(abpfor::b128::encode, abpfor::b128::decode, data, n, label);
        }
    }
}

// ---------------------------------------------------------------------------
// 2. b128 — delta
// ---------------------------------------------------------------------------

static void test_b128_delta()
{
    printf("test_b128_delta...\n");

    constexpr unsigned sizes[] = {1, 32, 127, 128, 200};
    constexpr uint32_t maxDeltas32[] = {1, 3, 15, 255, 65535};

    for (uint32_t md : maxDeltas32)
    {
        for (unsigned n : sizes)
        {
            uint32_t data[256];
            makeSorted<uint32_t>(data, n, md, 42 + md + n);
            char label[64];
            snprintf(label, sizeof(label), "b128-d1-u32-md%u-n%u", md, n);
            rtd<uint32_t>(abpfor::b128::encodeDelta1, abpfor::b128::decodeDelta1, data, n, label, uint32_t(0));
        }
    }

    // With non-zero start
    {
        uint32_t data[128];
        makeSorted<uint32_t>(data, 128, 10, 100);
        for (unsigned i = 0; i < 128; ++i) data[i] += 1000;
        rtd<uint32_t>(abpfor::b128::encodeDelta1, abpfor::b128::decodeDelta1, data, 128, "b128-d1-start1000", uint32_t(999));
    }

    // 64-bit delta
    {
        uint64_t data[200];
        makeSorted<uint64_t>(data, 200, 100, 300);
        for (unsigned i = 0; i < 200; ++i) data[i] += 1000000ULL;
        rtd<uint64_t>(abpfor::b128::encodeDelta1, abpfor::b128::decodeDelta1, data, 200, "b128-d1-u64", uint64_t(999999));
    }
}

// ---------------------------------------------------------------------------
// 3. b128 — outliers
// ---------------------------------------------------------------------------

static void test_b128_outliers()
{
    printf("test_b128_outliers...\n");

    constexpr unsigned excPcts[] = {5, 10, 25, 50};
    constexpr unsigned baseBws[] = {4, 8, 12, 16};

    for (unsigned excPct : excPcts)
    {
        for (unsigned baseBw : baseBws)
        {
            uint32_t data[128];
            makeRandomWithOutliers<uint32_t>(data, 128, baseBw, excPct, 42 + excPct * 100 + baseBw);
            char label[64];
            snprintf(label, sizeof(label), "b128-out-b%u-exc%u%%", baseBw, excPct);
            rt<uint32_t>(abpfor::b128::encode, abpfor::b128::decode, data, 128, label);
        }
    }

    // Delta with outlier-producing gaps
    for (unsigned excPct : excPcts)
    {
        uint32_t data[128];
        std::mt19937 rng(42 + excPct);
        uint32_t v = 0;
        for (unsigned i = 0; i < 128; ++i)
        {
            if ((rng() % 100) < excPct)
                v += 1000 + rng() % 50000;
            else
                v += 1 + rng() % 5;
            data[i] = v;
        }
        char label[64];
        snprintf(label, sizeof(label), "b128-d1-out-exc%u%%", excPct);
        rtd<uint32_t>(abpfor::b128::encodeDelta1, abpfor::b128::decodeDelta1, data, 128, label, uint32_t(0));
    }
}

// ---------------------------------------------------------------------------
// 4. b256 — all
// ---------------------------------------------------------------------------

static void test_b256_all()
{
    printf("test_b256_all...\n");

    for (unsigned bw = 1; bw <= 32; ++bw)
    {
        uint32_t data[256];
        makeRandom<uint32_t>(data, 256, bw, 42 + bw);
        char label[64];
        snprintf(label, sizeof(label), "b256-b%u-n256", bw);
        rt<uint32_t>(abpfor::b256::encode, abpfor::b256::decode, data, 256, label);
    }

    // n=600 (tail handling)
    for (unsigned bw : {4, 8, 16, 24, 32})
    {
        uint32_t data[600];
        makeRandom<uint32_t>(data, 600, bw, 77 + bw);
        char label[64];
        snprintf(label, sizeof(label), "b256-b%u-n600", bw);
        rt<uint32_t>(abpfor::b256::encode, abpfor::b256::decode, data, 600, label);
    }

    // Delta
    for (unsigned n : {256u, 500u, 600u})
    {
        uint32_t data[600];
        makeSorted<uint32_t>(data, n, 15, 42 + n);
        char label[64];
        snprintf(label, sizeof(label), "b256-d1-n%u", n);
        rtd<uint32_t>(abpfor::b256::encodeDelta1, abpfor::b256::decodeDelta1, data, n, label, uint32_t(0));
    }
}

// ---------------------------------------------------------------------------
// 5. b256 — delta + outliers
// ---------------------------------------------------------------------------

static void test_b256_delta_outliers()
{
    printf("test_b256_delta_outliers...\n");

    for (unsigned excPct : {5u, 10u, 25u})
    {
        uint32_t data[256];
        std::mt19937 rng(42 + excPct);
        uint32_t v = 0;
        for (unsigned i = 0; i < 256; ++i)
        {
            if ((rng() % 100) < excPct)
                v += 1000 + rng() % 50000;
            else
                v += 1 + rng() % 5;
            data[i] = v;
        }
        char label[64];
        snprintf(label, sizeof(label), "b256-d1-out-exc%u%%", excPct);
        rtd<uint32_t>(abpfor::b256::encodeDelta1, abpfor::b256::decodeDelta1, data, 256, label, uint32_t(0));
    }

    for (unsigned baseBw : {4u, 8u, 16u})
    {
        uint32_t data[256];
        makeRandomWithOutliers<uint32_t>(data, 256, baseBw, 15, 42 + baseBw);
        char label[64];
        snprintf(label, sizeof(label), "b256-out-b%u-exc15%%", baseBw);
        rt<uint32_t>(abpfor::b256::encode, abpfor::b256::decode, data, 256, label);
    }
}

// ---------------------------------------------------------------------------
// 6. Edge cases
// ---------------------------------------------------------------------------

static void test_edge_cases()
{
    printf("test_edge_cases...\n");

    // All zeros
    {
        uint32_t z[256] = {};
        rt<uint32_t>(abpfor::b128::encode, abpfor::b128::decode, z, 128, "edge-zeros-b128");
        rt<uint32_t>(abpfor::b256::encode, abpfor::b256::decode, z, 256, "edge-zeros-b256");
    }

    // All constant
    for (uint32_t val : {uint32_t(1), uint32_t(42), uint32_t(0xDEADBEEF), uint32_t(0xFFFFFFFF)})
    {
        uint32_t data[256];
        std::fill_n(data, 256, val);
        char label[64];
        snprintf(label, sizeof(label), "edge-const-0x%X-b128", val);
        rt<uint32_t>(abpfor::b128::encode, abpfor::b128::decode, data, 128, label);
        snprintf(label, sizeof(label), "edge-const-0x%X-b256", val);
        rt<uint32_t>(abpfor::b256::encode, abpfor::b256::decode, data, 256, label);
    }

    // Single element
    {
        uint32_t one = 12345;
        rt<uint32_t>(abpfor::b128::encode, abpfor::b128::decode, &one, 1, "edge-single");
    }

    // n=1 with delta
    {
        uint32_t one = 100;
        rtd<uint32_t>(abpfor::b128::encodeDelta1, abpfor::b128::decodeDelta1, &one, 1, "edge-single-d1", uint32_t(50));
    }

    // Alternating 0 / max
    {
        uint32_t data[128];
        for (unsigned i = 0; i < 128; ++i)
            data[i] = (i & 1) ? 0xFFFFFFFF : 0;
        rt<uint32_t>(abpfor::b128::encode, abpfor::b128::decode, data, 128, "edge-alt-0max-b128");
    }

    // 64-bit edge cases
    {
        uint64_t z[128] = {};
        rt<uint64_t>(abpfor::b128::encode, abpfor::b128::decode, z, 128, "edge-zeros-u64");

        uint64_t one = 999999999999ULL;
        rt<uint64_t>(abpfor::b128::encode, abpfor::b128::decode, &one, 1, "edge-single-u64");

        uint64_t one_d1 = 1000;
        rtd<uint64_t>(abpfor::b128::encodeDelta1, abpfor::b128::decodeDelta1, &one_d1, 1, "edge-single-d1-u64", uint64_t(500));
    }
}

// ---------------------------------------------------------------------------
// 7. Stress test
// ---------------------------------------------------------------------------

static void test_stress()
{
    printf("test_stress (1000 random blocks)...\n");

    for (unsigned trial = 0; trial < 1000; ++trial)
    {
        std::mt19937 rng(trial);

        unsigned n = 1 + rng() % 256;
        unsigned bw = rng() % 33;
        unsigned excPct = rng() % 60;

        uint32_t data[256];
        makeRandomWithOutliers<uint32_t>(data, n, bw, excPct, trial * 7 + 1);

        char label[80];

        if (n >= 128)
        {
            snprintf(label, sizeof(label), "stress-%u-b128-b%u-exc%u%%", trial, bw, excPct);
            rt<uint32_t>(abpfor::b128::encode, abpfor::b128::decode, data, n, label);

            uint32_t sorted[256];
            makeSorted<uint32_t>(sorted, n, (1u << std::min(bw, 16u)) + 1, trial * 13);
            snprintf(label, sizeof(label), "stress-%u-b128-d1", trial);
            rtd<uint32_t>(abpfor::b128::encodeDelta1, abpfor::b128::decodeDelta1, sorted, n, label, uint32_t(0));

            if (n >= 256)
            {
                snprintf(label, sizeof(label), "stress-%u-b256-b%u-exc%u%%", trial, bw, excPct);
                rt<uint32_t>(abpfor::b256::encode, abpfor::b256::decode, data, n, label);

                snprintf(label, sizeof(label), "stress-%u-b256-d1", trial);
                rtd<uint32_t>(abpfor::b256::encodeDelta1, abpfor::b256::decodeDelta1, sorted, n, label, uint32_t(0));
            }
        }

        // Always test b128 (handles any n via tail)
        snprintf(label, sizeof(label), "stress-%u-b128-any-b%u-exc%u%%", trial, bw, excPct);
        rt<uint32_t>(abpfor::b128::encode, abpfor::b128::decode, data, n, label);

        if (trial % 3 == 0)
        {
            uint32_t sorted[256];
            makeSorted<uint32_t>(sorted, n, 255, trial * 17);
            snprintf(label, sizeof(label), "stress-%u-b128-d1-any", trial);
            rtd<uint32_t>(abpfor::b128::encodeDelta1, abpfor::b128::decodeDelta1, sorted, n, label, uint32_t(0));
        }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main()
{
    test_b128_all_bitwidths();
    test_b128_delta();
    test_b128_outliers();
    test_b256_all();
    test_b256_delta_outliers();
    test_edge_cases();
    test_stress();

    if (failures == 0)
        printf("All comprehensive tests passed.\n");
    else
        printf("%d comprehensive test(s) FAILED.\n", failures);

    return failures > 0 ? 1 : 0;
}
