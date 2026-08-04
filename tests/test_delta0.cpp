// Test: Delta0 (plain delta, no -1) encode/decode roundtrip.

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

template <typename T>
using EncDFn = size_t(*)(const T*, unsigned, uint8_t*, T);
template <typename T>
using DecDFn = size_t(*)(const uint8_t*, unsigned, T*, T);

template <typename T>
void rt(EncDFn<T> enc, DecDFn<T> dec, const T* data, unsigned n, const char* label, T start)
{
    std::vector<uint8_t> comp(abpfor::maxCompressedSize<T>(n));
    std::vector<T> out(n, T(0xDEAD));

    size_t enc_bytes = enc(data, n, comp.data(), start);
    CHECK(enc_bytes > 0, "%s: encode returned 0", label);

    size_t dec_bytes = dec(comp.data(), n, out.data(), start);
    CHECK(dec_bytes == enc_bytes, "%s: decode consumed %zu, encode wrote %zu", label, dec_bytes, enc_bytes);

    bool ok = std::memcmp(data, out.data(), n * sizeof(T)) == 0;
    CHECK(ok, "%s: data mismatch", label);
    if (!ok)
    {
        for (unsigned i = 0; i < n && i < 8; ++i)
        {
            if (data[i] != out[i])
                printf("    [%u] expected %llu got %llu\n", i,
                       static_cast<unsigned long long>(data[i]), static_cast<unsigned long long>(out[i]));
        }
    }
}

int main()
{
    // === b128 Delta0 uint32_t ===
    printf("=== b128 Delta0 uint32_t ===\n");
    {
        uint32_t data[] = {10, 10, 10, 15, 15, 20, 25, 25, 25, 30};
        rt<uint32_t>(abpfor::b128::encodeDelta0, abpfor::b128::decodeDelta0, data, 10, "b128-d0-dup", uint32_t(10));
    }
    {
        uint32_t data[128];
        for (unsigned i = 0; i < 128; ++i) data[i] = i * 3 + 100;
        rt<uint32_t>(abpfor::b128::encodeDelta0, abpfor::b128::decodeDelta0, data, 128, "b128-d0-strict", uint32_t(0));
    }
    {
        uint32_t data[200];
        std::mt19937 rng(42);
        data[0] = static_cast<uint32_t>(rng() % 1000);
        for (unsigned i = 1; i < 200; ++i) data[i] = data[i - 1] + static_cast<uint32_t>(rng() % 10000);
        rt<uint32_t>(abpfor::b128::encodeDelta0, abpfor::b128::decodeDelta0, data, 200, "b128-d0-large", uint32_t(0));
    }

    // === b128 Delta0 uint64_t ===
    printf("=== b128 Delta0 uint64_t ===\n");
    {
        uint64_t data[128];
        for (unsigned i = 0; i < 128; ++i) data[i] = uint64_t(i) * 1000000ULL;
        rt<uint64_t>(abpfor::b128::encodeDelta0, abpfor::b128::decodeDelta0, data, 128, "b128-d0-u64", uint64_t(0));
    }

    // === b128 Delta0 (128 elements) ===
    printf("=== b128 Delta0 block ===\n");
    {
        uint32_t data[128];
        data[0] = 5;
        for (unsigned i = 1; i < 128; ++i) data[i] = data[i - 1] + (i % 3);
        rt<uint32_t>(abpfor::b128::encodeDelta0, abpfor::b128::decodeDelta0, data, 128, "b128-d0-block", uint32_t(0));
    }
    {
        uint32_t data[128];
        std::fill_n(data, 128, uint32_t(42));
        rt<uint32_t>(abpfor::b128::encodeDelta0, abpfor::b128::decodeDelta0, data, 128, "b128-d0-allsame", uint32_t(42));
    }

    // === b256 Delta0 ===
    printf("=== b256 Delta0 ===\n");
    {
        uint32_t data[256];
        data[0] = 100;
        for (unsigned i = 1; i < 256; ++i) data[i] = data[i - 1] + (i % 5);
        rt<uint32_t>(abpfor::b256::encodeDelta0, abpfor::b256::decodeDelta0, data, 256, "b256-d0-u32", uint32_t(0));
    }

    // === b128 Delta0 uint64_t (128 elements) ===
    printf("=== b128 Delta0 uint64_t block ===\n");
    {
        uint64_t data[128];
        data[0] = 1000;
        for (unsigned i = 1; i < 128; ++i) data[i] = data[i - 1] + (i % 4);
        rt<uint64_t>(abpfor::b128::encodeDelta0, abpfor::b128::decodeDelta0, data, 128, "b128-d0-u64-block", uint64_t(0));
    }

    // === Delta0 with nonzero start ===
    printf("=== Delta0 with nonzero start ===\n");
    {
        uint32_t data[128];
        for (unsigned i = 0; i < 128; ++i) data[i] = 5000 + i * 2;
        rt<uint32_t>(abpfor::b128::encodeDelta0, abpfor::b128::decodeDelta0, data, 128, "b128-d0-start5000", uint32_t(5000));
    }

    // === Delta0 start == first element ===
    printf("=== Delta0 start == first ===\n");
    {
        uint32_t data[] = {7, 7, 8, 8, 9};
        rt<uint32_t>(abpfor::b128::encodeDelta0, abpfor::b128::decodeDelta0, data, 5, "b128-d0-start-eq", uint32_t(7));
    }

    printf("\n%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
