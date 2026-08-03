// Test: uint64_t b128/b256 roundtrip — all bit-widths, delta, outliers.

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
using EncFn = size_t(*)(const T*, unsigned, uint8_t*);
template <typename T>
using DecFn = size_t(*)(const uint8_t*, unsigned, T*);
template <typename T>
using EncDFn = size_t(*)(const T*, unsigned, uint8_t*, T);
template <typename T>
using DecDFn = size_t(*)(const uint8_t*, unsigned, T*, T);

template <typename T>
void rt(EncFn<T> enc, DecFn<T> dec, const T* data, unsigned n, const char* label)
{
    std::vector<uint8_t> comp(abpfor::maxCompressedSize<T>(n));
    std::vector<T> out(n, T(0xDEAD));
    size_t enc_bytes = enc(data, n, comp.data());
    CHECK(enc_bytes > 0, "%s: encode returned 0", label);
    size_t dec_bytes = dec(comp.data(), n, out.data());
    CHECK(dec_bytes == enc_bytes, "%s: decode consumed %zu, encode wrote %zu", label, dec_bytes, enc_bytes);
    bool ok = std::memcmp(data, out.data(), n * sizeof(T)) == 0;
    CHECK(ok, "%s: data mismatch", label);
    if (!ok)
        for (unsigned i = 0; i < n && i < 8; ++i)
            if (data[i] != out[i])
                printf("    [%u] expected %llu got %llu\n", i,
                       (unsigned long long)data[i], (unsigned long long)out[i]);
}

template <typename T>
void rtd(EncDFn<T> enc, DecDFn<T> dec, const T* data, unsigned n, const char* label, T start)
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
        for (unsigned i = 0; i < n && i < 8; ++i)
            if (data[i] != out[i])
                printf("    [%u] expected %llu got %llu\n", i,
                       (unsigned long long)data[i], (unsigned long long)out[i]);
}

template <typename T>
void makeRandom(T* out, unsigned n, unsigned bw, unsigned seed)
{
    std::mt19937_64 rng(seed);
    T vmask = (bw >= 64) ? ~T(0) : ((T(1) << bw) - 1);
    for (unsigned i = 0; i < n; ++i) out[i] = static_cast<T>(rng()) & vmask;
}

template <typename T>
void makeSorted(T* out, unsigned n, T maxDelta, unsigned seed)
{
    std::mt19937_64 rng(seed);
    T v = 0;
    for (unsigned i = 0; i < n; ++i)
    {
        v += T(1) + static_cast<T>(rng() % static_cast<uint64_t>(maxDelta));
        out[i] = v;
    }
}

int main()
{
    printf("=== b128 uint64_t — all bit-widths ===\n");
    for (unsigned bw = 1; bw <= 64; ++bw)
    {
        uint64_t data[128];
        makeRandom<uint64_t>(data, 128, bw, 1000 + bw);
        char label[64];
        snprintf(label, sizeof(label), "b128-u64-bw%u", bw);
        rt<uint64_t>(abpfor::b128::encode, abpfor::b128::decode, data, 128, label);
    }

    printf("=== b128 uint64_t — delta ===\n");
    for (unsigned bw : {4u, 16u, 32u, 48u, 64u})
    {
        uint64_t data[128];
        makeSorted<uint64_t>(data, 128, (bw >= 64) ? uint64_t(1000) : ((uint64_t(1) << bw) - 1), 2000 + bw);
        char label[64];
        snprintf(label, sizeof(label), "b128-d1-u64-bw%u", bw);
        rtd<uint64_t>(abpfor::b128::encodeDelta1, abpfor::b128::decodeDelta1, data, 128, label, uint64_t(0));
    }

    printf("=== b128 uint64_t — with tail ===\n");
    {
        uint64_t data[200];
        makeRandom<uint64_t>(data, 200, 40, 3000);
        rt<uint64_t>(abpfor::b128::encode, abpfor::b128::decode, data, 200, "b128-u64-tail");
    }

    printf("=== b128 uint64_t — outliers ===\n");
    {
        uint64_t data[128];
        makeRandom<uint64_t>(data, 128, 8, 4000);
        data[5] = uint64_t(1) << 50;
        data[100] = uint64_t(1) << 60;
        rt<uint64_t>(abpfor::b128::encode, abpfor::b128::decode, data, 128, "b128-u64-outliers");
    }

    printf("=== b128 uint64_t — outliers + delta ===\n");
    {
        uint64_t data[128];
        makeSorted<uint64_t>(data, 128, 100, 4500);
        data[30] += uint64_t(1) << 50;
        for (unsigned i = 31; i < 128; ++i) data[i] = std::max(data[i], data[i - 1] + 1);
        rtd<uint64_t>(abpfor::b128::encodeDelta1, abpfor::b128::decodeDelta1, data, 128, "b128-d1-u64-outliers", uint64_t(0));
    }

    printf("=== b256 uint64_t — all bit-widths ===\n");
    for (unsigned bw = 1; bw <= 64; ++bw)
    {
        uint64_t data[256];
        makeRandom<uint64_t>(data, 256, bw, 5000 + bw);
        char label[64];
        snprintf(label, sizeof(label), "b256-u64-bw%u", bw);
        rt<uint64_t>(abpfor::b256::encode, abpfor::b256::decode, data, 256, label);
    }

    printf("=== b256 uint64_t — delta ===\n");
    for (unsigned bw : {4u, 16u, 32u, 48u, 64u})
    {
        uint64_t data[256];
        makeSorted<uint64_t>(data, 256, (bw >= 64) ? uint64_t(1000) : ((uint64_t(1) << bw) - 1), 6000 + bw);
        char label[64];
        snprintf(label, sizeof(label), "b256-d1-u64-bw%u", bw);
        rtd<uint64_t>(abpfor::b256::encodeDelta1, abpfor::b256::decodeDelta1, data, 256, label, uint64_t(0));
    }

    printf("=== b256 uint64_t — outliers + delta ===\n");
    {
        uint64_t data[256];
        makeSorted<uint64_t>(data, 256, 100, 7000);
        data[50] += uint64_t(1) << 55;
        for (unsigned i = 51; i < 256; ++i) data[i] = std::max(data[i], data[i - 1] + 1);
        rtd<uint64_t>(abpfor::b256::encodeDelta1, abpfor::b256::decodeDelta1, data, 256, "b256-d1-u64-outliers", uint64_t(0));
    }

    printf("\n%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
