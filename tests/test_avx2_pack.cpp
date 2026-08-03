// Layer 4 tests: AVX2 interleaved pack/unpack (Interleave8, 256 elements)

#include "simd/avx2_pack.h"
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

static void test_roundtrip_i8()
{
    printf("test_roundtrip_i8...\n");

    std::mt19937 rng(42);
    alignas(32) uint32_t orig[256];
    alignas(32) uint32_t decoded[256];
    alignas(32) uint8_t packed[256 * 4 + 64];

    for (unsigned b = 0; b <= 32; ++b)
    {
        uint32_t m = abpfor::mask<uint32_t>(b);
        for (unsigned i = 0; i < 256; ++i)
            orig[i] = (b == 0) ? 0 : (rng() & m);

        std::memset(packed, 0xCC, sizeof(packed));
        std::memset(decoded, 0xDD, sizeof(decoded));

        uint8_t* end = abpfor::packI8(orig, packed, b);
        unsigned written = static_cast<unsigned>(end - packed);
        unsigned expected = abpfor::packedBytes(256, b);
        CHECK(written == expected, "b=%u: pack wrote %u, expected %u", b, written, expected);

        const uint8_t* consumed = abpfor::unpackI8(packed, decoded, b);
        unsigned read = static_cast<unsigned>(consumed - packed);
        CHECK(read == expected, "b=%u: unpack read %u, expected %u", b, read, expected);

        for (unsigned i = 0; i < 256; ++i)
        {
            CHECK(decoded[i] == orig[i],
                  "b=%u i=%u: decoded=%u expected=%u", b, i, decoded[i], orig[i]);
            if (decoded[i] != orig[i]) break;
        }
    }
}

static void test_sequential_i8()
{
    printf("test_sequential_i8...\n");
    alignas(32) uint32_t data[256];
    alignas(32) uint32_t decoded[256];
    uint8_t packed[2048];

    for (unsigned b : {4u, 8u, 16u})
    {
        uint32_t m = abpfor::mask<uint32_t>(b);
        for (unsigned i = 0; i < 256; ++i) data[i] = i & m;

        abpfor::packI8(data, packed, b);
        abpfor::unpackI8(packed, decoded, b);

        for (unsigned i = 0; i < 256; ++i)
            CHECK(decoded[i] == data[i], "seq b=%u i=%u: %u != %u", b, i, decoded[i], data[i]);
    }
}

static void test_all_max_i8()
{
    printf("test_all_max_i8...\n");
    alignas(32) uint32_t data[256];
    alignas(32) uint32_t decoded[256];
    uint8_t packed[2048];

    for (unsigned b = 1; b <= 32; ++b)
    {
        uint32_t m = abpfor::mask<uint32_t>(b);
        for (unsigned i = 0; i < 256; ++i) data[i] = m;

        abpfor::packI8(data, packed, b);
        abpfor::unpackI8(packed, decoded, b);

        for (unsigned i = 0; i < 256; ++i)
        {
            CHECK(decoded[i] == m, "allmax b=%u i=%u: %u != %u", b, i, decoded[i], m);
            if (decoded[i] != m) break;
        }
    }
}

int main()
{
    test_roundtrip_i8();
    test_sequential_i8();
    test_all_max_i8();

    if (failures == 0)
        printf("All Layer 4 AVX2 tests passed.\n");
    else
        printf("%d Layer 4 AVX2 test(s) FAILED.\n", failures);

    return failures > 0 ? 1 : 0;
}
