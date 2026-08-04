// Layer 1 tests: scalar bitpack / bitunpack
// Verifies pack→unpack roundtrip for all bit-widths and various element counts.

#include "core/pack.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sys/mman.h>
#include <unistd.h>

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

// --- 32-bit pack/unpack roundtrip ---

static void test_pack32_roundtrip()
{
    printf("test_pack32_roundtrip...\n");

    std::mt19937 rng(42);
    uint32_t orig[256];
    uint32_t decoded[256];
    uint8_t packed[256 * 4 + 64]; // generous buffer

    for (unsigned b = 0; b <= 32; ++b)
    {
        uint32_t m = abpfor::mask<uint32_t>(b);

        for (unsigned n : {1u, 2u, 3u, 7u, 8u, 15u, 16u, 31u, 32u, 63u, 64u, 127u, 128u, 200u, 255u, 256u})
        {
            // Fill with random values that fit in b bits
            for (unsigned i = 0; i < n; ++i)
                orig[i] = (b == 0) ? 0 : static_cast<uint32_t>(rng() & m);

            std::memset(packed, 0xCC, sizeof(packed));
            std::memset(decoded, 0xDD, sizeof(decoded));

            uint8_t* end = abpfor::pack(orig, n, packed, b);
            unsigned written = static_cast<unsigned>(end - packed);
            unsigned expected = abpfor::packedBytes(n, b);

            CHECK(written == expected,
                  "b=%u n=%u: pack wrote %u bytes, expected %u", b, n, written, expected);

            const uint8_t* consumed = abpfor::unpack(packed, n, decoded, b);
            unsigned read = static_cast<unsigned>(consumed - packed);

            CHECK(read == expected,
                  "b=%u n=%u: unpack consumed %u bytes, expected %u", b, n, read, expected);

            for (unsigned i = 0; i < n; ++i)
            {
                CHECK(decoded[i] == orig[i],
                      "b=%u n=%u i=%u: decoded=%u expected=%u", b, n, i, decoded[i], orig[i]);
                if (decoded[i] != orig[i]) break; // avoid flood
            }
        }
    }
}

// --- 64-bit pack/unpack roundtrip ---

static void test_pack64_roundtrip()
{
    printf("test_pack64_roundtrip...\n");

    std::mt19937_64 rng(123);
    uint64_t orig[256];
    uint64_t decoded[256];
    uint8_t packed[256 * 8 + 64];

    for (unsigned b = 0; b <= 64; ++b)
    {
        uint64_t m = abpfor::mask<uint64_t>(b);

        for (unsigned n : {1u, 4u, 16u, 64u, 127u, 128u, 256u})
        {
            for (unsigned i = 0; i < n; ++i)
                orig[i] = (b == 0) ? 0 : (rng() & m);

            std::memset(packed, 0xCC, sizeof(packed));
            std::memset(decoded, 0xDD, sizeof(decoded));

            uint8_t* end = abpfor::pack(orig, n, packed, b);
            unsigned written = static_cast<unsigned>(end - packed);
            unsigned expected = abpfor::packedBytes(n, b);

            CHECK(written == expected,
                  "b=%u n=%u: pack64 wrote %u, expected %u", b, n, written, expected);

            const uint8_t* consumed = abpfor::unpack(packed, n, decoded, b);
            unsigned read = static_cast<unsigned>(consumed - packed);

            CHECK(read == expected,
                  "b=%u n=%u: unpack64 consumed %u, expected %u", b, n, read, expected);

            for (unsigned i = 0; i < n; ++i)
            {
                CHECK(decoded[i] == orig[i],
                      "b=%u n=%u i=%u: decoded=%llu expected=%llu",
                      b, n, i, static_cast<unsigned long long>(decoded[i]), static_cast<unsigned long long>(orig[i]));
                if (decoded[i] != orig[i]) break;
            }
        }
    }
}

// --- Edge cases ---

static void test_pack_edge_cases()
{
    printf("test_pack_edge_cases...\n");

    // b=0: no bytes written, all decoded as 0
    {
        uint32_t orig[8] = {};
        uint32_t decoded[8];
        uint8_t packed[1];
        uint8_t* end = abpfor::pack(orig, 8, packed, 0);
        CHECK(end == packed, "b=0: should write 0 bytes");
        const uint8_t* consumed = abpfor::unpack(packed, 8, decoded, 0);
        CHECK(consumed == packed, "b=0: should read 0 bytes");
        for (unsigned i = 0; i < 8; ++i)
            CHECK(decoded[i] == 0, "b=0: decoded[%u]=%u", i, decoded[i]);
    }

    // b=32: all bits, should be identity (modulo endianness)
    {
        uint32_t orig[4] = {0xDEADBEEF, 0x12345678, 0xFFFFFFFF, 0x00000001};
        uint32_t decoded[4];
        uint8_t packed[16];
        abpfor::pack(orig, 4, packed, 32);
        abpfor::unpack(packed, 4, decoded, 32);
        for (unsigned i = 0; i < 4; ++i)
            CHECK(decoded[i] == orig[i], "b=32 i=%u: %u != %u", i, decoded[i], orig[i]);
    }

    // n=1: single element at various bit widths
    {
        for (unsigned b = 1; b <= 32; ++b)
        {
            uint32_t val = (1u << (b - 1)) | 1u;
            uint32_t decoded = 0;
            uint8_t packed[8] = {};
            abpfor::pack(&val, 1, packed, b);
            abpfor::unpack(packed, 1, &decoded, b);
            CHECK(decoded == val, "n=1 b=%u: %u != %u", b, decoded, val);
        }
    }
}

// --- Read bounds ---
//
// unpack() reads whole 64-bit words and lets the last one run past the packed
// bytes rather than assembling it byte by byte. That is a deliberate speed
// choice, but it means the caller must provide slack after the packed data.
// The bound is a documented part of the API (see include/abpfor.h), so it is
// pinned here: a future width-specialised loop that grabs a full u64 per pair
// would silently widen it, and this test is what stops that.
//
// Method: place the packed data so that exactly kUnpackSlack bytes remain
// before an unmapped page. Any read beyond the documented bound faults.
// The asm barriers are required -- without them the compiler deletes the call
// as dead and the guard page never fires, so the test would pass vacuously.
static constexpr unsigned kUnpackSlack = 3;

static void test_unpack_read_bounds()
{
    printf("test_unpack_read_bounds...\n");

    const size_t pageSize = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    std::mt19937 rng(1234);
    uint32_t orig[128];
    uint32_t decoded[128];
    uint8_t packed[128 * 4 + 64];

    for (unsigned b = 1; b <= 32; ++b)
    {
        uint32_t m = abpfor::mask<uint32_t>(b);
        for (unsigned i = 0; i < 128; ++i) orig[i] = static_cast<uint32_t>(rng() & m);

        for (unsigned n = 1; n <= 128; ++n)
        {
            std::memset(packed, 0, sizeof(packed));
            abpfor::pack(orig, n, packed, b);
            const unsigned packedBytes = (n * b + 7u) / 8u;

            char* region = static_cast<char*>(mmap(nullptr, 2 * pageSize,
                                                   PROT_READ | PROT_WRITE,
                                                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
            CHECK(region != MAP_FAILED, "mmap failed");
            if (region == MAP_FAILED) return;
            CHECK(mprotect(region + pageSize, pageSize, PROT_NONE) == 0, "mprotect failed");

            uint8_t* tight = reinterpret_cast<uint8_t*>(
                region + pageSize - packedBytes - kUnpackSlack);
            std::memcpy(tight, packed, packedBytes);

            std::memset(decoded, 0, sizeof(decoded));
            asm volatile("" :: "r"(tight) : "memory");
            const uint8_t* end = abpfor::unpack(tight, n, decoded, b);
            asm volatile("" :: "r"(decoded) : "memory");

            CHECK(end == tight + packedBytes,
                  "b=%u n=%u: consumed %ld bytes, expected %u",
                  b, n, static_cast<long>(end - tight), packedBytes);
            for (unsigned i = 0; i < n; ++i)
                CHECK(decoded[i] == orig[i],
                      "b=%u n=%u i=%u: %u != %u", b, n, i, decoded[i], orig[i]);

            munmap(region, 2 * pageSize);
        }
    }
}

int main()
{
    test_pack32_roundtrip();
    test_pack64_roundtrip();
    test_pack_edge_cases();
    test_unpack_read_bounds();

    if (failures == 0)
        printf("All Layer 1 tests passed.\n");
    else
        printf("%d Layer 1 test(s) FAILED.\n", failures);

    return failures > 0 ? 1 : 0;
}
