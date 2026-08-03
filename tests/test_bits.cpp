// Layer 0 tests: bit utilities
// Tests bitwidth, mask, and unaligned load/store.

#include "core/bits.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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

// --- bitwidth ---

static void test_bitwidth32()
{
    printf("test_bitwidth32...\n");
    CHECK(abpfor::bitwidth(uint32_t(0)) == 0, "bitwidth(0)=%u", abpfor::bitwidth(uint32_t(0)));
    CHECK(abpfor::bitwidth(uint32_t(1)) == 1, "bitwidth(1)=%u", abpfor::bitwidth(uint32_t(1)));
    CHECK(abpfor::bitwidth(uint32_t(2)) == 2, "bitwidth(2)");
    CHECK(abpfor::bitwidth(uint32_t(3)) == 2, "bitwidth(3)");
    CHECK(abpfor::bitwidth(uint32_t(4)) == 3, "bitwidth(4)");
    CHECK(abpfor::bitwidth(uint32_t(127)) == 7, "bitwidth(127)");
    CHECK(abpfor::bitwidth(uint32_t(128)) == 8, "bitwidth(128)");
    CHECK(abpfor::bitwidth(uint32_t(255)) == 8, "bitwidth(255)");
    CHECK(abpfor::bitwidth(uint32_t(256)) == 9, "bitwidth(256)");
    CHECK(abpfor::bitwidth(uint32_t(0xFFFF)) == 16, "bitwidth(0xFFFF)");
    CHECK(abpfor::bitwidth(uint32_t(0x10000)) == 17, "bitwidth(0x10000)");
    CHECK(abpfor::bitwidth(uint32_t(0x7FFFFFFF)) == 31, "bitwidth(0x7FFFFFFF)");
    CHECK(abpfor::bitwidth(uint32_t(0x80000000)) == 32, "bitwidth(0x80000000)");
    CHECK(abpfor::bitwidth(uint32_t(0xFFFFFFFF)) == 32, "bitwidth(0xFFFFFFFF)");
}

static void test_bitwidth64()
{
    printf("test_bitwidth64...\n");
    CHECK(abpfor::bitwidth(uint64_t(0)) == 0, "bitwidth64(0)");
    CHECK(abpfor::bitwidth(uint64_t(1)) == 1, "bitwidth64(1)");
    CHECK(abpfor::bitwidth(uint64_t(0xFFFFFFFF)) == 32, "bitwidth64(32-bit max)");
    CHECK(abpfor::bitwidth(uint64_t(0x100000000ULL)) == 33, "bitwidth64(2^32)");
    CHECK(abpfor::bitwidth(uint64_t(0x8000000000000000ULL)) == 64, "bitwidth64(2^63)");
    CHECK(abpfor::bitwidth(uint64_t(0xFFFFFFFFFFFFFFFFULL)) == 64, "bitwidth64(64-bit max)");
}

// --- mask ---

static void test_mask32()
{
    printf("test_mask32...\n");
    CHECK(abpfor::mask<uint32_t>(0) == 0, "mask(0)");
    CHECK(abpfor::mask<uint32_t>(1) == 1, "mask(1)");
    CHECK(abpfor::mask<uint32_t>(8) == 0xFF, "mask(8)");
    CHECK(abpfor::mask<uint32_t>(16) == 0xFFFF, "mask(16)");
    CHECK(abpfor::mask<uint32_t>(31) == 0x7FFFFFFF, "mask(31)");
    CHECK(abpfor::mask<uint32_t>(32) == 0xFFFFFFFF, "mask(32)");
}

static void test_mask64()
{
    printf("test_mask64...\n");
    CHECK(abpfor::mask<uint64_t>(0) == 0, "mask64(0)");
    CHECK(abpfor::mask<uint64_t>(1) == 1, "mask64(1)");
    CHECK(abpfor::mask<uint64_t>(32) == 0xFFFFFFFF, "mask64(32)");
    CHECK(abpfor::mask<uint64_t>(63) == 0x7FFFFFFFFFFFFFFFULL, "mask64(63)");
    CHECK(abpfor::mask<uint64_t>(64) == 0xFFFFFFFFFFFFFFFFULL, "mask64(64)");
}

// --- load/store ---

static void test_loadu_storeu()
{
    printf("test_loadu_storeu...\n");

    // Test unaligned: offset by 1 byte from aligned buffer
    alignas(16) uint8_t buf[32] = {};

    // 16-bit
    abpfor::storeu<uint16_t>(buf + 1, 0x1234);
    CHECK(abpfor::loadu<uint16_t>(buf + 1) == 0x1234, "u16 roundtrip");
    // Little-endian byte order
    CHECK(buf[1] == 0x34 && buf[2] == 0x12, "u16 little-endian");

    // 32-bit
    abpfor::storeu<uint32_t>(buf + 3, 0xDEADBEEF);
    CHECK(abpfor::loadu<uint32_t>(buf + 3) == 0xDEADBEEF, "u32 roundtrip");
    CHECK(buf[3] == 0xEF && buf[4] == 0xBE && buf[5] == 0xAD && buf[6] == 0xDE, "u32 little-endian");

    // 64-bit
    abpfor::storeu<uint64_t>(buf + 5, 0x0102030405060708ULL);
    CHECK(abpfor::loadu<uint64_t>(buf + 5) == 0x0102030405060708ULL, "u64 roundtrip");
    CHECK(buf[5] == 0x08, "u64 le byte 0");
    CHECK(buf[12] == 0x01, "u64 le byte 7");
}

// --- divCeil ---

static void test_div_ceil()
{
    printf("test_divCeil...\n");
    CHECK(abpfor::divCeil(0u, 8u) == 0, "divCeil(0,8)");
    CHECK(abpfor::divCeil(1u, 8u) == 1, "divCeil(1,8)");
    CHECK(abpfor::divCeil(7u, 8u) == 1, "divCeil(7,8)");
    CHECK(abpfor::divCeil(8u, 8u) == 1, "divCeil(8,8)");
    CHECK(abpfor::divCeil(9u, 8u) == 2, "divCeil(9,8)");
    CHECK(abpfor::divCeil(128u * 7u, 8u) == 112, "divCeil(896,8)");
}

int main()
{
    test_bitwidth32();
    test_bitwidth64();
    test_mask32();
    test_mask64();
    test_loadu_storeu();
    test_div_ceil();

    if (failures == 0)
        printf("All Layer 0 tests passed.\n");
    else
        printf("%d Layer 0 test(s) FAILED.\n", failures);

    return failures > 0 ? 1 : 0;
}
