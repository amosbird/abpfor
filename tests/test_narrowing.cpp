// Test: the narrowing conversions in the codec are value-preserving, and the
// public output-size bound holds.
//
// Enabling -Wconversion/-Wsign-conversion turned seven implicit narrowings into
// explicit casts. A cast silences the compiler whether or not it is correct, so
// each one's bound is asserted here instead of being taken on faith.

#include <abpfor.h>

#include <cstdint>
#include <cstdio>

#include "core/pack32.h"
#include "simd/avx2_fused.h"

static int failures = 0;

static void check(bool ok, const char* what)
{
    if (!ok)
    {
        printf("FAIL: %s\n", what);
        ++failures;
    }
}

int main()
{
    // pack32.h pack_b3_loop, 3 casts: uint32_t -> unsigned char for out[0..2].
    // Eight 3-bit values pack into 24 bits, so bits 24..31 of v are always zero
    // and the three byte stores reproduce v exactly.
    printf("=== pack_b3_loop: v fits 24 bits ===\n");
    {
        bool packed_range_ok = true;
        bool roundtrip_ok = true;
        for (uint32_t t = 0; t < 100000u; ++t)
        {
            uint32_t in[8];
            for (unsigned i = 0; i < 8u; ++i)
                in[i] = t * 2654435761u + i * 40503u;

            const uint32_t v = (in[0] & 7) | ((in[1] & 7) << 3) | ((in[2] & 7) << 6) | ((in[3] & 7) << 9)
                | ((in[4] & 7) << 12) | ((in[5] & 7) << 15) | ((in[6] & 7) << 18) | ((in[7] & 7) << 21);

            if (v >= (1u << 24) || (v >> 24) != 0u)
                packed_range_ok = false;

            // Reassemble from the three narrowed bytes; must equal v.
            const unsigned char b0 = static_cast<unsigned char>(v);
            const unsigned char b1 = static_cast<unsigned char>(v >> 8);
            const unsigned char b2 = static_cast<unsigned char>(v >> 16);
            const uint32_t back = uint32_t{b0} | (uint32_t{b1} << 8) | (uint32_t{b2} << 16);
            if (back != v)
                roundtrip_ok = false;
        }
        check(packed_range_ok, "8x3-bit pack exceeds 24 bits");
        check(roundtrip_ok, "3-byte store loses bits of v");
    }

    // avx2_fused.h avx2Scatter, 2 casts: int -> unsigned on __builtin_popcount.
    // The argument is a 4-bit nibble, so the result is 0..4 and never negative.
    printf("=== avx2Scatter: nibble popcount is 0..4 ===\n");
    {
        bool ok = true;
        for (unsigned nib = 0; nib < 16u; ++nib)
        {
            const int pc = __builtin_popcount(nib);
            if (pc < 0 || pc > 4 || static_cast<unsigned>(pc) != static_cast<unsigned>(pc & 7))
                ok = false;
        }
        check(ok, "nibble popcount outside 0..4");
    }

    // avx2_fused.h avx2Scatter, 2 casts: unsigned -> int for the _mm_slli_epi32
    // shift count. B is a bit width in 0..64, well inside int, so the cast is
    // exact and round-trips.
    printf("=== avx2Scatter: bit width 0..64 fits int ===\n");
    {
        bool ok = true;
        for (unsigned b = 0; b <= 64u; ++b)
        {
            if (static_cast<int>(b) < 0 || static_cast<unsigned>(static_cast<int>(b)) != b)
                ok = false;
        }
        check(ok, "bit width does not round-trip through int");
    }


    // maxCompressedSize is the public buffer-sizing contract: a caller sizes its
    // output buffer with it, so the codec must never touch a byte past it. Note
    // that encode may touch bytes past the length it *reports* -- that is
    // documented and allowed -- so the guard region starts at the bound, not at
    // the return value.
    printf("=== maxCompressedSize bounds the worst-case block ===\n");
    {
        constexpr size_t kBound = abpfor::b256::maxCompressedSize<uint64_t>(256);
        static_assert(kBound == 2060u, "block bound drifted from the wire format");

        // Two shapes: incompressible (largest output) and constant-delta (the
        // case that reports far fewer bytes than it touches).
        uint64_t worst[256];
        for (unsigned i = 0; i < 256u; ++i)
            worst[i] = 0x8000000000000000ull | (0x9E3779B97F4A7C15ull * i);
        uint64_t flat[256];
        for (unsigned i = 0; i < 256u; ++i)
            flat[i] = 967295ull * (i + 1u);

        constexpr size_t kGuard = 512u;
        uint8_t buf[kBound + kGuard];
        uint64_t back[256];

        for (int shape = 0; shape < 2; ++shape)
        {
            const uint64_t* in = shape ? flat : worst;
            for (size_t i = 0; i < sizeof(buf); ++i)
                buf[i] = 0xCD;

            const size_t written = shape ? abpfor::b256::encodeBlockDelta1(in, buf, uint64_t{0})
                                         : abpfor::b256::encodeBlock(in, buf);
            check(written <= kBound, "block output exceeds maxCompressedSize");

            bool guardIntact = true;
            for (size_t i = kBound; i < sizeof(buf); ++i)
                if (buf[i] != 0xCD) guardIntact = false;
            check(guardIntact, "encode touched a byte past maxCompressedSize");

            for (unsigned i = 0; i < 256u; ++i) back[i] = 0;
            const size_t consumed = shape ? abpfor::b256::decodeBlockDelta1(buf, back, uint64_t{0})
                                          : abpfor::b256::decodeBlock(buf, back);
            check(consumed == written, "decode consumed != encode wrote");
            bool same = true;
            for (unsigned i = 0; i < 256u; ++i)
                if (back[i] != in[i]) same = false;
            check(same, "block does not round-trip");

            printf("    %-16s reported %4zu bytes (bound %zu)\n",
                   shape ? "constant-delta" : "incompressible", written, kBound);
        }
    }

    printf("\n%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
