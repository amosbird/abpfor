// Cross-platform compatibility test: verify SSE/AVX2 and scalar_interleaved
// implementations produce identical binary layouts and can decode each other's output.

#include "detail/codec.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

// Pull scalar implementations into a separate namespace to avoid ODR conflicts
// with the SSE/AVX2 versions already in namespace abpfor.
namespace scalar_ref {
    using abpfor::packedBytes; // reuse the constexpr utility
    namespace detail {
        // Inline the scalar logic directly (mirrors scalar_interleaved.h)
        template <unsigned Lanes>
        inline const uint8_t* unpackInterleaved(const uint8_t* in, uint32_t* out, unsigned b, unsigned count)
        {
            if (b == 0) { std::memset(out, 0, count * sizeof(uint32_t)); return in; }
            if (b == 32) { std::memcpy(out, in, count * sizeof(uint32_t)); return in + count * sizeof(uint32_t); }
            const uint32_t* stripes = reinterpret_cast<const uint32_t*>(in);
            const uint32_t vmask = (1u << b) - 1u;
            for (unsigned g = 0; g < 32; ++g) {
                unsigned bitOff = g * b;
                unsigned stripeIdx = bitOff / 32u;
                unsigned shift = bitOff % 32u;
                bool spans = (shift + b > 32u) && (b < 32u);
                for (unsigned lane = 0; lane < Lanes; ++lane) {
                    uint32_t val = stripes[stripeIdx * Lanes + lane] >> shift;
                    if (spans) val |= stripes[(stripeIdx + 1) * Lanes + lane] << (32u - shift);
                    out[g * Lanes + lane] = val & vmask;
                }
            }
            return in + packedBytes(count, b);
        }

        template <unsigned Lanes>
        inline uint8_t* packInterleaved(const uint32_t* in, uint8_t* out, unsigned b, unsigned count)
        {
            if (b == 0) return out;
            if (b == 32) { std::memcpy(out, in, count * sizeof(uint32_t)); return out + count * sizeof(uint32_t); }
            unsigned totalBits = 32u * b;
            unsigned numStripes = (totalBits + 31u) / 32u;
            unsigned numWords = numStripes * Lanes;
            uint32_t stripeBuf[32 * Lanes];
            std::memset(stripeBuf, 0, numWords * sizeof(uint32_t));
            for (unsigned g = 0; g < 32; ++g) {
                unsigned bitOff = g * b;
                unsigned stripeIdx = bitOff / 32u;
                unsigned shift = bitOff % 32u;
                bool spans = (shift + b > 32u) && (b < 32u);
                for (unsigned lane = 0; lane < Lanes; ++lane) {
                    uint32_t v = in[g * Lanes + lane];
                    stripeBuf[stripeIdx * Lanes + lane] |= v << shift;
                    if (spans) stripeBuf[(stripeIdx + 1) * Lanes + lane] |= v >> (32u - shift);
                }
            }
            unsigned totalBytes = packedBytes(count, b);
            std::memcpy(out, stripeBuf, totalBytes);
            return out + totalBytes;
        }

        template <unsigned Lanes, bool Delta, bool Patch>
        inline void fusedDecodeInterleaved(const uint8_t* in, uint32_t* out, unsigned b, uint32_t& carry,
                                           const uint64_t* bitmap, const uint32_t* residuals, unsigned count)
        {
            const uint32_t* pex = residuals;
            if (b == 0) {
                if constexpr (Delta) {
                    for (unsigned g = 0; g < 32; ++g)
                        for (unsigned lane = 0; lane < Lanes; ++lane) {
                            uint32_t val = 0;
                            if constexpr (Patch) {
                                unsigned idx = g * Lanes + lane;
                                if (bitmap[idx / 64] & (uint64_t(1) << (idx % 64))) val = *pex++;
                            }
                            carry += 1u + val;
                            out[g * Lanes + lane] = carry;
                        }
                } else {
                    std::memset(out, 0, count * sizeof(uint32_t));
                    if constexpr (Patch) {
                        for (unsigned i = 0; i < count; ++i)
                            if (bitmap[i / 64] & (uint64_t(1) << (i % 64))) out[i] = *pex++;
                    }
                }
                return;
            }
            if (b == 32) {
                std::memcpy(out, in, count * sizeof(uint32_t));
                if constexpr (Delta) {
                    for (unsigned i = 0; i < count; ++i) { carry += out[i] + 1u; out[i] = carry; }
                }
                return;
            }
            const uint32_t* stripes = reinterpret_cast<const uint32_t*>(in);
            const uint32_t vmask = (1u << b) - 1u;
            for (unsigned g = 0; g < 32; ++g) {
                unsigned bitOff = g * b;
                unsigned stripeIdx = bitOff / 32u;
                unsigned shift = bitOff % 32u;
                bool spans = (shift + b > 32u) && (b < 32u);
                for (unsigned lane = 0; lane < Lanes; ++lane) {
                    uint32_t val = stripes[stripeIdx * Lanes + lane] >> shift;
                    if (spans) val |= stripes[(stripeIdx + 1) * Lanes + lane] << (32u - shift);
                    val &= vmask;
                    if constexpr (Patch) {
                        unsigned idx = g * Lanes + lane;
                        if (bitmap[idx / 64] & (uint64_t(1) << (idx % 64))) val += (*pex++) << b;
                    }
                    if constexpr (Delta) { carry += val + 1u; out[g * Lanes + lane] = carry; }
                    else { out[g * Lanes + lane] = val; }
                }
            }
        }
    } // namespace detail

    inline const uint8_t* unpackI4(const uint8_t* in, uint32_t* out, unsigned b)
    { return detail::unpackInterleaved<4>(in, out, b, 128); }
    inline uint8_t* packI4(const uint32_t* in, uint8_t* out, unsigned b)
    { return detail::packInterleaved<4>(in, out, b, 128); }
    template <bool Delta, bool Patch>
    inline void fusedDecodeI4(const uint8_t* in, uint32_t* out, unsigned b, uint32_t& carry,
                              const uint64_t* bitmap, const uint32_t* residuals)
    { detail::fusedDecodeInterleaved<4, Delta, Patch>(in, out, b, carry, bitmap, residuals, 128); }

    inline const uint8_t* unpackI8(const uint8_t* in, uint32_t* out, unsigned b)
    { return detail::unpackInterleaved<8>(in, out, b, 256); }
    inline uint8_t* packI8(const uint32_t* in, uint8_t* out, unsigned b)
    { return detail::packInterleaved<8>(in, out, b, 256); }
    template <bool Delta, bool Patch>
    inline void fusedDecodeI8(const uint8_t* in, uint32_t* out, unsigned b, uint32_t& carry,
                              const uint64_t* bitmap, const uint32_t* residuals)
    { detail::fusedDecodeInterleaved<8, Delta, Patch>(in, out, b, carry, bitmap, residuals, 256); }
} // namespace scalar_ref

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::mt19937 rng(42);

static std::vector<uint32_t> randomValues(unsigned count, unsigned bits)
{
    std::vector<uint32_t> v(count);
    uint32_t mask = bits == 32 ? ~0u : (1u << bits) - 1u;
    for (auto& x : v) x = rng() & mask;
    return v;
}

static int failures = 0;

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); ++failures; } } while(0)

// ---------------------------------------------------------------------------
// Test 1: I4 format compatibility (SSE pack → scalar unpack, scalar pack → SSE unpack)
// ---------------------------------------------------------------------------
static void testI4Compatibility()
{
    printf("  I4 format compatibility (B=1..32)...\n");
    alignas(64) uint8_t buf[128 * 4 + 64];

    for (unsigned b = 1; b <= 32; ++b) {
        auto input = randomValues(128, b);
        uint32_t out_a[128], out_b[128];

        // SSE pack → scalar unpack
        abpfor::packI4(input.data(), buf, b);
        scalar_ref::unpackI4(buf, out_a, b);
        for (unsigned i = 0; i < 128; ++i)
            CHECK(out_a[i] == input[i], "I4 SSE→scalar b=%u i=%u: got %u want %u", b, i, out_a[i], input[i]);

        // Scalar pack → SSE unpack
        std::memset(buf, 0, sizeof(buf));
        scalar_ref::packI4(input.data(), buf, b);
        abpfor::unpackI4(buf, out_b, b);
        for (unsigned i = 0; i < 128; ++i)
            CHECK(out_b[i] == input[i], "I4 scalar→SSE b=%u i=%u: got %u want %u", b, i, out_b[i], input[i]);
    }
}

// ---------------------------------------------------------------------------
// Test 2: I8 format compatibility (AVX2 pack → scalar unpack, scalar pack → AVX2 unpack)
// ---------------------------------------------------------------------------
static void testI8Compatibility()
{
    printf("  I8 format compatibility (B=1..32)...\n");
    alignas(64) uint8_t buf[256 * 4 + 64];

    for (unsigned b = 1; b <= 32; ++b) {
        auto input = randomValues(256, b);
        uint32_t out_a[256], out_b[256];

        // AVX2 pack → scalar unpack
        abpfor::packI8(input.data(), buf, b);
        scalar_ref::unpackI8(buf, out_a, b);
        for (unsigned i = 0; i < 256; ++i)
            CHECK(out_a[i] == input[i], "I8 AVX2→scalar b=%u i=%u: got %u want %u", b, i, out_a[i], input[i]);

        // Scalar pack → AVX2 unpack
        std::memset(buf, 0, sizeof(buf));
        scalar_ref::packI8(input.data(), buf, b);
        abpfor::unpackI8(buf, out_b, b);
        for (unsigned i = 0; i < 256; ++i)
            CHECK(out_b[i] == input[i], "I8 scalar→AVX2 b=%u i=%u: got %u want %u", b, i, out_b[i], input[i]);
    }
}

// ---------------------------------------------------------------------------
// Test 3: Fused decode compatibility (SSE pack → scalar fusedDecodeI4)
// ---------------------------------------------------------------------------
static void testFusedDecodeCompatibility()
{
    printf("  Fused decode compatibility...\n");
    alignas(64) uint8_t buf[128 * 4 + 64];

    for (unsigned b = 1; b <= 31; ++b) {
        auto input = randomValues(128, b);

        // Pack with SSE
        abpfor::packI4(input.data(), buf, b);

        // Decode with scalar, no delta no patch
        {
            uint32_t out[128];
            uint32_t carry = 0;
            scalar_ref::fusedDecodeI4<false, false>(buf, out, b, carry, nullptr, nullptr);
            for (unsigned i = 0; i < 128; ++i)
                CHECK(out[i] == input[i], "fusedI4 noDelta noPatch b=%u i=%u", b, i);
        }

        // Decode with scalar, delta (need sorted input)
        {
            // Create sorted data for delta encoding
            auto sorted = randomValues(128, b);
            std::sort(sorted.begin(), sorted.end());
            // Compute gaps-minus-1 (what gets packed in delta mode)
            uint32_t gaps[128];
            uint32_t prev = 0;
            for (unsigned i = 0; i < 128; ++i) {
                // delta encoding stores (val - prev - 1), decoded as carry += gap + 1
                gaps[i] = sorted[i] - prev - 1;
                prev = sorted[i];
            }
            // But gaps might not fit in b bits for arbitrary sorted data.
            // Use values that guarantee gaps fit in b bits.
            uint32_t mask = (1u << b) - 1u;
            uint32_t running = 0;
            for (unsigned i = 0; i < 128; ++i) {
                gaps[i] = rng() & mask;
                running += gaps[i] + 1;
                sorted[i] = running;
            }

            abpfor::packI4(gaps, buf, b);
            uint32_t out[128];
            uint32_t carry = 0;
            scalar_ref::fusedDecodeI4<true, false>(buf, out, b, carry, nullptr, nullptr);
            for (unsigned i = 0; i < 128; ++i)
                CHECK(out[i] == sorted[i], "fusedI4 delta b=%u i=%u: got %u want %u", b, i, out[i], sorted[i]);
        }

        // Decode with scalar, patch (no delta)
        {
            auto base = randomValues(128, b);
            // Set some outlier bits
            uint64_t bitmap[2] = {0, 0};
            std::vector<uint32_t> residuals;
            for (unsigned i = 0; i < 128; i += 7) {
                bitmap[i / 64] |= uint64_t(1) << (i % 64);
                residuals.push_back(rng() & 0xFFu); // residual fits in remaining bits
            }

            abpfor::packI4(base.data(), buf, b);
            uint32_t out[128];
            uint32_t carry = 0;
            scalar_ref::fusedDecodeI4<false, true>(buf, out, b, carry, bitmap, residuals.data());

            // Verify: out[i] = base[i] + (residual << b) for patched, base[i] otherwise
            unsigned ri = 0;
            for (unsigned i = 0; i < 128; ++i) {
                uint32_t expected = base[i];
                if (bitmap[i / 64] & (uint64_t(1) << (i % 64)))
                    expected += residuals[ri++] << b;
                CHECK(out[i] == expected, "fusedI4 patch b=%u i=%u: got %u want %u", b, i, out[i], expected);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Test 4: Full scalar roundtrip (simulating ARM/non-x86 path)
// ---------------------------------------------------------------------------
static void testScalarRoundtrip()
{
    printf("  Scalar-only roundtrip...\n");
    alignas(64) uint8_t buf[256 * 4 + 64];

    // I4 roundtrip
    for (unsigned b = 0; b <= 32; ++b) {
        auto input = randomValues(128, b);
        uint32_t out[128];
        scalar_ref::packI4(input.data(), buf, b);
        scalar_ref::unpackI4(buf, out, b);
        for (unsigned i = 0; i < 128; ++i)
            CHECK(out[i] == input[i], "scalar I4 roundtrip b=%u i=%u", b, i);
    }

    // I8 roundtrip
    for (unsigned b = 0; b <= 32; ++b) {
        auto input = randomValues(256, b);
        uint32_t out[256];
        scalar_ref::packI8(input.data(), buf, b);
        scalar_ref::unpackI8(buf, out, b);
        for (unsigned i = 0; i < 256; ++i)
            CHECK(out[i] == input[i], "scalar I8 roundtrip b=%u i=%u", b, i);
    }

    // I4 fused decode roundtrip (delta + patch)
    for (unsigned b = 1; b <= 31; ++b) {
        uint32_t mask = (1u << b) - 1u;
        uint32_t gaps[128];
        uint32_t sorted[128];
        uint64_t bitmap[2] = {0, 0};
        std::vector<uint32_t> residuals;

        uint32_t running = 0;
        for (unsigned i = 0; i < 128; ++i) {
            gaps[i] = rng() & mask;
            running += gaps[i] + 1;
            sorted[i] = running;
        }
        // Add patches on top
        for (unsigned i = 0; i < 128; i += 11) {
            bitmap[i / 64] |= uint64_t(1) << (i % 64);
            uint32_t res = rng() & 0xFFu;
            residuals.push_back(res);
        }
        // Recompute expected with patches applied during delta decode
        uint32_t expected[128];
        uint32_t carry_exp = 0;
        unsigned ri = 0;
        for (unsigned i = 0; i < 128; ++i) {
            uint32_t val = gaps[i];
            if (bitmap[i / 64] & (uint64_t(1) << (i % 64)))
                val += residuals[ri++] << b;
            carry_exp += val + 1;
            expected[i] = carry_exp;
        }

        scalar_ref::packI4(gaps, buf, b);
        uint32_t out[128];
        uint32_t carry = 0;
        scalar_ref::fusedDecodeI4<true, true>(buf, out, b, carry, bitmap, residuals.data());
        for (unsigned i = 0; i < 128; ++i)
            CHECK(out[i] == expected[i], "scalar fusedI4 delta+patch b=%u i=%u: got %u want %u", b, i, out[i], expected[i]);
    }

    // I8 fused decode roundtrip (delta, no patch)
    for (unsigned b = 1; b <= 31; ++b) {
        uint32_t mask = (1u << b) - 1u;
        uint32_t gaps[256];
        uint32_t sorted[256];
        uint32_t running = 0;
        for (unsigned i = 0; i < 256; ++i) {
            gaps[i] = rng() & mask;
            running += gaps[i] + 1;
            sorted[i] = running;
        }

        scalar_ref::packI8(gaps, buf, b);
        uint32_t out[256];
        uint32_t carry = 0;
        scalar_ref::fusedDecodeI8<true, false>(buf, out, b, carry, nullptr, nullptr);
        for (unsigned i = 0; i < 256; ++i)
            CHECK(out[i] == sorted[i], "scalar fusedI8 delta b=%u i=%u: got %u want %u", b, i, out[i], sorted[i]);
    }
}

// ---------------------------------------------------------------------------
int main()
{
    printf("Cross-platform compatibility tests:\n");
    testI4Compatibility();
    testI8Compatibility();
    testFusedDecodeCompatibility();
    testScalarRoundtrip();

    if (failures == 0)
        printf("ALL PASSED\n");
    else
        printf("FAILURES: %d\n", failures);
    return failures ? 1 : 0;
}
