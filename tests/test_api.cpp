// Layer 6 tests: public API end-to-end roundtrip.
// Tests abpfor::b128 / abpfor::b256 namespace functions.

#include <abpfor.h>
#include "detail/codec.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

static int failures = 0;

#if ABPFOR_ARCH_X86
#ifndef ABPFOR_HAS_FUSED_I8_BITMAP_ENCODE
#error "production fused I8 bitmap encoder is required on x86"
#endif
#endif

#define CHECK(cond, ...)                                    \
    do {                                                    \
        if (!(cond)) {                                      \
            printf("  FAIL %s:%d: ", __FILE__, __LINE__);   \
            printf(__VA_ARGS__);                             \
            printf("\n");                                    \
            ++failures;                                     \
        }                                                   \
    } while (0)

// --- Generic roundtrip helpers ---

template <typename T>
using EncodeFn = size_t(*)(const T*, unsigned, uint8_t*);
template <typename T>
using DecodeFn = size_t(*)(const uint8_t*, unsigned, T*);
template <typename T>
using EncodeDeltaFn = size_t(*)(const T*, unsigned, uint8_t*, T);
template <typename T>
using DecodeDeltaFn = size_t(*)(const uint8_t*, unsigned, T*, T);

template <typename T>
static void rt(EncodeFn<T> enc, DecodeFn<T> dec, const T* data, unsigned n, const char* label)
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
                  label, i, static_cast<unsigned long long>(out[i]), static_cast<unsigned long long>(data[i]));
            break;
        }
    }
}

template <typename T>
static void rtd(EncodeDeltaFn<T> enc, DecodeDeltaFn<T> dec, const T* data, unsigned n, const char* label, T start)
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
                  label, i, static_cast<unsigned long long>(out[i]), static_cast<unsigned long long>(data[i]));
            break;
        }
    }
}

// --- b128 ---

static void test_b128()
{
    printf("test_b128...\n");
    std::mt19937 rng(42);

    // Various sizes
    for (unsigned n : {1u, 7u, 32u, 100u, 127u, 128u, 200u, 256u, 500u})
    {
        std::vector<uint32_t> data(n);
        for (auto& v : data) v = rng() & 0xFF;

        char label[64];
        snprintf(label, sizeof(label), "b128-n%u", n);
        rt<uint32_t>(abpfor::b128::encode, abpfor::b128::decode, data.data(), n, label);
    }

    // Delta1
    {
        uint32_t sorted[200];
        uint32_t v = 50;
        for (unsigned i = 0; i < 200; ++i) { v += 1 + static_cast<uint32_t>(rng() % 10); sorted[i] = v; }
        rtd<uint32_t>(abpfor::b128::encodeDelta1, abpfor::b128::decodeDelta1, sorted, 200, "b128-d1-200", uint32_t(50));
    }

    // Delta0
    {
        uint32_t sorted[200];
        uint32_t v = 50;
        for (unsigned i = 0; i < 200; ++i) { v += static_cast<uint32_t>(rng() % 10); sorted[i] = v; }
        rtd<uint32_t>(abpfor::b128::encodeDelta0, abpfor::b128::decodeDelta0, sorted, 200, "b128-d0-200", uint32_t(50));
    }

    // With tail
    {
        uint32_t data[300];
        for (auto& v : data) v = rng() & 0xFFF;
        rt<uint32_t>(abpfor::b128::encode, abpfor::b128::decode, data, 300, "b128-300-tail");
    }

    // With outliers
    {
        uint32_t data[128];
        for (auto& v : data) v = rng() & 0xF;
        data[10] = 0xABCD;
        data[50] = 0x1234;
        data[90] = 0xFFFF;
        rt<uint32_t>(abpfor::b128::encode, abpfor::b128::decode, data, 128, "b128-outliers");
    }

    // Delta + outliers
    {
        uint32_t sorted[128];
        uint32_t v2 = 0;
        for (unsigned i = 0; i < 128; ++i)
        {
            v2 += (rng() % 100 < 5) ? static_cast<uint32_t>(1000 + rng() % 5000) : static_cast<uint32_t>(1 + rng() % 3);
            sorted[i] = v2;
        }
        rtd<uint32_t>(abpfor::b128::encodeDelta1, abpfor::b128::decodeDelta1, sorted, 128, "b128-d1-outliers", uint32_t(0));
    }
}

// --- b256 ---

static void test_b256()
{
    printf("test_b256...\n");
    std::mt19937 rng(99);

    // Exact block
    {
        uint32_t data[256];
        for (auto& v : data) v = rng() & 0xFFF;
        rt<uint32_t>(abpfor::b256::encode, abpfor::b256::decode, data, 256, "b256-256");
    }

    // Multiple blocks
    {
        uint32_t data[768];
        for (auto& v : data) v = rng() & 0xFFF;
        rt<uint32_t>(abpfor::b256::encode, abpfor::b256::decode, data, 768, "b256-768");
    }

    // With tail
    {
        uint32_t data[600];
        for (auto& v : data) v = rng() & 0xFFF;
        rt<uint32_t>(abpfor::b256::encode, abpfor::b256::decode, data, 600, "b256-600-tail");
    }

    // Delta
    {
        uint32_t sorted[600];
        uint32_t v = 0;
        for (unsigned i = 0; i < 600; ++i) { v += 1 + static_cast<uint32_t>(rng() % 5); sorted[i] = v; }
        rtd<uint32_t>(abpfor::b256::encodeDelta1, abpfor::b256::decodeDelta1, sorted, 600, "b256-d1-600", uint32_t(0));
    }
}

static void test_b256_delta1_b1_exact()
{
    alignas(32) uint32_t in[512], out[512];
    uint32_t v = 1000;
    for (unsigned i = 0; i < 512; ++i)
    {
        v += 1u + (i & 1u);
        in[i] = v;
    }

    std::vector<uint8_t> b(abpfor::maxCompressedSize<uint32_t>(512));
    size_t n = abpfor::b256::encodeDelta1(in, 512, b.data(), 1000u);

    const uint8_t* block = b.data();
    CHECK((block[0] & abpfor::hdr::kTypeMask) == abpfor::hdr::kBitpackOnly,
          "b256 delta1 B=1 first block type");
    CHECK((block[0] & abpfor::hdr::kBitsMask) == 1,
          "b256 delta1 B=1 first block width");
    size_t firstBytes = abpfor::b256::decodeDelta1(block, 256, out, 1000u);

    block += firstBytes;
    CHECK((block[0] & abpfor::hdr::kTypeMask) == abpfor::hdr::kBitpackOnly,
          "b256 delta1 B=1 second block type");
    CHECK((block[0] & abpfor::hdr::kBitsMask) == 1,
          "b256 delta1 B=1 second block width");
    size_t secondBytes = abpfor::b256::decodeDelta1(block, 256, out + 256, in[255]);

    CHECK(firstBytes + secondBytes == n, "b256 delta1 B=1 consumed size");
    CHECK(std::memcmp(in, out, sizeof(in)) == 0, "b256 delta1 B=1 exact roundtrip");
}

// --- 64-bit ---

static void test_64bit()
{
    printf("test_64bit...\n");
    std::mt19937_64 rng(123);

    // b128 64-bit
    {
        uint64_t data[200];
        for (auto& v : data) v = rng() & 0xFFFF;
        rt<uint64_t>(abpfor::b128::encode, abpfor::b128::decode, data, 200, "u64-b128-200");
    }

    // b128 delta 64-bit
    {
        uint64_t sorted[200];
        uint64_t v = 1000000;
        for (unsigned i = 0; i < 200; ++i) { v += 1 + (rng() % 100); sorted[i] = v; }
        rtd<uint64_t>(abpfor::b128::encodeDelta1, abpfor::b128::decodeDelta1, sorted, 200, "u64-b128-d1", uint64_t(1000000));
    }
}

// --- Edge cases ---

static void test_edges()
{
    printf("test_edges...\n");

    // All zeros
    {
        uint32_t z[128] = {};
        rt<uint32_t>(abpfor::b128::encode, abpfor::b128::decode, z, 128, "zeros-b128");
    }

    // All constant
    {
        uint32_t c[256];
        std::fill_n(c, 256, 42u);
        rt<uint32_t>(abpfor::b128::encode, abpfor::b128::decode, c, 128, "const-b128");
        rt<uint32_t>(abpfor::b256::encode, abpfor::b256::decode, c, 256, "const-b256");
    }

    // Single element
    {
        uint32_t one = 12345;
        rt<uint32_t>(abpfor::b128::encode, abpfor::b128::decode, &one, 1, "single");
    }

    // n=1 delta
    {
        uint32_t one = 100;
        rtd<uint32_t>(abpfor::b128::encodeDelta1, abpfor::b128::decodeDelta1, &one, 1, "single-d1", uint32_t(50));
    }
}

// --- Sweep ---

static void test_sweep()
{
    printf("test_sweep...\n");
    std::mt19937 rng(42);

    for (unsigned bw : {2u, 4u, 8u, 16u, 24u, 32u})
    {
        for (unsigned excPct : {0u, 5u, 15u, 30u})
        {
            uint32_t data[128];
            uint32_t baseMask = bw == 32 ? ~0u : (1u << bw) - 1u;
            for (unsigned i = 0; i < 128; ++i)
            {
                if (excPct > 0 && (rng() % 100) < excPct)
                    data[i] = (bw < 32 ? (1u << bw) : 0u) + static_cast<uint32_t>(rng() & 0xFFFF);
                else
                    data[i] = static_cast<uint32_t>(rng() & baseMask);
            }

            char label[64];
            snprintf(label, sizeof(label), "sweep-b%u-exc%u%%-b128", bw, excPct);
            rt<uint32_t>(abpfor::b128::encode, abpfor::b128::decode, data, 128, label);
        }
    }
}

static void test_b128_public_fixed_i4_bitmap_dispatch()
{
    printf("test_b128_public_fixed_i4_bitmap_dispatch...\n");
    constexpr unsigned N = 128;
    const unsigned positions[] = {0, 1, 7, 15, 31, 47, 62, 63, 64, 65, 79, 95, 111, 126, 127};
    alignas(32) uint32_t residuals[N], input[N], out[N];
    uint8_t buf[4096];

    for (unsigned mode = 0; mode < 3; ++mode)
    {
        for (unsigned i = 0; i < N; ++i) residuals[i] = (i * 37u + 11u) & 0xffu;
        for (unsigned pos : positions) residuals[pos] |= ((1u << 24) - 1u) << 8;

        const uint32_t start = 1234;
        uint32_t running = start;
        for (unsigned i = 0; i < N; ++i)
        {
            if (mode == 0) input[i] = residuals[i];
            else
            {
                running += residuals[i] + (mode == 2 ? 1u : 0u);
                input[i] = running;
            }
        }

        size_t written;
        if (mode == 0) written = abpfor::b128::encodeBlock(input, buf);
        else if (mode == 1) written = abpfor::b128::encodeBlockDelta0(input, buf, start);
        else written = abpfor::b128::encodeBlockDelta1(input, buf, start);

        CHECK((buf[0] & abpfor::hdr::kTypeMask) == abpfor::hdr::kBitmapOutlier,
              "b128 public mode=%u uses bitmap", mode);
        CHECK((buf[0] & abpfor::hdr::kBitsMask) == 8, "b128 public mode=%u b=8", mode);
        CHECK(buf[1] == 24, "b128 public mode=%u pb=24", mode);

        size_t consumed;
        if (mode == 0) consumed = abpfor::b128::decodeBlock(buf, out);
        else if (mode == 1) consumed = abpfor::b128::decodeBlockDelta0(buf, out, start);
        else consumed = abpfor::b128::decodeBlockDelta1(buf, out, start);

        CHECK(consumed == written, "b128 public mode=%u consumed=%zu written=%zu", mode, consumed, written);
        CHECK(std::memcmp(out, input, sizeof(out)) == 0, "b128 public mode=%u roundtrip", mode);
    }
}

template <unsigned BlockSize, typename T, bool MinusOne>
static void check_all_zero_direct(unsigned n)
{
    std::vector<T> residuals(n, T{});
    uint8_t encoded[16] = {};
    size_t written;
    if constexpr (BlockSize == 128)
        written = abpfor::detail::encodeBlockI4(residuals.data(), n, encoded);
    else
        written = abpfor::detail::encodeBlockI8(residuals.data(), n, encoded);
    CHECK(written == 1, "direct I%u u%zu d%u n=%u written=%zu", BlockSize / 32,
          sizeof(T) * 8, MinusOne ? 1u : 0u, n, written);
    CHECK(encoded[0] == 0xc0, "direct I%u u%zu d%u n=%u wire=%02x", BlockSize / 32,
          sizeof(T) * 8, MinusOne ? 1u : 0u, n, unsigned(encoded[0]));
    std::vector<T> out(n, T(~T{}));
    T carry = T(37);
    size_t consumed;
    if constexpr (BlockSize == 128)
        consumed = abpfor::detail::decodeBlockI4<T, MinusOne>(encoded, n, out.data(), carry);
    else
        consumed = abpfor::detail::decodeBlockI8<T, MinusOne>(encoded, n, out.data(), carry);

    CHECK(consumed == 1, "direct I%u u%zu d%u n=%u consumed=%zu", BlockSize / 32,
          sizeof(T) * 8, MinusOne ? 1u : 0u, n, consumed);
    for (unsigned i = 0; i < n; ++i)
    {
        T expected = MinusOne ? T(37 + i + 1) : T(37);
        CHECK(out[i] == expected, "direct I%u u%zu d%u n=%u i=%u", BlockSize / 32,
              sizeof(T) * 8, MinusOne ? 1u : 0u, n, i);
    }
    CHECK(carry == (MinusOne ? T(37 + n) : T(37)), "direct I%u u%zu d%u n=%u carry",
          BlockSize / 32, sizeof(T) * 8, MinusOne ? 1u : 0u, n);
}

template <abpfor::Layout L, abpfor::Delta D, typename T>
static void check_all_zero_public(unsigned n)
{
    constexpr unsigned Full = L == abpfor::Layout::Interleave4 ? 128 : 256;
    const T start = T(37);
    std::vector<T> input(n), blockOut(Full), streamOut(n);
    for (unsigned i = 0; i < n; ++i)
        input[i] = D == abpfor::Delta::Delta1 ? T(start + i + 1) : start;
    std::vector<uint8_t> encoded(abpfor::maxCompressedSize<T>(n));
    size_t written = abpfor::encode<L, D>(std::span<const T>(input), encoded.data(), start);
    const size_t encodedBlocks = (n + Full - 1) / Full;
    CHECK(written == encodedBlocks, "public I%u u%zu d%u n=%u written=%zu blocks=%zu", Full / 32,
          sizeof(T) * 8, D == abpfor::Delta::Delta1 ? 1u : 0u, n, written, encodedBlocks);
    for (size_t block = 0; block < encodedBlocks; ++block)
        CHECK(encoded[block] == 0xc0, "public stream I%u u%zu d%u n=%u block=%zu wire=%02x", Full / 32,
              sizeof(T) * 8, D == abpfor::Delta::Delta1 ? 1u : 0u, n, block, unsigned(encoded[block]));
    size_t consumed = abpfor::decode<L, D>(encoded.data(), std::span<T>(streamOut), start);
    CHECK(consumed == written, "public stream I%u u%zu d%u n=%u consumed", Full / 32,
          sizeof(T) * 8, D == abpfor::Delta::Delta1 ? 1u : 0u, n);
    CHECK(streamOut == input, "public stream I%u u%zu d%u n=%u output", Full / 32,
          sizeof(T) * 8, D == abpfor::Delta::Delta1 ? 1u : 0u, n);
    CHECK(streamOut.back() == (D == abpfor::Delta::Delta1 ? T(start + n) : start),
          "public stream I%u u%zu d%u n=%u carry", Full / 32, sizeof(T) * 8,
          D == abpfor::Delta::Delta1 ? 1u : 0u, n);

    if (n == Full)
    {
        size_t blockConsumed;
        if constexpr (Full == 128 && D == abpfor::Delta::Delta1)
            blockConsumed = abpfor::b128::decodeBlockDelta1(encoded.data(), blockOut.data(), start);
        else if constexpr (Full == 128)
            blockConsumed = abpfor::b128::decodeBlockDelta0(encoded.data(), blockOut.data(), start);
        else if constexpr (D == abpfor::Delta::Delta1)
            blockConsumed = abpfor::b256::decodeBlockDelta1(encoded.data(), blockOut.data(), start);
        else
            blockConsumed = abpfor::b256::decodeBlockDelta0(encoded.data(), blockOut.data(), start);
        CHECK(blockConsumed == written, "public block I%u u%zu d%u consumed", Full / 32,
              sizeof(T) * 8, D == abpfor::Delta::Delta1 ? 1u : 0u);
        CHECK(encoded[0] == 0xc0, "public block I%u u%zu d%u wire=%02x", Full / 32,
              sizeof(T) * 8, D == abpfor::Delta::Delta1 ? 1u : 0u, unsigned(encoded[0]));
        CHECK(std::equal(input.begin(), input.end(), blockOut.begin()),
              "public block I%u u%zu d%u output", Full / 32, sizeof(T) * 8,
              D == abpfor::Delta::Delta1 ? 1u : 0u);
        CHECK(blockOut[Full - 1] == (D == abpfor::Delta::Delta1 ? T(start + Full) : start),
              "public block I%u u%zu d%u carry", Full / 32, sizeof(T) * 8,
              D == abpfor::Delta::Delta1 ? 1u : 0u);
    }
}

template <unsigned BlockSize, abpfor::Layout L, typename T>
static void check_all_zero_matrix()
{
    for (unsigned n : {1u, 7u, BlockSize - 1, BlockSize})
    {
        check_all_zero_direct<BlockSize, T, false>(n);
        check_all_zero_direct<BlockSize, T, true>(n);
        check_all_zero_public<L, abpfor::Delta::Delta0, T>(n);
        check_all_zero_public<L, abpfor::Delta::Delta1, T>(n);
    }
    check_all_zero_public<L, abpfor::Delta::Delta0, T>(BlockSize * 2 + 7);
    check_all_zero_public<L, abpfor::Delta::Delta1, T>(BlockSize * 2 + 7);
}

static void test_all_zero_delta_wrappers()
{
    printf("test_all_zero_delta_wrappers...\n");
    check_all_zero_matrix<128, abpfor::Layout::Interleave4, uint32_t>();
    check_all_zero_matrix<128, abpfor::Layout::Interleave4, uint64_t>();
    check_all_zero_matrix<256, abpfor::Layout::Interleave8, uint32_t>();
    check_all_zero_matrix<256, abpfor::Layout::Interleave8, uint64_t>();
}

static size_t generic_i8_bitmap_reference(const uint32_t* input, uint8_t* output, unsigned b)
{
    constexpr unsigned N = 256;
    const uint32_t baseMask = abpfor::mask<uint32_t>(b);
    uint8_t positions[N];
    alignas(32) uint32_t residuals[N], masked[N];
    uint32_t outlierOr;
    const unsigned oc = abpfor::detail::collectOutliers(input, N, b, baseMask, positions, residuals, outlierOr);
    for (unsigned i = 0; i < N; ++i) masked[i] = input[i] & baseMask;

    uint8_t* op = output;
    *op++ = abpfor::hdr::kBitmapOutlier | static_cast<uint8_t>(b);
    *op++ = static_cast<uint8_t>(abpfor::bitwidth(outlierOr) - b);
    std::memset(op, 0, N / 8);
    for (unsigned i = 0; i < oc; ++i) op[positions[i] >> 3] |= uint8_t(1) << (positions[i] & 7);
    op += N / 8;
    op = abpfor::pack(residuals, oc, op, abpfor::bitwidth(outlierOr) - b);
    op = abpfor::detail::packInterleaved<uint32_t, N>(masked, op, b, abpfor::packI8);
    return static_cast<size_t>(op - output);
}

static void make_i8_bitmap_case(std::array<uint32_t, 256>& residuals, unsigned requestedB,
                                unsigned requestedPb, unsigned requestedOc, std::mt19937& rng)
{
    const uint32_t baseMask = abpfor::mask<uint32_t>(requestedB);
    std::uniform_int_distribution<uint32_t> base(0, baseMask);
    for (uint32_t& value : residuals) value = base(rng);

    std::array<unsigned, 256> positions;
    std::iota(positions.begin(), positions.end(), 0u);
    std::shuffle(positions.begin(), positions.end(), rng);
    const uint64_t highLimit = requestedPb == 32 ? uint64_t(~0u) : ((uint64_t(1) << requestedPb) - 1);
    std::uniform_int_distribution<uint64_t> high(1, highLimit);
    for (unsigned i = 0; i < requestedOc; ++i)
        residuals[positions[i]] = static_cast<uint32_t>((high(rng) << requestedB) |
                                                        (residuals[positions[i]] & baseMask));
}

static void test_b256_bitmap_encode_byte_identity()
{
    printf("test_b256_bitmap_encode_byte_identity...\n");
    constexpr unsigned N = 256;
    constexpr unsigned Attempts = 12288;
    static constexpr unsigned Counts[] = {16, 31, 32, 63, 64, 65, 96, 127, 160, 192, 224, 255};
    std::mt19937 rng(0x8b17u);
    std::array<uint32_t, N> residuals{}, input{};
    std::array<uint8_t, 2048> expected{}, actual{};
    unsigned checked = 0;

    for (unsigned attempt = 0; attempt < Attempts; ++attempt)
    {
        const unsigned requestedB = attempt % 32;
        const unsigned requestedPb = 1 + ((attempt / 32) % (32 - requestedB));
        make_i8_bitmap_case(residuals, requestedB, requestedPb,
                            Counts[(attempt / 1024 + attempt) % std::size(Counts)], rng);
        unsigned pbx;
        const unsigned b = abpfor::optimalWidth(residuals.data(), N, &pbx);
        if (pbx != 33) continue;
        const size_t expectedBytes = generic_i8_bitmap_reference(residuals.data(), expected.data(), b);

        for (unsigned mode = 0; mode < 3; ++mode)
        {
            const uint32_t start = 1234;
            uint32_t running = start;
            for (unsigned i = 0; i < N; ++i)
            {
                if (mode == 0) input[i] = residuals[i];
                else
                {
                    running += residuals[i] + (mode == 2 ? 1u : 0u);
                    input[i] = running;
                }
            }

            size_t actualBytes;
            if (mode == 0) actualBytes = abpfor::b256::encodeBlock(input.data(), actual.data());
            else if (mode == 1) actualBytes = abpfor::b256::encodeBlockDelta0(input.data(), actual.data(), start);
            else actualBytes = abpfor::b256::encodeBlockDelta1(input.data(), actual.data(), start);
            CHECK(actualBytes == expectedBytes, "I8 bitmap property attempt=%u mode=%u size", attempt, mode);
            CHECK(std::memcmp(actual.data(), expected.data(), expectedBytes) == 0,
                  "I8 bitmap property attempt=%u mode=%u bytes", attempt, mode);
        }
        ++checked;
    }
    CHECK(checked >= 1000, "I8 bitmap property checked=%u", checked);
}

// --- Block-level API ---

static void test_encodeBlock_public()
{
    printf("test_encodeBlock_public...\n");
    uint32_t data[256];
    for (unsigned i = 0; i < 256; ++i) data[i] = i * 3 + 7;

    uint8_t buf[4096];
    uint32_t out[256];

    // b128 Delta1
    {
        size_t enc = abpfor::b128::encodeBlockDelta1(data, buf, 0u);
        CHECK(enc > 0, "b128::encodeBlockDelta1 wrote bytes");
        size_t dec = abpfor::b128::decodeBlockDelta1(buf, out, 0u);
        CHECK(dec == enc, "decodeBlockDelta1 size matches");
        CHECK(memcmp(data, out, 128 * sizeof(uint32_t)) == 0, "b128::encodeBlockDelta1 roundtrip");
    }

    // b128 None
    {
        size_t enc = abpfor::b128::encodeBlock(data, buf);
        size_t dec = abpfor::b128::decodeBlock(buf, out);
        CHECK(dec == enc, "b128::decodeBlock size matches");
        CHECK(memcmp(data, out, 128 * sizeof(uint32_t)) == 0, "b128::encodeBlock roundtrip");
    }

    // b256 Delta1
    {
        size_t enc = abpfor::b256::encodeBlockDelta1(data, buf, 0u);
        size_t dec = abpfor::b256::decodeBlockDelta1(buf, out, 0u);
        CHECK(dec == enc, "b256::decodeBlockDelta1 size matches");
        CHECK(memcmp(data, out, 256 * sizeof(uint32_t)) == 0, "b256::encodeBlockDelta1 roundtrip");
    }
}

int main()
{
    test_b128();
    test_b256();
    test_b256_delta1_b1_exact();
    test_64bit();
    test_edges();
    test_sweep();
    test_b128_public_fixed_i4_bitmap_dispatch();
    test_all_zero_delta_wrappers();
    test_b256_bitmap_encode_byte_identity();
    test_encodeBlock_public();

    if (failures == 0)
        printf("All API tests passed.\n");
    else
        printf("%d API test(s) FAILED.\n", failures);

    return failures > 0 ? 1 : 0;
}
