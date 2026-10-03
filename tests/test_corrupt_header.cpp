// Layer 7 tests: a corrupt or hostile block header must be rejected instead of
// being dispatched through a wild function pointer.
//
// Two header fields index fixed-size function-pointer tables:
//   * the 6-bit `b` field (0..63) indexes 33-entry tables on the 32-bit path
//   * the residual width `pb`, a full byte (0..255), indexes the same tables on
//     every path, including the 64-bit one where they have 65 entries
// A correct encoder only ever writes b <= 32 and pb <= 32 (or 64), so only
// corrupt or hostile input reaches the cases below. Decoding such a block must
// return 0 -- nothing consumed -- rather than call a pointer read from past the
// end of a table.
//
// Each decode is handed a large zero-filled buffer, so a failure here can only
// come from the dispatch index, never from reading past the input.

#include <abpfor.h>

#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <vector>

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

namespace
{

// Room for any single block plus the decoders' word-level lookahead.
constexpr size_t kBufferBytes = 4096;

// A zero-filled input with `header` written at the front.
std::vector<uint8_t> makeBlock(std::initializer_list<unsigned> header)
{
    std::vector<uint8_t> buf(kBufferBytes, 0);
    size_t i = 0;
    for (unsigned byte : header)
        buf[i++] = static_cast<uint8_t>(byte);
    return buf;
}

// --- 32-bit: the 6-bit bit-width field of the header byte ---
void test_bitpack_width_u32()
{
    uint32_t out[256] = {};

    for (unsigned b = 33; b <= 63; ++b)
    {
        const std::vector<uint8_t> in = makeBlock({0x00u | b}); // type=00, bitpack only

        CHECK(abpfor::b256::decodeBlock(in.data(), out) == 0,
              "b256::decodeBlock rejects bitpack width b=%u", b);
        CHECK(abpfor::b256::decodeBlockDelta1(in.data(), out, 0u) == 0,
              "b256::decodeBlockDelta1 rejects bitpack width b=%u", b);
        CHECK(abpfor::b256::decodeTail(in.data(), 4u, out) == 0,
              "b256::decodeTail rejects bitpack width b=%u", b);
        CHECK(abpfor::b256::decodeTailDelta1(in.data(), 4u, out, 0u) == 0,
              "b256::decodeTailDelta1 rejects bitpack width b=%u", b);
        CHECK(abpfor::b128::decodeBlock(in.data(), out) == 0,
              "b128::decodeBlock rejects bitpack width b=%u", b);
    }

    // Positive control: the largest legal width must still decode. Over-
    // rejecting would break the encoder's own output.
    const std::vector<uint8_t> ok = makeBlock({0x00u | 32u});
    CHECK(abpfor::b256::decodeBlock(ok.data(), out) != 0, "b256::decodeBlock accepts b=32");
    CHECK(abpfor::b256::decodeTail(ok.data(), 4u, out) != 0, "b256::decodeTail accepts b=32");
}

// --- Outlier blocks: `pb` is a raw byte, not a 6-bit field ---
void test_outlier_width_u32()
{
    uint32_t out[256] = {};

    for (unsigned pb = 33; pb <= 255; ++pb)
    {
        {
            const std::vector<uint8_t> in = makeBlock({0x40u | 8u, pb}); // type=01, bitmap outliers
            CHECK(abpfor::b256::decodeBlock(in.data(), out) == 0,
                  "b256::decodeBlock rejects bitmap pb=%u", pb);
            CHECK(abpfor::b256::decodeTail(in.data(), 4u, out) == 0,
                  "b256::decodeTail rejects bitmap pb=%u", pb);
        }
        {
            const std::vector<uint8_t> in = makeBlock({0x80u | 8u, pb, 1u}); // type=10, sparse outliers
            CHECK(abpfor::b256::decodeBlock(in.data(), out) == 0,
                  "b256::decodeBlock rejects sparse pb=%u", pb);
        }
    }

    // Positive control.
    const std::vector<uint8_t> ok = makeBlock({0x40u | 8u, 32u});
    CHECK(abpfor::b256::decodeBlock(ok.data(), out) != 0, "b256::decodeBlock accepts pb=32");
}

// --- Sparse outliers on the tail path dispatch on the header width ---
void test_sparse_width_u32()
{
    uint32_t out[256] = {};

    for (unsigned b = 33; b <= 63; ++b)
    {
        const std::vector<uint8_t> in = makeBlock({0x80u | b, 4u, 1u});
        CHECK(abpfor::b256::decodeBlock(in.data(), out) == 0,
              "b256::decodeBlock rejects sparse width b=%u", b);
        CHECK(abpfor::b256::decodeTail(in.data(), 4u, out) == 0,
              "b256::decodeTail rejects sparse width b=%u", b);
    }
}

// --- 64-bit: the header width is always in range, `pb` is not ---
void test_64bit()
{
    uint64_t out[256] = {};

    // b <= 63 < 64, so every header width is legal here and must still decode.
    for (unsigned b = 33; b <= 63; ++b)
    {
        const std::vector<uint8_t> in = makeBlock({0x00u | b});
        CHECK(abpfor::b256::decodeBlock(in.data(), out) != 0,
              "b256::decodeBlock(uint64_t) accepts b=%u", b);
    }

    // pb is a raw byte and can run past the 65-entry table.
    for (unsigned pb = 65; pb <= 255; ++pb)
    {
        const std::vector<uint8_t> in = makeBlock({0x40u | 8u, pb});
        CHECK(abpfor::b256::decodeBlock(in.data(), out) == 0,
              "b256::decodeBlock(uint64_t) rejects bitmap pb=%u", pb);
        CHECK(abpfor::b256::decodeTail(in.data(), 4u, out) == 0,
              "b256::decodeTail(uint64_t) rejects bitmap pb=%u", pb);
    }

    // Positive control.
    const std::vector<uint8_t> ok = makeBlock({0x40u | 8u, 64u});
    CHECK(abpfor::b256::decodeBlock(ok.data(), out) != 0, "b256::decodeBlock(uint64_t) accepts pb=64");
}

// --- Outlier position lists ---
//
// `positions` is a raw byte off the wire and `oc` (its count) is a raw byte too,
// so a corrupt block can ask the decoder to patch an element past the end of
// `out`, which holds only `n` elements. That is an out-of-bounds *write* at an
// attacker-chosen offset with an attacker-chosen value, and it is reachable
// through the tail APIs -- which is what a caller with a partial block uses, and
// what a caller that sizes its output to the element count gets.
//
// `out` is sized to exactly `n` on purpose: a larger buffer would absorb the
// write and hide it from ASan.

void test_sparse_position_past_block_end()
{
    std::vector<uint32_t> out(4, 0); // n = 4, sized exactly as a tail caller would
    // type=10 sparse, b=8, pb=4, oc=1, positions[0] = 200
    const std::vector<uint8_t> in = makeBlock({0x80u | 8u, 4u, 1u, 200u});

    CHECK(abpfor::b256::decodeTail(in.data(), 4u, out.data()) == 0,
          "decodeTail rejects sparse position 200 when n=4");
    CHECK(abpfor::b256::decodeTailDelta1(in.data(), 4u, out.data(), 0u) == 0,
          "decodeTailDelta1 rejects sparse position 200 when n=4");
}

void test_bitmap_past_block_end()
{
    std::vector<uint32_t> out(4, 0);
    // type=01 bitmap, b=8, pb=4. n=4 -> bitmapBytes = divCeil(4, 8) = 1, and bit 7
    // of that byte addresses an element past the end of the block.
    const std::vector<uint8_t> in = makeBlock({0x40u | 8u, 4u, 0x80u});

    CHECK(abpfor::b256::decodeTail(in.data(), 4u, out.data()) == 0,
          "decodeTail rejects a bitmap bit set past the block end (n=4)");
    CHECK(abpfor::b256::decodeTailDelta1(in.data(), 4u, out.data(), 0u) == 0,
          "decodeTailDelta1 rejects a bitmap bit set past the block end (n=4)");
}

void test_outlier_count_beyond_block()
{
    std::vector<uint32_t> out(4, 0);
    // oc = 250 for a block of 4 elements: more outliers than elements.
    const std::vector<uint8_t> in = makeBlock({0x80u | 8u, 4u, 250u});

    CHECK(abpfor::b256::decodeTail(in.data(), 4u, out.data()) == 0,
          "decodeTail rejects oc=250 when n=4");
}

// The same two cases on the b128 + uint64_t path. This block differs from the
// ones above: the uint64 sparse branches are inlined per block size, so the
// shared `decodeSparseOutliers` guard does not cover them. ClickHouse uses only
// b256, but leaving one inlined copy unguarded is how this class of bug comes
// back.
void test_b128_u64_sparse_positions_out_of_range()
{
    std::vector<uint64_t> out(128, 0); // a b128 block is 128 elements
    // type=10 sparse, b=8, pb=4, oc=1, positions[0] = 200 -- past the block end
    const std::vector<uint8_t> in = makeBlock({0x80u | 8u, 4u, 1u, 200u});
    CHECK(abpfor::b128::decodeBlock(in.data(), out.data()) == 0,
          "b128::decodeBlock(uint64_t) rejects sparse position 200");

    // oc = 250 for a 128-element block: overruns the 128-element residual buffer.
    const std::vector<uint8_t> tooMany = makeBlock({0x80u | 8u, 4u, 250u});
    CHECK(abpfor::b128::decodeBlock(tooMany.data(), out.data()) == 0,
          "b128::decodeBlock(uint64_t) rejects oc=250");
}

// --- Rejection must not touch the output ---
//
// `decodeBlockDelta0` / `decodeBlockDelta1` run an undelta pass over `out` after
// the block decode. If the decode is rejected they must stop there: `out` still
// holds whatever the caller had there, which for a fresh buffer is uninitialised
// memory. These cases fill `out` with a sentinel so "untouched" is observable
// without a memory sanitizer.

void test_rejection_leaves_output_untouched()
{
    const std::vector<uint8_t> in = makeBlock({0x00u | 33u}); // bitpack-only, b = 33
    constexpr uint32_t sentinel = 0xDEADBEEFu;

    std::vector<uint32_t> out(256, sentinel);
    CHECK(abpfor::b256::decodeBlockDelta1(in.data(), out.data(), 0u) == 0,
          "decodeBlockDelta1 rejects b=33");
    CHECK(abpfor::b256::decodeBlockDelta0(in.data(), out.data(), 0u) == 0,
          "decodeBlockDelta0 rejects b=33");
    for (unsigned i = 0; i < 256; ++i)
        CHECK(out[i] == sentinel, "delta wrappers must not touch out on rejection, i=%u", i);
}

// The bitmap tail path unpacks the base values into `out` before it reaches the
// merge loop, so rejecting inside the merge loop would leave `out` overwritten.
// The rejection has to happen up front instead.
void test_bitmap_rejection_leaves_output_untouched()
{
    const std::vector<uint8_t> in = makeBlock({0x40u | 8u, 4u, 0x80u}); // bit 7 set, n = 4
    constexpr uint32_t sentinel = 0xABCDEF01u;

    std::vector<uint32_t> out(4, sentinel);
    CHECK(abpfor::b256::decodeTail(in.data(), 4u, out.data()) == 0,
          "decodeTail rejects a bitmap bit past the block end");
    CHECK(abpfor::b256::decodeTailDelta1(in.data(), 4u, out.data(), 0u) == 0,
          "decodeTailDelta1 rejects a bitmap bit past the block end");
    for (unsigned i = 0; i < 4; ++i)
        CHECK(out[i] == sentinel, "bitmap rejection must not touch out, i=%u", i);
}

// The stream APIs advance the element position independently of the byte count,
// so a rejected block used to leave the cursor in place and decode the same bad
// header again into the next slots, returning a non-zero total.
void test_stream_propagates_rejection()
{
    std::vector<uint8_t> in(kBufferBytes, 0);
    in[0] = 0x00u | 33u; // first b128 block has an invalid bitpack width
    std::vector<uint32_t> out(256, 0);

    CHECK(abpfor::b128::decode(in.data(), 256u, out.data()) == 0,
          "b128::decode propagates a rejected block");
    CHECK(abpfor::b128::decodeDelta0(in.data(), 256u, out.data(), 0u) == 0,
          "b128::decodeDelta0 propagates a rejected block");
    CHECK(abpfor::b128::decodeDelta1(in.data(), 256u, out.data(), 0u) == 0,
          "b128::decodeDelta1 propagates a rejected block");
}

// The b128 + uint32_t path, which is what the shared `decodeSparseOutliers` guard
// protects. Note the guard cannot fire for a b256 full block: there n == 256 and
// both `oc` and every position are single bytes, so neither can reach n. That is
// why the out-of-bounds write is only reachable through the tail APIs, where n is
// smaller than the block.
void test_b128_sparse_positions_out_of_range()
{
    std::vector<uint32_t> out(128, 0); // a b128 block is 128 elements
    // type=10 sparse, b=8, pb=4, oc=1, positions[0] = 200 -- past the block end
    const std::vector<uint8_t> in = makeBlock({0x80u | 8u, 4u, 1u, 200u});
    CHECK(abpfor::b128::decodeBlock(in.data(), out.data()) == 0,
          "b128::decodeBlock rejects sparse position 200");

    // oc = 250 for a 128-element block: overruns the residual buffer.
    const std::vector<uint8_t> tooMany = makeBlock({0x80u | 8u, 4u, 250u});
    CHECK(abpfor::b128::decodeBlock(tooMany.data(), out.data()) == 0,
          "b128::decodeBlock rejects oc=250");
}

/// The kSpecial short-circuit in `headerWidthsValid` is load-bearing: there `b`
/// is a sub-type rather than a width, and kRaw is b == 63 -- which the width
/// check would reject if the short-circuit were ever dropped.
void test_special_blocks_still_decode()
{
    uint32_t out[256] = {};

    const std::vector<uint8_t> all_zero = makeBlock({0xC0u | 0u}); // kAllZero
    CHECK(abpfor::b256::decodeBlock(all_zero.data(), out) == 1, "kAllZero still decodes (1 byte)");

    const std::vector<uint8_t> constant = makeBlock({0xC0u | 8u, 0x12u}); // constant, 8-bit value
    CHECK(abpfor::b256::decodeBlock(constant.data(), out) == 2, "constant still decodes (1 + 1 byte)");

    const std::vector<uint8_t> raw = makeBlock({0xC0u | 63u}); // kRaw: b == 63
    CHECK(abpfor::b256::decodeBlock(raw.data(), out) == 1 + 256 * sizeof(uint32_t),
          "kRaw still decodes (1 + n*sizeof(T))");
}

// Positive control: a well-formed outlier block must still decode, so an
// over-strict position check cannot pass by rejecting everything.
void test_outlier_roundtrip_still_works()
{
    uint32_t values[64] = {};
    for (unsigned i = 0; i < 64; ++i)
        values[i] = (i % 17 == 0) ? 100000u + i : i; // a few values far above the rest

    uint8_t buf[kBufferBytes] = {};
    const size_t written = abpfor::b256::encodeTail(values, 64, buf);

    uint32_t out[64] = {};
    const size_t consumed = abpfor::b256::decodeTail(buf, 64, out);
    CHECK(consumed == written, "outlier tail: consumed=%zu written=%zu", consumed, written);
    for (unsigned i = 0; i < 64; ++i)
        CHECK(out[i] == values[i], "outlier tail i=%u: got=%u expected=%u", i, out[i], values[i]);
}

} // namespace

int main()
{
    test_bitpack_width_u32();
    test_outlier_width_u32();
    test_sparse_width_u32();
    test_64bit();
    test_sparse_position_past_block_end();
    test_bitmap_past_block_end();
    test_outlier_count_beyond_block();
    test_b128_u64_sparse_positions_out_of_range();
    test_b128_sparse_positions_out_of_range();
    test_special_blocks_still_decode();
    test_rejection_leaves_output_untouched();
    test_bitmap_rejection_leaves_output_untouched();
    test_stream_propagates_rejection();
    test_outlier_roundtrip_still_works();

    if (failures == 0)
        printf("All corrupt-header tests passed.\n");
    else
        printf("%d corrupt-header test(s) FAILED.\n", failures);

    return failures > 0 ? 1 : 0;
}
