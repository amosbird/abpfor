#pragma once

// abpfor — public API.
//
// Two block sizes: b128 (128 elements, 4-lane interleave) and b256 (256
// elements, 8-lane interleave).
//
// Return value: all functions return the number of bytes written (encode) or
// consumed (decode) in the compressed buffer.
//
// Parameters:
//   in    — input data (raw values for encode, compressed bytes for decode)
//   out   — output buffer (compressed bytes for encode, decoded values for decode)
//   n     — number of elements (stream functions only; block functions are fixed-size)
//   start — reference value for delta coding (previous element, or 0 for first block)
//
// Block functions operate on exactly 128 or 256 elements (no n parameter).
// Stream functions process n elements: full blocks via interleaved codec, tail via scalar.
//
// UNDEFINED BEHAVIOUR — this library is performance-first and validates nothing.
// The caller owns every precondition below; violating any of them is UB, not an
// error return:
//
//   * Decode input must be a byte stream previously produced by the matching
//     encode function. Decoding untrusted, truncated, or corrupted bytes is UB:
//     the block header's 6-bit width field is used to index a 33-entry dispatch
//     table with no bounds check, so a hostile header jumps through a wild
//     pointer. Validate or authenticate your input before decoding it.
//   * `out` must be large enough — size it with `maxCompressedSize`, not with a
//     length a previous encode returned. Encode may touch bytes past the length
//     it reports (a highly compressible block can report 4 bytes after writing a
//     few beyond that); decode writes exactly n elements. No size is checked.
//   * Decode input must have at least 3 readable bytes after the compressed
//     data. Bit unpacking loads whole 64-bit words and lets the final load run
//     past the packed bits rather than assembling that word byte by byte; the
//     values are discarded, but the bytes are read. Buffers sized to exactly
//     the compressed length may fault on the last block. The bound is pinned
//     by test_unpack_read_bounds in tests/test_pack.cpp.
//   * Tail functions require 1 <= n < block size; `n == 0` is UB.
//   * Delta block functions take `start` by value and do not return the updated
//     carry. Chaining them across a multi-block stream silently decodes wrong
//     data — use the stream functions for multi-block input.
//   * The compressed format is little-endian and is not portable across builds
//     with different block sizes.
//
// x86-64 requires AVX2 (compile-time baseline, not runtime-detected).
//
// RETURN VALUES — every encode/decode returns the number of bytes written
// (encode) or consumed (decode) as a `size_t`, and all of them are
// [[nodiscard]]: the byte count is the only way to know where the next block
// starts, so dropping it silently desynchronises the caller's cursor. Use
// `std::ignore = ...` when a caller genuinely knows the length by other means.
//
// The count is `size_t` even for the single-block functions, whose output is
// bounded by a compile-time constant (~2 KiB). Returning a narrower type there
// would be safe but pointless: it buys nothing, since [[nodiscard]] is what
// catches a dropped count, and it costs a zero-extension per call at any caller
// that adds the count to a 64-bit cursor (x86-64 does not guarantee the upper
// half of a 32-bit return register). A caller that stores lengths in a 32-bit
// field should narrow at that field, where the bound justifying it is visible.

#include <cstddef>
#include <cstdint>

namespace abpfor::b128 {

/// Upper bound on compressed output size (bytes) for n elements of type T.
template <typename T>
constexpr size_t maxCompressedSize(size_t n) { return n * sizeof(T) + (n / 128 + 2) * 4; }

// Single block (128 elements)
[[nodiscard]] size_t encodeBlock(const uint32_t* in, uint8_t* out);
[[nodiscard]] size_t encodeBlockDelta0(const uint32_t* in, uint8_t* out, uint32_t start);
[[nodiscard]] size_t encodeBlockDelta1(const uint32_t* in, uint8_t* out, uint32_t start);
[[nodiscard]] size_t decodeBlock(const uint8_t* in, uint32_t* out);
[[nodiscard]] size_t decodeBlockDelta0(const uint8_t* in, uint32_t* out, uint32_t start);
[[nodiscard]] size_t decodeBlockDelta1(const uint8_t* in, uint32_t* out, uint32_t start);

[[nodiscard]] size_t encodeBlock(const uint64_t* in, uint8_t* out);
[[nodiscard]] size_t encodeBlockDelta0(const uint64_t* in, uint8_t* out, uint64_t start);
[[nodiscard]] size_t encodeBlockDelta1(const uint64_t* in, uint8_t* out, uint64_t start);
[[nodiscard]] size_t decodeBlock(const uint8_t* in, uint64_t* out);
[[nodiscard]] size_t decodeBlockDelta0(const uint8_t* in, uint64_t* out, uint64_t start);
[[nodiscard]] size_t decodeBlockDelta1(const uint8_t* in, uint64_t* out, uint64_t start);

// Stream (n elements, main loop interleaved, tail scalar)
[[nodiscard]] size_t encode(const uint32_t* in, unsigned n, uint8_t* out);
[[nodiscard]] size_t encodeDelta0(const uint32_t* in, unsigned n, uint8_t* out, uint32_t start);
[[nodiscard]] size_t encodeDelta1(const uint32_t* in, unsigned n, uint8_t* out, uint32_t start);
[[nodiscard]] size_t decode(const uint8_t* in, unsigned n, uint32_t* out);
[[nodiscard]] size_t decodeDelta0(const uint8_t* in, unsigned n, uint32_t* out, uint32_t start);
[[nodiscard]] size_t decodeDelta1(const uint8_t* in, unsigned n, uint32_t* out, uint32_t start);

[[nodiscard]] size_t encode(const uint64_t* in, unsigned n, uint8_t* out);
[[nodiscard]] size_t encodeDelta0(const uint64_t* in, unsigned n, uint8_t* out, uint64_t start);
[[nodiscard]] size_t encodeDelta1(const uint64_t* in, unsigned n, uint8_t* out, uint64_t start);
[[nodiscard]] size_t decode(const uint8_t* in, unsigned n, uint64_t* out);
[[nodiscard]] size_t decodeDelta0(const uint8_t* in, unsigned n, uint64_t* out, uint64_t start);
[[nodiscard]] size_t decodeDelta1(const uint8_t* in, unsigned n, uint64_t* out, uint64_t start);

// Tail (< 128 elements, scalar codec, no interleaving overhead)
[[nodiscard]] size_t encodeTail(const uint32_t* in, unsigned n, uint8_t* out);
[[nodiscard]] size_t encodeTailDelta0(const uint32_t* in, unsigned n, uint8_t* out, uint32_t start);
[[nodiscard]] size_t encodeTailDelta1(const uint32_t* in, unsigned n, uint8_t* out, uint32_t start);
[[nodiscard]] size_t decodeTail(const uint8_t* in, unsigned n, uint32_t* out);
[[nodiscard]] size_t decodeTailDelta0(const uint8_t* in, unsigned n, uint32_t* out, uint32_t start);
[[nodiscard]] size_t decodeTailDelta1(const uint8_t* in, unsigned n, uint32_t* out, uint32_t start);

[[nodiscard]] size_t encodeTail(const uint64_t* in, unsigned n, uint8_t* out);
[[nodiscard]] size_t encodeTailDelta0(const uint64_t* in, unsigned n, uint8_t* out, uint64_t start);
[[nodiscard]] size_t encodeTailDelta1(const uint64_t* in, unsigned n, uint8_t* out, uint64_t start);
[[nodiscard]] size_t decodeTail(const uint8_t* in, unsigned n, uint64_t* out);
[[nodiscard]] size_t decodeTailDelta0(const uint8_t* in, unsigned n, uint64_t* out, uint64_t start);
[[nodiscard]] size_t decodeTailDelta1(const uint8_t* in, unsigned n, uint64_t* out, uint64_t start);
}

namespace abpfor::b256 {

/// Upper bound on compressed output size (bytes) for n elements of type T.
template <typename T>
constexpr size_t maxCompressedSize(size_t n) { return n * sizeof(T) + (n / 256 + 2) * 4; }

// Single block (256 elements)
[[nodiscard]] size_t encodeBlock(const uint32_t* in, uint8_t* out);
[[nodiscard]] size_t encodeBlockDelta0(const uint32_t* in, uint8_t* out, uint32_t start);
[[nodiscard]] size_t encodeBlockDelta1(const uint32_t* in, uint8_t* out, uint32_t start);
[[nodiscard]] size_t decodeBlock(const uint8_t* in, uint32_t* out);
[[nodiscard]] size_t decodeBlockDelta0(const uint8_t* in, uint32_t* out, uint32_t start);
[[nodiscard]] size_t decodeBlockDelta1(const uint8_t* in, uint32_t* out, uint32_t start);

[[nodiscard]] size_t encodeBlock(const uint64_t* in, uint8_t* out);
[[nodiscard]] size_t encodeBlockDelta0(const uint64_t* in, uint8_t* out, uint64_t start);
[[nodiscard]] size_t encodeBlockDelta1(const uint64_t* in, uint8_t* out, uint64_t start);
[[nodiscard]] size_t decodeBlock(const uint8_t* in, uint64_t* out);
[[nodiscard]] size_t decodeBlockDelta0(const uint8_t* in, uint64_t* out, uint64_t start);
[[nodiscard]] size_t decodeBlockDelta1(const uint8_t* in, uint64_t* out, uint64_t start);

[[nodiscard]] size_t encode(const uint32_t* in, unsigned n, uint8_t* out);
[[nodiscard]] size_t encodeDelta0(const uint32_t* in, unsigned n, uint8_t* out, uint32_t start);
[[nodiscard]] size_t encodeDelta1(const uint32_t* in, unsigned n, uint8_t* out, uint32_t start);
[[nodiscard]] size_t decode(const uint8_t* in, unsigned n, uint32_t* out);
[[nodiscard]] size_t decodeDelta0(const uint8_t* in, unsigned n, uint32_t* out, uint32_t start);
[[nodiscard]] size_t decodeDelta1(const uint8_t* in, unsigned n, uint32_t* out, uint32_t start);

[[nodiscard]] size_t encode(const uint64_t* in, unsigned n, uint8_t* out);
[[nodiscard]] size_t encodeDelta0(const uint64_t* in, unsigned n, uint8_t* out, uint64_t start);
[[nodiscard]] size_t encodeDelta1(const uint64_t* in, unsigned n, uint8_t* out, uint64_t start);
[[nodiscard]] size_t decode(const uint8_t* in, unsigned n, uint64_t* out);
[[nodiscard]] size_t decodeDelta0(const uint8_t* in, unsigned n, uint64_t* out, uint64_t start);
[[nodiscard]] size_t decodeDelta1(const uint8_t* in, unsigned n, uint64_t* out, uint64_t start);

// Tail (< 256 elements, scalar codec, no interleaving overhead)
[[nodiscard]] size_t encodeTail(const uint32_t* in, unsigned n, uint8_t* out);
[[nodiscard]] size_t encodeTailDelta0(const uint32_t* in, unsigned n, uint8_t* out, uint32_t start);
[[nodiscard]] size_t encodeTailDelta1(const uint32_t* in, unsigned n, uint8_t* out, uint32_t start);
[[nodiscard]] size_t decodeTail(const uint8_t* in, unsigned n, uint32_t* out);
[[nodiscard]] size_t decodeTailDelta0(const uint8_t* in, unsigned n, uint32_t* out, uint32_t start);
[[nodiscard]] size_t decodeTailDelta1(const uint8_t* in, unsigned n, uint32_t* out, uint32_t start);

[[nodiscard]] size_t encodeTail(const uint64_t* in, unsigned n, uint8_t* out);
[[nodiscard]] size_t encodeTailDelta0(const uint64_t* in, unsigned n, uint8_t* out, uint64_t start);
[[nodiscard]] size_t encodeTailDelta1(const uint64_t* in, unsigned n, uint8_t* out, uint64_t start);
[[nodiscard]] size_t decodeTail(const uint8_t* in, unsigned n, uint64_t* out);
[[nodiscard]] size_t decodeTailDelta0(const uint8_t* in, unsigned n, uint64_t* out, uint64_t start);
[[nodiscard]] size_t decodeTailDelta1(const uint8_t* in, unsigned n, uint64_t* out, uint64_t start);
}
