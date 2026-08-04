# abpfor

Fast integer compression for sorted and unsorted 32/64-bit sequences. Adaptive bit-packing with lightweight exception coding.

## Features

- **32-bit and 64-bit** support
- **Delta encoding** for sorted sequences (successive differences minus 1)
- **Fused decode** — delta + unpack in a single pass, no intermediate buffer
- **Adaptive outlier handling** — bitmap or sparse exception coding
- **SIMD** — SSE 4-way and AVX2 8-way interleaved packing
- Header-only public API, static library for dispatch tables

## Quick Start

```cpp
#include <abpfor.h>
using namespace abpfor;

// Compress a block of 128 sorted uint32_t values
std::vector<uint32_t> data = { /* 128 sorted values */ };
std::vector<uint8_t> compressed(b128::maxCompressedSize<uint32_t>(data.size()));
size_t bytes = b128::encodeDelta1(data.data(), data.size(), compressed.data(), 0);

// Decompress
std::vector<uint32_t> decoded(128);
size_t consumed = b128::decodeDelta1(compressed.data(), decoded.size(), decoded.data(), 0);
```

### Stream API

```cpp
// Encode/decode arbitrary-length streams (auto-blocks internally)
size_t bytes = b256::encodeDelta1(data.data(), n, out, start);
size_t consumed = b256::decodeDelta1(in, n, out, start);
```

## API

Two block sizes: `b128` (128 elements, SSE interleave) and `b256` (256 elements, AVX2 interleave).

```cpp
namespace abpfor::b128 {  // or abpfor::b256

// Single block (fixed-size: 128 or 256 elements)
[[nodiscard]] size_t encodeBlock(const uint32_t* in, uint8_t* out);
[[nodiscard]] size_t encodeBlockDelta1(const uint32_t* in, uint8_t* out, uint32_t start);
[[nodiscard]] size_t decodeBlock(const uint8_t* in, uint32_t* out);
[[nodiscard]] size_t decodeBlockDelta1(const uint8_t* in, uint32_t* out, uint32_t start);

// Stream (any length, blocks + scalar tail)
[[nodiscard]] size_t encode(const uint32_t* in, unsigned n, uint8_t* out);
[[nodiscard]] size_t encodeDelta1(const uint32_t* in, unsigned n, uint8_t* out, uint32_t start);
[[nodiscard]] size_t decode(const uint8_t* in, unsigned n, uint32_t* out);
[[nodiscard]] size_t decodeDelta1(const uint8_t* in, unsigned n, uint32_t* out, uint32_t start);

// Also: Delta0 variants, uint64_t overloads, Tail functions
}
```

### Return values

Every function returns the number of bytes written (encode) or consumed
(decode). That count is the only way to locate the next block, so all of them
are `[[nodiscard]]` — dropping it desynchronises the caller's cursor. If a
caller genuinely knows the length by other means, say so explicitly:

```cpp
std::ignore = b256::decodeBlockDelta1(in, out, start);
```

Every count is a `size_t`, including the single-block functions whose output is
bounded by a compile-time constant (`maxCompressedSize<uint64_t>(256)`, about
2 KiB). A narrower return type there would be safe, but it buys nothing —
`[[nodiscard]]` is what catches a dropped count — and it costs a zero-extension
per call at any caller that adds the count to a 64-bit cursor, since x86-64 does
not guarantee the upper half of a 32-bit return register. Callers that keep
lengths in a 32-bit field should narrow at that field, where the bound making it
safe is visible.

Advance a cursor with the count rather than reconstructing an end pointer —
`p += decodeBlock(p, out)`, not `end = decodeBlock(p, out) + p`. Encode wants the
output base and decode the input base, so a swapped base compiles fine and
corrupts silently.

### Decode buffer requirement

The decode input buffer must have at least **3 readable bytes** past the compressed data (the unpacker loads full 64-bit words and discards overflow bits).

### Output buffer size

`Block` functions take no element count and always write exactly the block size
— `b128` writes 128 elements, `b256` writes 256. A smaller output buffer is
overrun. Use the `Tail` functions for a partial block (they require
`1 <= n < BlockSize`):

```cpp
std::vector<uint32_t> out(256);        // b256::decodeBlock writes all 256
std::ignore = b256::decodeBlock(in, out.data());

std::vector<uint32_t> partial(37);     // fewer than a block
std::ignore = b256::decodeTail(in, 37, partial.data());
```

This library validates nothing — the header lists every precondition the caller
owns.

Size an **encode** output buffer with `maxCompressedSize`, never with a length a
previous encode returned. Encode may touch bytes past the length it reports — a
highly compressible block can report 4 bytes after scribbling a few beyond that —
so only `maxCompressedSize` is a safe allocation:

```cpp
std::vector<uint8_t> buf(b256::maxCompressedSize<uint32_t>(n));   // safe
// NOT: buf.resize(previously_returned_length)
```

The reported length is exact for *reading back*: decode consumes precisely that
many bytes, so it is the right value to store on disk and to advance a cursor by.

## Block Format

```
Header byte: [type:2][bits:6]

type=00  Bitpack only    — packed data follows
type=01  Bitmap outliers — [pb:1] [bitmap] [exceptions] [base]
type=10  Sparse outliers — [pb:1] [count:1] [positions] [exceptions] [base]
type=11  Special         — bits=0: all zeros; bits=1..62: constant; bits=63: raw
```

## Building

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build
```

Requires C++20, **clang 15+**, x86-64 with AVX2.

### Clang only

CMake rejects any other compiler. This is not a portability gap left open by
accident:

- GCC's `-O3` loop vectoriser miscompiles the interleaved unpack path. Four
  test binaries fail at bit widths 10/12/26/28/30 (32-bit) and 38/44 (64-bit);
  all pass under `-fno-tree-loop-vectorize`. Rather than ship a library that is
  silently wrong under a supported compiler, the build fails up front.
- The codegen is tuned against clang: `-fvectorize`, `-mbranches-within-32B-boundaries`
  and the AVX2 intrinsic paths are all validated on clang only.

abpfor is vendored into ClickHouse, which is itself clang-only, so this costs
nothing in practice.

### Warnings

The build uses ClickHouse's policy — `-Weverything -Wpedantic`, with individual
warnings disabled only where they are noise for this kind of code, each with a
reason in `CMakeLists.txt`. `-DABPFOR_WERROR=ON` promotes them to errors for CI.

Every deliberate truncation is written as an explicit cast, so an implicit
narrowing is a bug rather than a style question. Keep it that way when editing:
the conversion diagnostics are load-bearing.

## License

Apache 2.0. See LICENSE.
