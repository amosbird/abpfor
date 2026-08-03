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
size_t consumed = b128::decodeDelta1(compressed, decoded.size(), decoded.data(), 0);
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
size_t encodeBlock(const uint32_t* in, uint8_t* out);
size_t encodeBlockDelta1(const uint32_t* in, uint8_t* out, uint32_t start);
size_t decodeBlock(const uint8_t* in, uint32_t* out);
size_t decodeBlockDelta1(const uint8_t* in, uint32_t* out, uint32_t start);

// Stream (any length, blocks + scalar tail)
size_t encode(const uint32_t* in, unsigned n, uint8_t* out);
size_t encodeDelta1(const uint32_t* in, unsigned n, uint8_t* out, uint32_t start);
size_t decode(const uint8_t* in, unsigned n, uint32_t* out);
size_t decodeDelta1(const uint8_t* in, unsigned n, uint32_t* out, uint32_t start);

// Also: Delta0 variants, uint64_t overloads, Tail functions
}
```

### Decode buffer requirement

The decode input buffer must have at least **3 readable bytes** past the compressed data (the unpacker loads full 64-bit words and discards overflow bits).

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

Requires C++20, Clang 15+ or GCC 12+. x86-64 with AVX2 for full performance.

## License

Apache 2.0. See LICENSE.
