// b256 API: 256-element blocks with 8-lane interleave.
#include "api_block.h"

namespace abpfor::b256 {

// --- uint32_t single block ---
size_t encodeBlock(const uint32_t* in, uint8_t* out) { return detail::apiEncodeBlock<256>(in, out); }
size_t encodeBlockDelta0(const uint32_t* in, uint8_t* out, uint32_t start) { return detail::apiEncodeBlockDelta0<256>(in, out, start); }
size_t encodeBlockDelta1(const uint32_t* in, uint8_t* out, uint32_t start) { return detail::apiEncodeBlockDelta1<256>(in, out, start); }
size_t decodeBlock(const uint8_t* in, uint32_t* out) { return detail::apiDecodeBlock<256, uint32_t>(in, out); }
size_t decodeBlockDelta0(const uint8_t* in, uint32_t* out, uint32_t start) { return detail::apiDecodeBlockDelta0<256>(in, out, start); }
size_t decodeBlockDelta1(const uint8_t* in, uint32_t* out, uint32_t start) { return detail::apiDecodeBlockDelta1<256>(in, out, start); }

// --- uint64_t single block ---
size_t encodeBlock(const uint64_t* in, uint8_t* out) { return detail::apiEncodeBlock<256>(in, out); }
size_t encodeBlockDelta0(const uint64_t* in, uint8_t* out, uint64_t start) { return detail::apiEncodeBlockDelta0<256>(in, out, start); }
size_t encodeBlockDelta1(const uint64_t* in, uint8_t* out, uint64_t start) { return detail::apiEncodeBlockDelta1<256>(in, out, start); }
size_t decodeBlock(const uint8_t* in, uint64_t* out) { return detail::apiDecodeBlock<256, uint64_t>(in, out); }
size_t decodeBlockDelta0(const uint8_t* in, uint64_t* out, uint64_t start) { return detail::apiDecodeBlockDelta0<256>(in, out, start); }
size_t decodeBlockDelta1(const uint8_t* in, uint64_t* out, uint64_t start) { return detail::apiDecodeBlockDelta1<256>(in, out, start); }

// --- uint32_t stream ---
size_t encode(const uint32_t* in, unsigned n, uint8_t* out) { return detail::apiEncode<256>(in, n, out); }
size_t encodeDelta0(const uint32_t* in, unsigned n, uint8_t* out, uint32_t start) { return detail::apiEncodeDelta0<256>(in, n, out, start); }
size_t encodeDelta1(const uint32_t* in, unsigned n, uint8_t* out, uint32_t start) { return detail::apiEncodeDelta1<256>(in, n, out, start); }
size_t decode(const uint8_t* in, unsigned n, uint32_t* out) { return detail::apiDecode<256>(in, n, out); }
size_t decodeDelta0(const uint8_t* in, unsigned n, uint32_t* out, uint32_t start) { return detail::apiDecodeDelta0<256>(in, n, out, start); }
size_t decodeDelta1(const uint8_t* in, unsigned n, uint32_t* out, uint32_t start) { return detail::apiDecodeDelta1<256>(in, n, out, start); }

// --- uint64_t stream ---
size_t encode(const uint64_t* in, unsigned n, uint8_t* out) { return detail::apiEncode<256>(in, n, out); }
size_t encodeDelta0(const uint64_t* in, unsigned n, uint8_t* out, uint64_t start) { return detail::apiEncodeDelta0<256>(in, n, out, start); }
size_t encodeDelta1(const uint64_t* in, unsigned n, uint8_t* out, uint64_t start) { return detail::apiEncodeDelta1<256>(in, n, out, start); }
size_t decode(const uint8_t* in, unsigned n, uint64_t* out) { return detail::apiDecode<256>(in, n, out); }
size_t decodeDelta0(const uint8_t* in, unsigned n, uint64_t* out, uint64_t start) { return detail::apiDecodeDelta0<256>(in, n, out, start); }
size_t decodeDelta1(const uint8_t* in, unsigned n, uint64_t* out, uint64_t start) { return detail::apiDecodeDelta1<256>(in, n, out, start); }

// --- Tail ---
size_t encodeTail(const uint32_t* in, unsigned n, uint8_t* out) { return detail::apiEncodeTail<256>(in, n, out); }
size_t encodeTailDelta0(const uint32_t* in, unsigned n, uint8_t* out, uint32_t start) { return detail::apiEncodeTailDelta0<256>(in, n, out, start); }
size_t encodeTailDelta1(const uint32_t* in, unsigned n, uint8_t* out, uint32_t start) { return detail::apiEncodeTailDelta1<256>(in, n, out, start); }
size_t decodeTail(const uint8_t* in, unsigned n, uint32_t* out) { return detail::apiDecodeTail<256>(in, n, out); }
size_t decodeTailDelta0(const uint8_t* in, unsigned n, uint32_t* out, uint32_t start) { return detail::apiDecodeTailDelta0<256>(in, n, out, start); }
size_t decodeTailDelta1(const uint8_t* in, unsigned n, uint32_t* out, uint32_t start) { return detail::apiDecodeTailDelta1<256>(in, n, out, start); }

size_t encodeTail(const uint64_t* in, unsigned n, uint8_t* out) { return detail::apiEncodeTail<256>(in, n, out); }
size_t encodeTailDelta0(const uint64_t* in, unsigned n, uint8_t* out, uint64_t start) { return detail::apiEncodeTailDelta0<256>(in, n, out, start); }
size_t encodeTailDelta1(const uint64_t* in, unsigned n, uint8_t* out, uint64_t start) { return detail::apiEncodeTailDelta1<256>(in, n, out, start); }
size_t decodeTail(const uint8_t* in, unsigned n, uint64_t* out) { return detail::apiDecodeTail<256>(in, n, out); }
size_t decodeTailDelta0(const uint8_t* in, unsigned n, uint64_t* out, uint64_t start) { return detail::apiDecodeTailDelta0<256>(in, n, out, start); }
size_t decodeTailDelta1(const uint8_t* in, unsigned n, uint64_t* out, uint64_t start) { return detail::apiDecodeTailDelta1<256>(in, n, out, start); }

} // namespace abpfor::b256
