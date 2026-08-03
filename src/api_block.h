#pragma once

// Template implementation for b128 and b256 API. Instantiated in
// api_b128.cpp and api_b256.cpp with BS=128/256 respectively.

#include "abpfor.h"
#include "detail/codec.h"
#include "core/block.h"
#include "core/delta.h"
#include <cassert>

namespace abpfor::detail
{

// Traits to select I4 vs I8 codec by block size.
template <unsigned BS> struct BlockTraits;

template <> struct BlockTraits<128>
{
    template <typename T>
    static size_t encodeBlock(const T* in, unsigned n, uint8_t* out) { return encodeBlockI4(in, n, out); }
    template <typename T, bool MinusOne = false>
    static size_t decodeBlockDelta(const uint8_t* in, unsigned n, T* out, T& carry) { return decodeBlockI4<T, MinusOne>(in, n, out, carry); }
    template <typename T>
    static size_t decodeBlock(const uint8_t* in, unsigned n, T* out) { return decodeBlockI4<T>(in, n, out); }
};

template <> struct BlockTraits<256>
{
    template <typename T>
    static size_t encodeBlock(const T* in, unsigned n, uint8_t* out) { return encodeBlockI8(in, n, out); }
    template <typename T, bool MinusOne = false>
    static size_t decodeBlockDelta(const uint8_t* in, unsigned n, T* out, T& carry) { return decodeBlockI8<T, MinusOne>(in, n, out, carry); }
    template <typename T>
    static size_t decodeBlock(const uint8_t* in, unsigned n, T* out) { return decodeBlockI8<T>(in, n, out); }
};

// --- Single block ---

template <unsigned BS, typename T>
size_t apiEncodeBlock(const T* in, uint8_t* out)
{
    return BlockTraits<BS>::encodeBlock(in, BS, out);
}

template <unsigned BS, typename T>
size_t apiEncodeBlockDelta0(const T* in, uint8_t* out, T start)
{
    T tmp[BS];
    delta0(in, BS, tmp, start);
    return BlockTraits<BS>::encodeBlock(tmp, BS, out);
}

template <unsigned BS, typename T>
size_t apiEncodeBlockDelta1(const T* in, uint8_t* out, T start)
{
    T tmp[BS];
    delta(in, BS, tmp, start);
    return BlockTraits<BS>::encodeBlock(tmp, BS, out);
}

template <unsigned BS, typename T>
size_t apiDecodeBlock(const uint8_t* in, T* out)
{
    return BlockTraits<BS>::decodeBlock(in, BS, out);
}

template <unsigned BS, typename T>
size_t apiDecodeBlockDelta0(const uint8_t* in, T* out, T start)
{
    T carry = start;
    return BlockTraits<BS>::template decodeBlockDelta<T, false>(in, BS, out, carry);
}

template <unsigned BS, typename T>
size_t apiDecodeBlockDelta1(const uint8_t* in, T* out, T start)
{
    T carry = start;
    return BlockTraits<BS>::template decodeBlockDelta<T, true>(in, BS, out, carry);
}

// --- Tail ---

template <unsigned BS, typename T>
size_t apiEncodeTail(const T* in, unsigned n, uint8_t* out)
{
    assert(n < BS && "tail must be < block size");
    return abpfor::encodeBlock(in, n, out);
}

template <unsigned BS, typename T>
size_t apiEncodeTailDelta0(const T* in, unsigned n, uint8_t* out, T start)
{
    assert(n < BS && "tail must be < block size");
    return abpfor::encodeBlockDelta0(in, n, out, start);
}

template <unsigned BS, typename T>
size_t apiEncodeTailDelta1(const T* in, unsigned n, uint8_t* out, T start)
{
    assert(n < BS && "tail must be < block size");
    return abpfor::encodeBlockDelta1(in, n, out, start);
}

template <unsigned BS, typename T>
size_t apiDecodeTail(const uint8_t* in, unsigned n, T* out)
{
    assert(n < BS && "tail must be < block size");
    return abpfor::decodeBlock(in, n, out);
}

template <unsigned BS, typename T>
size_t apiDecodeTailDelta0(const uint8_t* in, unsigned n, T* out, T carry)
{
    assert(n < BS && "tail must be < block size");
    return abpfor::decodeBlockDelta0(in, n, out, carry);
}

template <unsigned BS, typename T>
size_t apiDecodeTailDelta1(const uint8_t* in, unsigned n, T* out, T carry)
{
    assert(n < BS && "tail must be < block size");
    return abpfor::decodeBlockDelta1(in, n, out, carry);
}

// --- Stream ---

template <unsigned BS, typename T>
size_t apiEncode(const T* in, unsigned n, uint8_t* out)
{
    uint8_t* op = out;
    unsigned pos = 0;
    while (pos + BS <= n) { op += apiEncodeBlock<BS>(in + pos, op); pos += BS; }
    if (pos < n) op += apiEncodeTail<BS>(in + pos, n - pos, op);
    return static_cast<size_t>(op - out);
}

template <unsigned BS, typename T>
size_t apiEncodeDelta0(const T* in, unsigned n, uint8_t* out, T start)
{
    uint8_t* op = out;
    unsigned pos = 0;
    T s = start;
    while (pos + BS <= n) { op += apiEncodeBlockDelta0<BS>(in + pos, op, s); s = in[pos + BS - 1]; pos += BS; }
    if (pos < n) op += apiEncodeTailDelta0<BS>(in + pos, n - pos, op, s);
    return static_cast<size_t>(op - out);
}

template <unsigned BS, typename T>
size_t apiEncodeDelta1(const T* in, unsigned n, uint8_t* out, T start)
{
    uint8_t* op = out;
    unsigned pos = 0;
    T s = start;
    while (pos + BS <= n) { op += apiEncodeBlockDelta1<BS>(in + pos, op, s); s = in[pos + BS - 1]; pos += BS; }
    if (pos < n) op += apiEncodeTailDelta1<BS>(in + pos, n - pos, op, s);
    return static_cast<size_t>(op - out);
}

template <unsigned BS, typename T>
size_t apiDecode(const uint8_t* in, unsigned n, T* out)
{
    const uint8_t* ip = in;
    unsigned pos = 0;
    while (pos + BS <= n) { ip += apiDecodeBlock<BS>(ip, out + pos); pos += BS; }
    if (pos < n) ip += apiDecodeTail<BS>(ip, n - pos, out + pos);
    return static_cast<size_t>(ip - in);
}

template <unsigned BS, typename T>
size_t apiDecodeDelta0(const uint8_t* in, unsigned n, T* out, T start)
{
    const uint8_t* ip = in;
    unsigned pos = 0;
    T carry = start;
    while (pos + BS <= n) { ip += BlockTraits<BS>::template decodeBlockDelta<T, false>(ip, BS, out + pos, carry); pos += BS; }
    if (pos < n) ip += apiDecodeTailDelta0<BS>(ip, n - pos, out + pos, carry);
    return static_cast<size_t>(ip - in);
}

template <unsigned BS, typename T>
size_t apiDecodeDelta1(const uint8_t* in, unsigned n, T* out, T start)
{
    const uint8_t* ip = in;
    unsigned pos = 0;
    T carry = start;
    while (pos + BS <= n) { ip += BlockTraits<BS>::template decodeBlockDelta<T, true>(ip, BS, out + pos, carry); pos += BS; }
    if (pos < n) ip += apiDecodeTailDelta1<BS>(ip, n - pos, out + pos, carry);
    return static_cast<size_t>(ip - in);
}

} // namespace abpfor::detail
