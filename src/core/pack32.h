#pragma once

// 32-bit optimized bitpack/unpack with dispatch tables.

#include "arch.h"
#include "pack_internal.h"

#if ABPFOR_ARCH_X86
#include <immintrin.h>
#endif

namespace abpfor::detail::bitops
{

// --- Bitpack implementation ---

template <unsigned B, unsigned Base, size_t I>
static ABPFOR_INLINE void pack_one(const uint32_t* __restrict in, uint64_t* __restrict w)
{
    constexpr unsigned idx = Base + static_cast<unsigned>(I);
    constexpr unsigned bitpos = static_cast<unsigned>(I) * B;
    constexpr unsigned wi = bitpos / 64u;
    constexpr unsigned sh = bitpos % 64u;
    w[wi] |= static_cast<uint64_t>(in[idx]) << sh;
    if constexpr (sh + B > 64u) w[wi + 1u] |= static_cast<uint64_t>(in[idx]) >> (64u - sh);
}

template <unsigned B, unsigned Base, size_t... I>
static ABPFOR_INLINE void pack_all(const uint32_t* __restrict in, uint64_t* __restrict w, std::index_sequence<I...>)
{
    (pack_one<B, Base, I>(in, w), ...);
}

template <unsigned B, unsigned K, unsigned Base>
static ABPFOR_INLINE unsigned char* pack_block(const uint32_t* __restrict in, unsigned char* __restrict out)
{
    constexpr unsigned total_bits = K * B;
    constexpr unsigned total_bytes = (total_bits + 7u) / 8u;
    constexpr unsigned word_count = (total_bits + 63u) / 64u;
    constexpr unsigned last_bytes = total_bytes - (word_count - 1u) * 8u;

    uint64_t w[word_count] = {};
    pack_all<B, Base>(in, w, std::make_index_sequence<K>{});

    unsigned char* op = out;
    for (unsigned i = 0; i + 1u < word_count; ++i)
    {
        storeU64Fast(op, w[i]);
        op += 8u;
    }
    if constexpr (last_bytes == 8u)
    {
        storeU64Fast(op, w[word_count - 1u]);
        op += 8u;
    }
    else
    {
        store_partial<last_bytes>(op, w[word_count - 1u]);
    }
    return op;
}

template <unsigned B, unsigned N, unsigned Base>
static ABPFOR_INLINE unsigned char* pack_blocks(const uint32_t* __restrict in, unsigned char* __restrict out)
{
    if constexpr (N == 0u)
    {
        return out;
    }
    else
    {
        constexpr unsigned block = choose_block_size(B, N);
        unsigned char* op = pack_block<B, block, Base>(in, out);
        if constexpr (N == block)
            return op;
        else
            return pack_blocks<B, N - block, Base + block>(in, op);
    }
}

template <unsigned B, unsigned N>
static ABPFOR_INLINE unsigned char* pack_n_b(const uint32_t* __restrict in, unsigned char* __restrict out)
{
    static_assert(B >= 1 && B <= 32);
    static_assert(N >= 1 && N <= 32);

    // shortcut: B=32 handled by pack32() top-level memcpy; not reachable here
    if constexpr (B == 16)
    {
        unsigned char* op = out;
        for (unsigned i = 0; i < N; ++i)
        {
            storeU16Fast(op, static_cast<uint16_t>(in[i]));
            op += 2u;
        }
        return op;
    }
    else if constexpr (B == 8)
    {
        unsigned char* op = out;
        for (unsigned i = 0; i < N; ++i) *op++ = static_cast<unsigned char>(in[i]);
        return op;
    }
    else
    {
        return pack_blocks<B, N, 0u>(in, out);
    }
}

template <unsigned B>
static ABPFOR_INLINE unsigned char* pack_dispatch_n(const uint32_t* in, unsigned n, unsigned char* out)
{
    switch (n)
    {
    case 1u:
        return pack_n_b<B, 1>(in, out);
    case 2u:
        return pack_n_b<B, 2>(in, out);
    case 3u:
        return pack_n_b<B, 3>(in, out);
    case 4u:
        return pack_n_b<B, 4>(in, out);
    case 5u:
        return pack_n_b<B, 5>(in, out);
    case 6u:
        return pack_n_b<B, 6>(in, out);
    case 7u:
        return pack_n_b<B, 7>(in, out);
    case 8u:
        return pack_n_b<B, 8>(in, out);
    case 9u:
        return pack_n_b<B, 9>(in, out);
    case 10u:
        return pack_n_b<B, 10>(in, out);
    case 11u:
        return pack_n_b<B, 11>(in, out);
    case 12u:
        return pack_n_b<B, 12>(in, out);
    case 13u:
        return pack_n_b<B, 13>(in, out);
    case 14u:
        return pack_n_b<B, 14>(in, out);
    case 15u:
        return pack_n_b<B, 15>(in, out);
    case 16u:
        return pack_n_b<B, 16>(in, out);
    case 17u:
        return pack_n_b<B, 17>(in, out);
    case 18u:
        return pack_n_b<B, 18>(in, out);
    case 19u:
        return pack_n_b<B, 19>(in, out);
    case 20u:
        return pack_n_b<B, 20>(in, out);
    case 21u:
        return pack_n_b<B, 21>(in, out);
    case 22u:
        return pack_n_b<B, 22>(in, out);
    case 23u:
        return pack_n_b<B, 23>(in, out);
    case 24u:
        return pack_n_b<B, 24>(in, out);
    case 25u:
        return pack_n_b<B, 25>(in, out);
    case 26u:
        return pack_n_b<B, 26>(in, out);
    case 27u:
        return pack_n_b<B, 27>(in, out);
    case 28u:
        return pack_n_b<B, 28>(in, out);
    case 29u:
        return pack_n_b<B, 29>(in, out);
    case 30u:
        return pack_n_b<B, 30>(in, out);
    case 31u:
        return pack_n_b<B, 31>(in, out);
    default:
        __builtin_unreachable();
    }
}

// --- Byte-loop pack/unpack for specific widths ---
// All byte-loop functions are noinline: they are only called from pack_b<B> /
// unpack_b<B> which are function-pointer targets (pack_table[], unpack_table[]).
// Inlining them into pack_b<B> increases the combined function body enough to
// degrade clang's register allocation (measured: B=3 pack -5% when inlined).
// Since the call boundary is the only caller anyway, noinline costs one
// call/ret (~1ns) but buys cleaner regalloc across all widths.

__attribute__((noinline)) static unsigned char* pack_b2_loop(const uint32_t* __restrict in, unsigned n, unsigned char* __restrict out)
{
    const uint32_t* end = in + (n & ~63u);
    while (in < end)
    {
        for (unsigned b = 0; b < 16; ++b)
        {
            out[b] = static_cast<unsigned char>((in[b * 4 + 0] & 3) | ((in[b * 4 + 1] & 3) << 2) |
                                                ((in[b * 4 + 2] & 3) << 4) | ((in[b * 4 + 3] & 3) << 6));
        }
        in += 64;
        out += 16;
    }
    if (n & 32u)
    {
        for (unsigned b = 0; b < 8; ++b)
        {
            out[b] = static_cast<unsigned char>((in[b * 4 + 0] & 3) | ((in[b * 4 + 1] & 3) << 2) |
                                                ((in[b * 4 + 2] & 3) << 4) | ((in[b * 4 + 3] & 3) << 6));
        }
        in += 32;
        out += 8;
    }
    n &= 31u;
    if (n == 0u) return out;
    return pack_dispatch_n<2>(in, n, out);
}

__attribute__((noinline)) static unsigned char* pack_b4_loop(const uint32_t* __restrict in, unsigned n, unsigned char* __restrict out)
{
    const uint32_t* end = in + (n & ~63u);
    while (in < end)
    {
        for (unsigned b = 0; b < 32; ++b)
        {
            out[b] = static_cast<unsigned char>((in[b * 2] & 0xF) | ((in[b * 2 + 1] & 0xF) << 4));
        }
        in += 64;
        out += 32;
    }
    if (n & 32u)
    {
        for (unsigned b = 0; b < 16; ++b)
        {
            out[b] = static_cast<unsigned char>((in[b * 2] & 0xF) | ((in[b * 2 + 1] & 0xF) << 4));
        }
        in += 32;
        out += 16;
    }
    n &= 31u;
    if (n == 0u) return out;
    return pack_dispatch_n<4>(in, n, out);
}

__attribute__((noinline)) static unsigned char* pack_b12_loop(const uint32_t* __restrict in, unsigned n, unsigned char* __restrict out)
{
    using U32A = uint32_t __attribute__((__may_alias__));
    const uint32_t* end = in + (n & ~63u);
    while (in < end)
    {
        for (unsigned g = 0; g < 32; ++g)
        {
            uint32_t w = (in[g * 2] & 4095u) | ((in[g * 2 + 1] & 4095u) << 12);
            *reinterpret_cast<U32A*>(out + g * 3) = w;
        }
        in += 64;
        out += 96;
    }
    if (n & 32u)
    {
        for (unsigned g = 0; g < 16; ++g)
        {
            uint32_t w = (in[g * 2] & 4095u) | ((in[g * 2 + 1] & 4095u) << 12);
            *reinterpret_cast<U32A*>(out + g * 3) = w;
        }
        in += 32;
        out += 48;
    }
    n &= 31u;
    if (n == 0u) return out;
    return pack_dispatch_n<12>(in, n, out);
}

__attribute__((noinline)) static unsigned char* pack_b3_loop(const uint32_t* __restrict in, unsigned n,
                                                             unsigned char* __restrict out)
{
    const uint32_t* end = in + (n & ~7u);
    while (in < end)
    {
        uint32_t v = (in[0] & 7) | ((in[1] & 7) << 3) | ((in[2] & 7) << 6) | ((in[3] & 7) << 9) | ((in[4] & 7) << 12) |
                     ((in[5] & 7) << 15) | ((in[6] & 7) << 18) | ((in[7] & 7) << 21);
        // 8 values x 3 bits = 24 significant bits, so v < 2^24 and the three byte
        // stores below cover it exactly: bits 24..31 are always zero, and the
        // uint32_t -> unsigned char narrowing discards nothing.
        out[0] = static_cast<unsigned char>(v);
        out[1] = static_cast<unsigned char>(v >> 8);
        out[2] = static_cast<unsigned char>(v >> 16);
        out += 3;
        in += 8;
    }
    n &= 7u;
    if (n == 0u) return out;
    return pack_dispatch_n<3>(in, n, out);
}

template <unsigned B> static ABPFOR_INLINE unsigned char* pack_b(const uint32_t* in, unsigned n, unsigned char* out)
{
    if constexpr (B == 2)
    {
        return pack_b2_loop(in, n, out);
    }
    else if constexpr (B == 3)
    {
        return pack_b3_loop(in, n, out);
    }
    else if constexpr (B == 4)
    {
        return pack_b4_loop(in, n, out);
    }

#if !ABPFOR_BIG_ENDIAN
    else if constexpr (B == 12)
    {
        // shortcut: pack_b12_loop uses raw U32A stores; BE falls through to generic template path
        return pack_b12_loop(in, n, out);
    }
#endif
    else
    {
        const uint32_t* end = in + (n & ~31u);
        while (in < end)
        {
            out = pack_n_b<B, 32>(in, out);
            in += 32;
        }
        n &= 31u;
        if (n == 0u) return out;
        return pack_dispatch_n<B>(in, n, out);
    }
}

using PackFn = unsigned char* (*)(const uint32_t*, unsigned, unsigned char*);

// shortcut: explicit table; could use integer_sequence but this is clear and compiles fine
inline const PackFn pack_table[33] = {
    nullptr,     &pack_b<1>,  &pack_b<2>,  &pack_b<3>,  &pack_b<4>,  &pack_b<5>,  &pack_b<6>,  &pack_b<7>,  &pack_b<8>,
    &pack_b<9>,  &pack_b<10>, &pack_b<11>, &pack_b<12>, &pack_b<13>, &pack_b<14>, &pack_b<15>, &pack_b<16>, &pack_b<17>,
    &pack_b<18>, &pack_b<19>, &pack_b<20>, &pack_b<21>, &pack_b<22>, &pack_b<23>, &pack_b<24>, &pack_b<25>, &pack_b<26>,
    &pack_b<27>, &pack_b<28>, &pack_b<29>, &pack_b<30>, &pack_b<31>, &pack_b<32>,
};

// --- Bitunpack implementation ---

template <unsigned B, unsigned Base, size_t I>
static ABPFOR_INLINE void unpack_emit_one(const uint64_t* __restrict w, uint32_t* __restrict out)
{
    constexpr unsigned idx = Base + static_cast<unsigned>(I);
    constexpr unsigned bitpos = static_cast<unsigned>(I) * B;
    constexpr unsigned wi = bitpos / 64u;
    constexpr unsigned sh = bitpos % 64u;
    constexpr uint32_t mask = static_cast<uint32_t>((uint64_t{1} << B) - 1u);

    uint64_t v = w[wi] >> sh;
    if constexpr (sh + B > 64u) v |= w[wi + 1u] << (64u - sh);

    // shortcut: volatile store for B==17 prevents SLP vectorizer from merging stores (verified -8.5% without)
    uint32_t val = static_cast<uint32_t>(v) & mask;
    if constexpr (B == 17)
    {
        volatile uint32_t* vout = out;
        vout[idx] = val;
    }
    else
    {
        out[idx] = val;
    }
}
template <unsigned B, unsigned Base, size_t... I>
static ABPFOR_INLINE void unpack_emit(const uint64_t* __restrict w, uint32_t* __restrict out, std::index_sequence<I...>)
{
    (unpack_emit_one<B, Base, I>(w, out), ...);
}

template <unsigned B, unsigned K, unsigned Base>
static ABPFOR_INLINE const unsigned char* unpack_block(const unsigned char* __restrict in, uint32_t* __restrict out)
{
    constexpr unsigned total_bits = K * B;
    constexpr unsigned total_bytes = (total_bits + 7u) / 8u;
    constexpr unsigned word_count = (total_bits + 63u) / 64u;
    constexpr unsigned last_bytes = total_bytes - (word_count - 1u) * 8u;

    uint64_t w[word_count];
    const unsigned char* ip = in;
    for (unsigned i = 0; i + 1u < word_count; ++i)
    {
        w[i] = loadU64Fast(ip);
        ip += 8u;
    }
    if constexpr (last_bytes == 8u)
    {
        w[word_count - 1u] = loadU64Fast(ip);
        ip += 8u;
    }
    else
    {
        w[word_count - 1u] = load_partial<last_bytes>(ip);
    }

    unpack_emit<B, Base>(w, out, std::make_index_sequence<K>{});
    return ip;
}

template <unsigned B, unsigned N, unsigned Base>
static ABPFOR_INLINE const unsigned char* unpack_blocks(const unsigned char* __restrict in, uint32_t* __restrict out)
{
    if constexpr (N == 0u)
    {
        return in;
    }
    else
    {
        constexpr unsigned block = choose_block_size(B, N);
        const unsigned char* ip = unpack_block<B, block, Base>(in, out);
        if constexpr (N == block)
            return ip;
        else
            return unpack_blocks<B, N - block, Base + block>(ip, out);
    }
}

template <unsigned B, unsigned N>
static ABPFOR_INLINE const unsigned char* unpack_n_b(const unsigned char* __restrict in, uint32_t* __restrict out)
{
    static_assert(B >= 1 && B <= 32);
    static_assert(N >= 1 && N <= 32);

    // shortcut: B=32 handled by unpack_b<32>/unpack_table[32] which uses top-level path
    if constexpr (B == 16)
    {
        const unsigned char* ip = in;
        for (unsigned i = 0; i < N; ++i)
        {
            out[i] = loadU16Fast(ip);
            ip += 2u;
        }
        return ip;
    }
    else if constexpr (B == 8)
    {
        const unsigned char* ip = in;
        for (unsigned i = 0; i < N; ++i)
        {
            out[i] = *ip++;
        }
        return ip;
    }
    else
    {
        return unpack_blocks<B, N, 0u>(in, out);
    }
}

template <unsigned B>
static ABPFOR_INLINE const unsigned char* unpack(const unsigned char* in, unsigned n, uint32_t* out)
{
    switch (n)
    {
    case 1u:
        return unpack_n_b<B, 1>(in, out);
    case 2u:
        return unpack_n_b<B, 2>(in, out);
    case 3u:
        return unpack_n_b<B, 3>(in, out);
    case 4u:
        return unpack_n_b<B, 4>(in, out);
    case 5u:
        return unpack_n_b<B, 5>(in, out);
    case 6u:
        return unpack_n_b<B, 6>(in, out);
    case 7u:
        return unpack_n_b<B, 7>(in, out);
    case 8u:
        return unpack_n_b<B, 8>(in, out);
    case 9u:
        return unpack_n_b<B, 9>(in, out);
    case 10u:
        return unpack_n_b<B, 10>(in, out);
    case 11u:
        return unpack_n_b<B, 11>(in, out);
    case 12u:
        return unpack_n_b<B, 12>(in, out);
    case 13u:
        return unpack_n_b<B, 13>(in, out);
    case 14u:
        return unpack_n_b<B, 14>(in, out);
    case 15u:
        return unpack_n_b<B, 15>(in, out);
    case 16u:
        return unpack_n_b<B, 16>(in, out);
    case 17u:
        return unpack_n_b<B, 17>(in, out);
    case 18u:
        return unpack_n_b<B, 18>(in, out);
    case 19u:
        return unpack_n_b<B, 19>(in, out);
    case 20u:
        return unpack_n_b<B, 20>(in, out);
    case 21u:
        return unpack_n_b<B, 21>(in, out);
    case 22u:
        return unpack_n_b<B, 22>(in, out);
    case 23u:
        return unpack_n_b<B, 23>(in, out);
    case 24u:
        return unpack_n_b<B, 24>(in, out);
    case 25u:
        return unpack_n_b<B, 25>(in, out);
    case 26u:
        return unpack_n_b<B, 26>(in, out);
    case 27u:
        return unpack_n_b<B, 27>(in, out);
    case 28u:
        return unpack_n_b<B, 28>(in, out);
    case 29u:
        return unpack_n_b<B, 29>(in, out);
    case 30u:
        return unpack_n_b<B, 30>(in, out);
    case 31u:
        return unpack_n_b<B, 31>(in, out);
    default:
        __builtin_unreachable();
    }
}

__attribute__((noinline)) static const unsigned char* unpack_b1_loop(const unsigned char* __restrict in, unsigned n,
                                                                     uint32_t* __restrict out)
{
    uint32_t* end = out + (n & ~63u);
    while (out < end)
    {
        for (unsigned b = 0; b < 8; ++b)
        {
            uint8_t byte = in[b];
            out[b * 8 + 0] = (byte >> 0) & 1;
            out[b * 8 + 1] = (byte >> 1) & 1;
            out[b * 8 + 2] = (byte >> 2) & 1;
            out[b * 8 + 3] = (byte >> 3) & 1;
            out[b * 8 + 4] = (byte >> 4) & 1;
            out[b * 8 + 5] = (byte >> 5) & 1;
            out[b * 8 + 6] = (byte >> 6) & 1;
            out[b * 8 + 7] = (byte >> 7) & 1;
        }
        in += 8;
        out += 64;
    }
    if (n & 32u)
    {
        for (unsigned b = 0; b < 4; ++b)
        {
            uint8_t byte = in[b];
            out[b * 8 + 0] = (byte >> 0) & 1;
            out[b * 8 + 1] = (byte >> 1) & 1;
            out[b * 8 + 2] = (byte >> 2) & 1;
            out[b * 8 + 3] = (byte >> 3) & 1;
            out[b * 8 + 4] = (byte >> 4) & 1;
            out[b * 8 + 5] = (byte >> 5) & 1;
            out[b * 8 + 6] = (byte >> 6) & 1;
            out[b * 8 + 7] = (byte >> 7) & 1;
        }
        in += 4;
        out += 32;
    }
    n &= 31u;
    if (n == 0u) return in;
    return unpack<1>(in, n, out);
}

__attribute__((noinline)) static const unsigned char* unpack_b2_loop(const unsigned char* __restrict in, unsigned n,
                                                                     uint32_t* __restrict out)
{
    uint32_t* end = out + (n & ~63u);
    while (out < end)
    {
        for (unsigned b = 0; b < 16; ++b)
        {
            uint8_t byte = in[b];
            out[b * 4 + 0] = (byte >> 0) & 3;
            out[b * 4 + 1] = (byte >> 2) & 3;
            out[b * 4 + 2] = (byte >> 4) & 3;
            out[b * 4 + 3] = (byte >> 6) & 3;
        }
        in += 16;
        out += 64;
    }
    if (n & 32u)
    {
        for (unsigned b = 0; b < 8; ++b)
        {
            uint8_t byte = in[b];
            out[b * 4 + 0] = (byte >> 0) & 3;
            out[b * 4 + 1] = (byte >> 2) & 3;
            out[b * 4 + 2] = (byte >> 4) & 3;
            out[b * 4 + 3] = (byte >> 6) & 3;
        }
        in += 8;
        out += 32;
    }
    n &= 31u;
    if (n == 0u) return in;
    return unpack<2>(in, n, out);
}

__attribute__((noinline)) static const unsigned char* unpack_b4_loop(const unsigned char* __restrict in, unsigned n,
                                                                     uint32_t* __restrict out)
{
    uint32_t* end = out + (n & ~63u);
    while (out < end)
    {
        for (unsigned b = 0; b < 32; ++b)
        {
            uint8_t byte = in[b];
            out[b * 2 + 0] = byte & 0xF;
            out[b * 2 + 1] = byte >> 4;
        }
        in += 32;
        out += 64;
    }
    if (n & 32u)
    {
        for (unsigned b = 0; b < 16; ++b)
        {
            uint8_t byte = in[b];
            out[b * 2 + 0] = byte & 0xF;
            out[b * 2 + 1] = byte >> 4;
        }
        in += 16;
        out += 32;
    }
    n &= 31u;
    if (n == 0u) return in;
    return unpack<4>(in, n, out);
}

__attribute__((noinline)) static const unsigned char* unpack_b12_loop(const unsigned char* __restrict in, unsigned n,
                                                                      uint32_t* __restrict out)
{
    using U32A = uint32_t __attribute__((__may_alias__));
    uint32_t* end = out + (n & ~63u);
    while (out < end)
    {
        for (unsigned g = 0; g < 32; ++g)
        {
            uint32_t w = *reinterpret_cast<const U32A*>(in + g * 3);
            out[g * 2 + 0] = (w >> 0) & 4095;
            out[g * 2 + 1] = (w >> 12) & 4095;
        }
        in += 96;
        out += 64;
    }
    if (n & 32u)
    {
        for (unsigned g = 0; g < 16; ++g)
        {
            uint32_t w = *reinterpret_cast<const U32A*>(in + g * 3);
            out[g * 2 + 0] = (w >> 0) & 4095;
            out[g * 2 + 1] = (w >> 12) & 4095;
        }
        in += 48;
        out += 32;
    }
    n &= 31u;
    if (n == 0u) return in;
    return unpack<12>(in, n, out);
}

__attribute__((noinline)) static const unsigned char* unpack_b24_loop(const unsigned char* __restrict in, unsigned n,
                                                                      uint32_t* __restrict out)
{
    using U32A = uint32_t __attribute__((__may_alias__));
    uint32_t* end = out + (n & ~63u);
    while (out < end)
    {
        for (unsigned i = 0; i < 64; ++i)
        {
            out[i] = *reinterpret_cast<const U32A*>(in + i * 3) & 0xFFFFFF;
        }
        in += 192;
        out += 64;
    }
    if (n & 32u)
    {
        for (unsigned i = 0; i < 32; ++i)
        {
            out[i] = *reinterpret_cast<const U32A*>(in + i * 3) & 0xFFFFFF;
        }
        in += 96;
        out += 32;
    }
    n &= 31u;
    if (n == 0u) return in;
    return unpack<24>(in, n, out);
}

__attribute__((noinline)) static const unsigned char* unpack_b3_loop(const unsigned char* __restrict in, unsigned n,
                                                                     uint32_t* __restrict out)
{
    const uint32_t* end = out + (n & ~7u);
    while (out < end)
    {
        uint32_t v = in[0] | (uint32_t(in[1]) << 8) | (uint32_t(in[2]) << 16);
        out[0] = v & 7;
        out[1] = (v >> 3) & 7;
        out[2] = (v >> 6) & 7;
        out[3] = (v >> 9) & 7;
        out[4] = (v >> 12) & 7;
        out[5] = (v >> 15) & 7;
        out[6] = (v >> 18) & 7;
        out[7] = (v >> 21) & 7;
        in += 3;
        out += 8;
    }
    n &= 7u;
    if (n == 0u) return in;
    return unpack<3>(in, n, out);
}

__attribute__((noinline)) static const unsigned char* unpack_b5_loop(const unsigned char* __restrict in, unsigned n,
                                                                     uint32_t* __restrict out)
{
    const uint32_t* end = out + (n & ~7u);
    while (out < end)
    {
        uint64_t v = uint64_t(in[0]) | (uint64_t(in[1]) << 8) | (uint64_t(in[2]) << 16) | (uint64_t(in[3]) << 24) |
                     (uint64_t(in[4]) << 32);
        out[0] = v & 0x1f;
        out[1] = (v >> 5) & 0x1f;
        out[2] = (v >> 10) & 0x1f;
        out[3] = (v >> 15) & 0x1f;
        out[4] = (v >> 20) & 0x1f;
        out[5] = (v >> 25) & 0x1f;
        out[6] = (v >> 30) & 0x1f;
        out[7] = (v >> 35) & 0x1f;
        in += 5;
        out += 8;
    }
    n &= 7u;
    if (n == 0u) return in;
    return unpack<5>(in, n, out);
}

// Bit widths whose plain unpack reduces to whole-byte loads, so it runs
// 3-6x faster than generic bit extraction:
//
//   B = 8/16/32  one widening load per element, no bit manipulation at all
//   B = 1/4      a whole number of elements per byte, via unpack_b1/b4_loop
//   B = 12/24    a whole number of elements per 3 bytes, via unpack_b12/b24_loop
//
// Every other width straddles byte boundaries and needs the generic shift/mask
// path, which is no faster than what the fused delta loop already does.
//
// B = 2/3/5 have hand-written loops too, but they are not in this set: measured
// at 38-42ns per 128 elements they are no faster than generic extraction, so
// splitting the pass would cost more than it saves. Having a dedicated kernel
// and being fast are different things.
template <unsigned B> constexpr bool plainUnpackIsByteAligned()
{
    if constexpr (B == 8 || B == 16 || B == 32) return true;
    else if constexpr (B == 1 || B == 4) return true;
#if !ABPFOR_BIG_ENDIAN
    else if constexpr (B == 12 || B == 24) return true;
#endif
    else return false;
}

// Two values in one unaligned load, for widths that are a multiple of 4.
//
// When B % 4 == 0, a pair of values occupies exactly B/4 bytes, so one u64
// load holds both with the second at a fixed shift -- no cross-word merge.
// That merge is the main cost of the generic unpack_blocks path, which is why
// this is worth a special case for the widths that have no other fast path.
//
// Measured per 128 values (generic / pair), clang 22, CPUs 0/2/4:
//   B=20: 38.7 / 30.6    B=28: 43.7 / 30.8
// B=12 and B=24 already have dedicated loops that beat this (18.7 and 20.6),
// so they are not routed here.
//
// Read bound: the loop loads 8 bytes from each pair's offset, so it stops
// while that stays inside the 3 bytes of slack the public API requires after
// the packed data (see include/abpfor.h, pinned by test_unpack_read_bounds).
// The remainder goes through the generic path, which honours the same bound.
template <unsigned B>
static inline const unsigned char* unpack_pair_loop(const unsigned char* __restrict in,
                                                    unsigned n, uint32_t* __restrict out)
{
    static_assert(B % 4u == 0u && B < 32u);
    constexpr unsigned STEP = B / 4u;
    constexpr uint32_t M = static_cast<uint32_t>((uint64_t{1} << B) - 1u);

    const unsigned readable = (n * B + 7u) / 8u + kUnpackSlack;
    const unsigned inBounds = readable >= 8u ? ((readable - 8u) / STEP + 1u) : 0u;
    const unsigned pairs = (n / 2u) < inBounds ? (n / 2u) : inBounds;

    for (unsigned p = 0; p < pairs; ++p)
    {
        const uint64_t w = loadU64Fast(in);
        out[0] = static_cast<uint32_t>(w) & M;
        out[1] = static_cast<uint32_t>(w >> B) & M;
        in += STEP;
        out += 2;
    }
    const unsigned done = pairs * 2u;
    if (done == n) return in;
    return unpack<B>(in, n - done, out);
}

template <unsigned B>
static ABPFOR_INLINE const unsigned char* unpack_b(const unsigned char* in, unsigned n, uint32_t* out)
{
    if constexpr (B == 32)
    {
        detail::copyU32ArrayFromLe(out, in, n);
        return in + n * 4u;
    }
    else if constexpr (B == 1)
    {
        return unpack_b1_loop(in, n, out);
    }
    else if constexpr (B == 2)
    {
        return unpack_b2_loop(in, n, out);
    }
    else if constexpr (B == 3)
    {
        return unpack_b3_loop(in, n, out);
    }
    else if constexpr (B == 4)
    {
        return unpack_b4_loop(in, n, out);
    }
    else if constexpr (B == 5)
    {
        return unpack_b5_loop(in, n, out);
    }
#if !ABPFOR_BIG_ENDIAN
    // shortcut: unpack_b12/b24_loop use raw U32A loads; BE falls through to generic template path
    else if constexpr (B == 12)
    {
        return unpack_b12_loop(in, n, out);
    }
    else if constexpr (B == 24)
    {
        return unpack_b24_loop(in, n, out);
    }
#endif
    else if constexpr (B == 20 || B == 28)
    {
        return unpack_pair_loop<B>(in, n, out);
    }
    else
    {
        uint32_t* end = out + (n & ~31u);
        while (out < end)
        {
            in = unpack_n_b<B, 32>(in, out);
            out += 32;
        }
        n &= 31u;
        if (n == 0u) return in;
        return unpack<B>(in, n, out);
    }
}

using UnpackFn = const unsigned char* (*)(const unsigned char*, unsigned, uint32_t*);

// shortcut: explicit table; could use integer_sequence but this is clear and compiles fine
inline const UnpackFn unpack_table[33] = {
    nullptr,       &unpack_b<1>,  &unpack_b<2>,  &unpack_b<3>,  &unpack_b<4>,  &unpack_b<5>,  &unpack_b<6>,
    &unpack_b<7>,  &unpack_b<8>,  &unpack_b<9>,  &unpack_b<10>, &unpack_b<11>, &unpack_b<12>, &unpack_b<13>,
    &unpack_b<14>, &unpack_b<15>, &unpack_b<16>, &unpack_b<17>, &unpack_b<18>, &unpack_b<19>, &unpack_b<20>,
    &unpack_b<21>, &unpack_b<22>, &unpack_b<23>, &unpack_b<24>, &unpack_b<25>, &unpack_b<26>, &unpack_b<27>,
    &unpack_b<28>, &unpack_b<29>, &unpack_b<30>, &unpack_b<31>, &unpack_b<32>,
};

// --- Public dispatch functions ---

inline unsigned char* pack32(const uint32_t* in, unsigned n, unsigned char* out, unsigned b)
{
    if (b == 0u) [[unlikely]]
        return out;
    if (b == 32u) [[unlikely]]
    {
        detail::copyU32ArrayToLe(out, in, n);
        return out + n * 4u;
    }
    return pack_table[b](in, n, out);
}

inline const unsigned char* unpack32(const unsigned char* in, unsigned n, uint32_t* out, unsigned b)
{
    if (b == 0u) [[unlikely]]
    {
        std::fill(out, out + n, 0u);
        return in;
    }
    return unpack_table[b](in, n, out);
}

// --- Fused delta unpack: template-expanded 1-pass decode + prefix-sum ---

// --- Specialized B=8 delta unpack: simple byte reads, no bit extraction ---
template <bool MinusOne = false, size_t... I>
static ABPFOR_INLINE void unpack_delta_b8_emit(const unsigned char* __restrict in, uint32_t* __restrict out,
                                               uint32_t& acc, std::index_sequence<I...>)
{
    // A lambda per element rather than a comma-operator fold: identical codegen
    // (verified by comparing generated assembly), but the sequencing is explicit
    // instead of leaning on the comma operator, which reads as a typo to both
    // humans and -Wcomma.
    const auto step = [&](unsigned i)
    {
        acc += static_cast<uint32_t>(in[i]);
        out[i] = acc + (MinusOne ? (i + 1u) : 0u);
    };
    (step(static_cast<unsigned>(I)), ...);
}

template <bool MinusOne = false>
static ABPFOR_INLINE const unsigned char* unpack_delta_b8_block(const unsigned char* in, uint32_t* out, uint32_t& acc)
{
    unpack_delta_b8_emit<MinusOne>(in, out, acc, std::make_index_sequence<32>{});
    return in + 32u;
}

// --- Specialized B=16 delta unpack: 16-bit reads, no bit extraction ---
template <bool MinusOne = false, size_t... I>
static ABPFOR_INLINE void unpack_delta_b16_emit(const unsigned char* __restrict in, uint32_t* __restrict out,
                                                uint32_t& acc, std::index_sequence<I...>)
{
    const auto step = [&](unsigned i)
    {
        acc += static_cast<uint32_t>(loadU16Fast(in + 2u * i));
        out[i] = acc + (MinusOne ? (i + 1u) : 0u);
    };
    (step(static_cast<unsigned>(I)), ...);
}

template <bool MinusOne = false>
static ABPFOR_INLINE const unsigned char* unpack_delta_b16_block(const unsigned char* in, uint32_t* out, uint32_t& acc)
{
    unpack_delta_b16_emit<MinusOne>(in, out, acc, std::make_index_sequence<32>{});
    return in + 64u;
}

template <unsigned B, unsigned Base, size_t I, bool MinusOne = false>
static ABPFOR_INLINE void unpack_delta_emit_one(const uint64_t* __restrict w, uint32_t* __restrict out, uint32_t& acc)
{
    constexpr unsigned idx = Base + static_cast<unsigned>(I);
    constexpr unsigned bitpos = static_cast<unsigned>(I) * B;
    constexpr unsigned wi = bitpos / 64u;
    constexpr unsigned sh = bitpos % 64u;
    constexpr uint32_t mask = static_cast<uint32_t>((uint64_t{1} << B) - 1u);

    uint64_t v = w[wi] >> sh;
    if constexpr (sh + B > 64u) v |= w[wi + 1u] << (64u - sh);

    acc += static_cast<uint32_t>(v) & mask;
    out[idx] = acc + (MinusOne ? (idx + 1u) : 0u);
}

template <unsigned B, unsigned Base, bool MinusOne = false, size_t... I>
static ABPFOR_INLINE void unpack_delta_emit(const uint64_t* __restrict w, uint32_t* __restrict out, uint32_t& acc,
                                            std::index_sequence<I...>)
{
    (unpack_delta_emit_one<B, Base, I, MinusOne>(w, out, acc), ...);
}

template <unsigned B, unsigned K, unsigned Base, bool MinusOne = false>
static ABPFOR_INLINE const unsigned char* unpack_delta_block(const unsigned char* __restrict in,
                                                             uint32_t* __restrict out, uint32_t& acc)
{
    constexpr unsigned total_bits = K * B;
    constexpr unsigned total_bytes = (total_bits + 7u) / 8u;
    constexpr unsigned word_count = (total_bits + 63u) / 64u;

    uint64_t w[word_count];
    for (unsigned i = 0; i < word_count; ++i) w[i] = loadU64Fast(in + i * 8u);
    unpack_delta_emit<B, Base, MinusOne>(w, out, acc, std::make_index_sequence<K>{});
    return in + total_bytes;
}

// In-lane inclusive prefix sum of 8 lanes: log2(8) shift-add steps.
// _mm256_slli_si256 shifts each 128-bit lane independently, so the first two
// steps produce two separate 4-lane prefixes; the permute+broadcast then folds
// the low lane's total into the high lane.
#if ABPFOR_ARCH_X86
static ABPFOR_INLINE __m256i inlanePrefixSum(__m256i v)
{
    v = _mm256_add_epi32(v, _mm256_slli_si256(v, 4));
    v = _mm256_add_epi32(v, _mm256_slli_si256(v, 8));
    __m256i lo = _mm256_permute2x128_si256(v, v, 0x08);
    return _mm256_add_epi32(v, _mm256_shuffle_epi32(lo, 0xFF));
}

// out[i] = start + sum(out[0..i]) + (MinusOne ? i+1 : 0), in place.
//
// Two-level scan: four 8-lane groups are summed independently, then combined
// with three adds. That cuts the serial carry chain from one broadcast per 8
// elements to one per 32. Measured on 128 elements: 32.4 -> 27.8 ns.
template <bool MinusOne>
static ABPFOR_INLINE void prefixSumInPlace(uint32_t* out, unsigned n, uint32_t start)
{
    const __m256i bc7 = _mm256_set1_epi32(7);
    const __m256i eight = _mm256_set1_epi32(8);
    __m256i carry = _mm256_set1_epi32(static_cast<int>(start));
    __m256i ramp = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    const __m256i ramp32 = _mm256_set1_epi32(32);

    unsigned i = 0;
    for (; i + 32u <= n; i += 32u)
    {
        __m256i a = inlanePrefixSum(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(out + i)));
        __m256i b = inlanePrefixSum(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(out + i + 8)));
        __m256i c = inlanePrefixSum(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(out + i + 16)));
        __m256i d = inlanePrefixSum(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(out + i + 24)));

        __m256i cb = _mm256_add_epi32(carry, _mm256_permutevar8x32_epi32(a, bc7));
        __m256i cc = _mm256_add_epi32(cb, _mm256_permutevar8x32_epi32(b, bc7));
        __m256i cd = _mm256_add_epi32(cc, _mm256_permutevar8x32_epi32(c, bc7));

        __m256i r1 = _mm256_add_epi32(ramp, eight);
        __m256i r2 = _mm256_add_epi32(r1, eight);
        __m256i r3 = _mm256_add_epi32(r2, eight);

        a = _mm256_add_epi32(a, carry);
        b = _mm256_add_epi32(b, cb);
        c = _mm256_add_epi32(c, cc);
        d = _mm256_add_epi32(d, cd);

        // Carry comes from d before the ramp is folded in: the ramp is part of
        // the delta1 output, not part of the running sum.
        carry = _mm256_permutevar8x32_epi32(d, bc7);

        if constexpr (MinusOne)
        {
            a = _mm256_add_epi32(a, ramp);
            b = _mm256_add_epi32(b, r1);
            c = _mm256_add_epi32(c, r2);
            d = _mm256_add_epi32(d, r3);
            ramp = _mm256_add_epi32(ramp, ramp32);
        }
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), a);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i + 8), b);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i + 16), c);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i + 24), d);
    }
    for (; i + 8u <= n; i += 8u)
    {
        __m256i v = inlanePrefixSum(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(out + i)));
        v = _mm256_add_epi32(v, carry);
        __m256i o = MinusOne ? _mm256_add_epi32(v, ramp) : v;
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), o);
        carry = _mm256_permutevar8x32_epi32(v, bc7);
        if constexpr (MinusOne) ramp = _mm256_add_epi32(ramp, eight);
    }
    uint32_t acc = static_cast<uint32_t>(_mm256_extract_epi32(carry, 0));
    for (; i < n; ++i)
    {
        acc += out[i];
        out[i] = acc + (MinusOne ? i + 1u : 0u);
    }
}
#endif

template <unsigned B, bool MinusOne = false>
static const unsigned char* unpack_delta_b(const unsigned char* in, unsigned n, uint32_t* out, uint32_t start)
{
#if ABPFOR_ARCH_X86
    // Two passes beat one when the plain unpack has a dedicated fast path.
    //
    // Fusing the prefix sum into the unpack forces generic bit extraction: the
    // running total has to be added to each value as it is produced, so the
    // whole-byte and whole-word kernels can't be used, and the serial
    // dependency also blocks the extraction itself from vectorising. Splitting
    // the work lets each half run at its own natural speed -- a fast plain
    // unpack, then a SIMD prefix sum -- at the cost of one extra pass over
    // 512 bytes, which stays in L1.
    //
    // Only worth it where that fast path exists. For the other widths the plain
    // unpack is the same generic extraction the fused loop already does, so
    // splitting would add a pass and buy nothing.
    //
    // Measured per 128 elements (fused / split), clang 22:
    //   B= 1: 62.0 / 49.3    B= 8: 62.4 / 38.3    B=16: 63.2 / 38.4
    //   B= 4: 62.9 / 42.6    B=12: 65.1 / 48.7    B=24: 71.3 / 54.7
    // and for a width without a fast path, showing why it is gated:
    //   B=20: 71.9 / 85.8    B=28: 84.1 / 94.5
    if constexpr (plainUnpackIsByteAligned<B>())
    {
        const unsigned char* ip = unpack_b<B>(in, n, out);
        prefixSumInPlace<MinusOne>(out, n, start);
        return ip;
    }
    else
    {
#endif
    uint32_t acc = start;
    uint32_t* end = out + (n & ~31u);
    while (out < end)
    {
        if constexpr (B == 16)
            in = unpack_delta_b16_block<MinusOne>(in, out, acc);
        else if constexpr (B == 8)
            in = unpack_delta_b8_block<MinusOne>(in, out, acc);
        else
            in = unpack_delta_block<B, 32, 0, MinusOne>(in, out, acc);
        acc += MinusOne ? 32 : 0; // shortcut: acc + 32 == out[31] after emit (delta1); delta0 needs no offset
        out += 32;
    }
    n &= 31u;
    if (n == 0u) return in;
    // Tail: unpack normally then prefix-sum
    const unsigned char* tail_end = unpack_table[B](in, n, out);
    for (unsigned i = 0; i < n; ++i)
    {
        acc += out[i] + (MinusOne ? 1u : 0u);
        out[i] = acc;
    }
    in = tail_end;
    return in;
#if ABPFOR_ARCH_X86
    }
#endif
}

using UnpackDeltaFn = const unsigned char* (*)(const unsigned char*, unsigned, uint32_t*, uint32_t);

const unsigned char* unpack32_delta(const unsigned char* in, unsigned n, uint32_t* out, unsigned b, uint32_t start);

} // namespace abpfor::detail::bitops
