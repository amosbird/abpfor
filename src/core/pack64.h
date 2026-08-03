#pragma once

// 64-bit optimized bitpack/unpack with dispatch tables.

#include "pack_internal.h"

#include <cstring>

namespace abpfor::detail::bitops
{

// --- Pack64: streaming accumulator ---

struct PackState64
{
    unsigned char* op;
    uint64_t w;
};

template <unsigned B, unsigned Base, size_t I>
static ABPFOR_INLINE PackState64 pack64_stream_one(const uint64_t* __restrict in, PackState64 s)
{
    constexpr unsigned idx = Base + static_cast<unsigned>(I);
    constexpr unsigned bitpos = static_cast<unsigned>(I) * B;
    constexpr unsigned sh = bitpos % 64u;
    constexpr unsigned end_bit = sh + B;

    if constexpr (sh == 0u)
    {
        s.w = in[idx];
        if constexpr (B == 64u)
        {
            storeU64Fast(s.op, s.w);
            s.op += 8u;
            s.w = 0;
        }
    }
    else if constexpr (end_bit <= 64u)
    {
        s.w |= in[idx] << sh;
        if constexpr (end_bit == 64u)
        {
            storeU64Fast(s.op, s.w);
            s.op += 8u;
            s.w = 0;
        }
    }
    else
    {
        s.w |= in[idx] << sh;
        storeU64Fast(s.op, s.w);
        s.op += 8u;
        s.w = in[idx] >> (64u - sh);
    }
    return s;
}

template <unsigned B, unsigned Base, size_t I0, size_t... Is>
static ABPFOR_INLINE PackState64 pack64_stream_fold(const uint64_t* __restrict in, PackState64 s,
                                                    std::index_sequence<I0, Is...>)
{
    s = pack64_stream_one<B, Base, I0>(in, s);
    if constexpr (sizeof...(Is) > 0)
        return pack64_stream_fold<B, Base>(in, s, std::index_sequence<Is...>{});
    else
        return s;
}

template <unsigned B, unsigned K, unsigned Base>
static ABPFOR_INLINE unsigned char* pack64_block(const uint64_t* __restrict in, unsigned char* __restrict out)
{
    constexpr unsigned total_bits = K * B;
    constexpr unsigned total_bytes = (total_bits + 7u) / 8u;
    constexpr unsigned tail_bits = total_bits % 64u;
    constexpr unsigned tail_bytes = tail_bits == 0u ? 0u : (tail_bits + 7u) / 8u;

    PackState64 s{out, 0};
    s = pack64_stream_fold<B, Base>(in, s, std::make_index_sequence<K>{});

    if constexpr (tail_bits > 0u)
    {
        if constexpr (tail_bytes == 8u)
        {
            storeU64Fast(s.op, s.w);
            s.op += 8u;
        }
        else
        {
            store_partial<tail_bytes>(s.op, s.w);
        }
    }

    return out + total_bytes;
}

template <unsigned B, unsigned N>
static ABPFOR_INLINE unsigned char* pack64_n_b(const uint64_t* __restrict in, unsigned char* __restrict out)
{
    static_assert(B >= 1 && B <= 64);
    static_assert(N >= 1 && N <= 32);

    if constexpr (B == 64)
    {
        std::memcpy(out, in, N * 8u);
        return out + N * 8u;
    }
    else if constexpr (N * B <= 32 && N * B > 0)
    {
        uint32_t w = 0;
        for (unsigned i = 0; i < N; ++i) w |= static_cast<uint32_t>(in[i]) << (i * B);
        constexpr unsigned total_bytes = (N * B + 7u) / 8u;
        if constexpr (total_bytes == 4u)
        {
            storeU32Fast(out, w);
        }
        else if constexpr (total_bytes == 3u)
        {
            storeU16Fast(out, static_cast<uint16_t>(w));
            out[2] = static_cast<unsigned char>(w >> 16);
        }
        else if constexpr (total_bytes == 2u)
        {
            storeU16Fast(out, static_cast<uint16_t>(w));
        }
        else
        {
            out[0] = static_cast<unsigned char>(w);
        }
        return out + total_bytes;
    }
    else if constexpr (B == 32)
    {
        unsigned char* op = out;
        for (unsigned i = 0; i < N; ++i)
        {
            storeU32Fast(op, static_cast<uint32_t>(in[i]));
            op += 4u;
        }
        return op;
    }
    else if constexpr (B == 16)
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
        return pack64_block<B, N, 0u>(in, out);
    }
}

template <unsigned B>
static ABPFOR_INLINE unsigned char* pack64_dispatch_n(const uint64_t* in, unsigned n, unsigned char* out)
{
    switch (n)
    {
    case 1u:
        return pack64_n_b<B, 1>(in, out);
    case 2u:
        return pack64_n_b<B, 2>(in, out);
    case 3u:
        return pack64_n_b<B, 3>(in, out);
    case 4u:
        return pack64_n_b<B, 4>(in, out);
    case 5u:
        return pack64_n_b<B, 5>(in, out);
    case 6u:
        return pack64_n_b<B, 6>(in, out);
    case 7u:
        return pack64_n_b<B, 7>(in, out);
    case 8u:
        return pack64_n_b<B, 8>(in, out);
    case 9u:
        return pack64_n_b<B, 9>(in, out);
    case 10u:
        return pack64_n_b<B, 10>(in, out);
    case 11u:
        return pack64_n_b<B, 11>(in, out);
    case 12u:
        return pack64_n_b<B, 12>(in, out);
    case 13u:
        return pack64_n_b<B, 13>(in, out);
    case 14u:
        return pack64_n_b<B, 14>(in, out);
    case 15u:
        return pack64_n_b<B, 15>(in, out);
    case 16u:
        return pack64_n_b<B, 16>(in, out);
    case 17u:
        return pack64_n_b<B, 17>(in, out);
    case 18u:
        return pack64_n_b<B, 18>(in, out);
    case 19u:
        return pack64_n_b<B, 19>(in, out);
    case 20u:
        return pack64_n_b<B, 20>(in, out);
    case 21u:
        return pack64_n_b<B, 21>(in, out);
    case 22u:
        return pack64_n_b<B, 22>(in, out);
    case 23u:
        return pack64_n_b<B, 23>(in, out);
    case 24u:
        return pack64_n_b<B, 24>(in, out);
    case 25u:
        return pack64_n_b<B, 25>(in, out);
    case 26u:
        return pack64_n_b<B, 26>(in, out);
    case 27u:
        return pack64_n_b<B, 27>(in, out);
    case 28u:
        return pack64_n_b<B, 28>(in, out);
    case 29u:
        return pack64_n_b<B, 29>(in, out);
    case 30u:
        return pack64_n_b<B, 30>(in, out);
    case 31u:
        return pack64_n_b<B, 31>(in, out);
    default:
        __builtin_unreachable();
    }
}

template <unsigned B> static unsigned char* pack64_b(const uint64_t* in, unsigned n, unsigned char* out)
{
    if constexpr (B == 64)
    {
        std::memcpy(out, in, n * 8u);
        return out + n * 8u;
    }
    else
    {
        const uint64_t* end = in + (n & ~31u);
        while (in < end)
        {
            out = pack64_n_b<B, 32>(in, out);
            in += 32;
        }
        n &= 31u;
        if (n == 0u) return out;
        return pack64_dispatch_n<B>(in, n, out);
    }
}

using Pack64Fn = unsigned char* (*)(const uint64_t*, unsigned, unsigned char*);

inline const Pack64Fn pack64_table[65] = {
    nullptr,       &pack64_b<1>,  &pack64_b<2>,  &pack64_b<3>,  &pack64_b<4>,  &pack64_b<5>,  &pack64_b<6>,
    &pack64_b<7>,  &pack64_b<8>,  &pack64_b<9>,  &pack64_b<10>, &pack64_b<11>, &pack64_b<12>, &pack64_b<13>,
    &pack64_b<14>, &pack64_b<15>, &pack64_b<16>, &pack64_b<17>, &pack64_b<18>, &pack64_b<19>, &pack64_b<20>,
    &pack64_b<21>, &pack64_b<22>, &pack64_b<23>, &pack64_b<24>, &pack64_b<25>, &pack64_b<26>, &pack64_b<27>,
    &pack64_b<28>, &pack64_b<29>, &pack64_b<30>, &pack64_b<31>, &pack64_b<32>, &pack64_b<33>, &pack64_b<34>,
    &pack64_b<35>, &pack64_b<36>, &pack64_b<37>, &pack64_b<38>, &pack64_b<39>, &pack64_b<40>, &pack64_b<41>,
    &pack64_b<42>, &pack64_b<43>, &pack64_b<44>, &pack64_b<45>, &pack64_b<46>, &pack64_b<47>, &pack64_b<48>,
    &pack64_b<49>, &pack64_b<50>, &pack64_b<51>, &pack64_b<52>, &pack64_b<53>, &pack64_b<54>, &pack64_b<55>,
    &pack64_b<56>, &pack64_b<57>, &pack64_b<58>, &pack64_b<59>, &pack64_b<60>, &pack64_b<61>, &pack64_b<62>,
    &pack64_b<63>, &pack64_b<64>,
};

inline unsigned char* pack64(const uint64_t* in, unsigned n, unsigned char* out, unsigned b)
{
    if (b == 0u) [[unlikely]]
        return out;
    return pack64_table[b](in, n, out);
}

// --- Unpack64 ---
// Strategy 1 (B<=32): Pre-loaded word array + fold expression emit
// Strategy 2 (33<=B<=63): Interleaved load/emit with C macro expansion

// Prevent SLP auto-vectorization for non-delta stores in the interleaved path.
static ABPFOR_INLINE void store_u64_noslp(uint64_t* out, unsigned idx, uint64_t val)
{
    volatile uint64_t* vout = out;
    vout[idx] = val;
}

// Strategy 1: word-array preload + fold (B <= 32)
template <unsigned B, unsigned Base, size_t I>
static ABPFOR_INLINE void unpack64_emit_one(const uint64_t* __restrict w, uint64_t* __restrict out)
{
    constexpr unsigned idx = Base + static_cast<unsigned>(I);
    constexpr unsigned bitpos = static_cast<unsigned>(I) * B;
    constexpr unsigned wi = bitpos / 64u;
    constexpr unsigned sh = bitpos % 64u;
    constexpr uint64_t mask = (B == 64u) ? ~0ull : ((1ull << B) - 1ull);

    uint64_t v;
    if constexpr (sh + B <= 64u)
        v = (w[wi] >> sh) & mask;
    else
        v = ((w[wi] >> sh) | (w[wi + 1u] << (64u - sh))) & mask;

    out[idx] = v;
}

template <unsigned B, unsigned Base, size_t... I>
static ABPFOR_INLINE void unpack64_emit(const uint64_t* __restrict w, uint64_t* __restrict out,
                                        std::index_sequence<I...>)
{
    (unpack64_emit_one<B, Base, I>(w, out), ...);
}

// Strategy 2: Byte-aligned load for 33 <= B <= 63
// Each element is loaded from its byte-aligned position: loadU64(in + bitpos/8) >> (bitpos%8)
// This minimizes cross-boundary loads (0 for B<=58, few for B>58).
#define ABPFOR_UNPACK64_ELEM(IDX)                                                                                      \
    do                                                                                                                 \
    {                                                                                                                  \
        if constexpr ((IDX) < K)                                                                                       \
        {                                                                                                              \
            constexpr unsigned bitpos_ = (IDX) * B;                                                                    \
            constexpr unsigned byte_off_ = bitpos_ / 8u;                                                               \
            constexpr unsigned bit_off_ = bitpos_ % 8u;                                                                \
            constexpr unsigned out_idx_ = Base + (IDX);                                                                \
            uint64_t v_;                                                                                               \
            if constexpr (bit_off_ + B <= 64u)                                                                         \
            {                                                                                                          \
                v_ = (loadU64Fast(in + byte_off_) >> bit_off_) & mask;                                                 \
            }                                                                                                          \
            else                                                                                                       \
            {                                                                                                          \
                uint64_t lo = loadU64Fast(in + byte_off_);                                                             \
                uint64_t hi = loadU64Fast(in + byte_off_ + 8u);                                                        \
                v_ = (lo >> bit_off_) | ((hi << (64u - bit_off_)) & mask);                                             \
            }                                                                                                          \
            store_u64_noslp(out, out_idx_, v_);                                                                        \
        }                                                                                                              \
    } while (0)

template <unsigned B, unsigned K, unsigned Base>
static ABPFOR_INLINE void unpack64_interleaved(const unsigned char* __restrict in, uint64_t* __restrict out)
{
    static_assert(B > 32u && B < 64u);
    constexpr uint64_t mask = (1ull << B) - 1ull;

    ABPFOR_UNPACK64_ELEM(0);
    ABPFOR_UNPACK64_ELEM(1);
    ABPFOR_UNPACK64_ELEM(2);
    ABPFOR_UNPACK64_ELEM(3);
    ABPFOR_UNPACK64_ELEM(4);
    ABPFOR_UNPACK64_ELEM(5);
    ABPFOR_UNPACK64_ELEM(6);
    ABPFOR_UNPACK64_ELEM(7);
    ABPFOR_UNPACK64_ELEM(8);
    ABPFOR_UNPACK64_ELEM(9);
    ABPFOR_UNPACK64_ELEM(10);
    ABPFOR_UNPACK64_ELEM(11);
    ABPFOR_UNPACK64_ELEM(12);
    ABPFOR_UNPACK64_ELEM(13);
    ABPFOR_UNPACK64_ELEM(14);
    ABPFOR_UNPACK64_ELEM(15);
    ABPFOR_UNPACK64_ELEM(16);
    ABPFOR_UNPACK64_ELEM(17);
    ABPFOR_UNPACK64_ELEM(18);
    ABPFOR_UNPACK64_ELEM(19);
    ABPFOR_UNPACK64_ELEM(20);
    ABPFOR_UNPACK64_ELEM(21);
    ABPFOR_UNPACK64_ELEM(22);
    ABPFOR_UNPACK64_ELEM(23);
    ABPFOR_UNPACK64_ELEM(24);
    ABPFOR_UNPACK64_ELEM(25);
    ABPFOR_UNPACK64_ELEM(26);
    ABPFOR_UNPACK64_ELEM(27);
    ABPFOR_UNPACK64_ELEM(28);
    ABPFOR_UNPACK64_ELEM(29);
    ABPFOR_UNPACK64_ELEM(30);
    ABPFOR_UNPACK64_ELEM(31);
}

#undef ABPFOR_UNPACK64_ELEM

// Block unpack: selects strategy based on B
template <unsigned B, unsigned K, unsigned Base>
static ABPFOR_INLINE const unsigned char* unpack64_block(const unsigned char* __restrict in, uint64_t* __restrict out)
{
    constexpr unsigned total_bits = K * B;
    constexpr unsigned total_bytes = (total_bits + 7u) / 8u;

    if constexpr (B > 32u && B < 64u)
    {
        unpack64_interleaved<B, K, Base>(in, out);
    }
    else
    {
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
            w[word_count - 1u] = loadU64Fast(ip);
        else
            w[word_count - 1u] = load_partial<last_bytes>(ip);

        unpack64_emit<B, Base>(w, out, std::make_index_sequence<K>{});
    }
    return in + total_bytes;
}

// Recursively unpack blocks of optimal size
template <unsigned B, unsigned N, unsigned Base>
static ABPFOR_INLINE const unsigned char* unpack64_blocks(const unsigned char* __restrict in, uint64_t* __restrict out)
{
    if constexpr (N == 0u)
    {
        return in;
    }
    else
    {
        constexpr unsigned block = choose_block_size(B, N);
        const unsigned char* ip = unpack64_block<B, block, Base>(in, out);
        if constexpr (N == block)
            return ip;
        else
            return unpack64_blocks<B, N - block, Base + block>(ip, out);
    }
}

// Top-level unpack: N elements at compile-time bit-width B
template <unsigned B, unsigned N>
static ABPFOR_INLINE const unsigned char* unpack64_n_b(const unsigned char* __restrict in, uint64_t* __restrict out)
{
    static_assert(B >= 1 && B <= 64);
    static_assert(N >= 1 && N <= 32);

    // Byte-aligned fast paths
    if constexpr (B == 64)
    {
        std::memcpy(out, in, N * 8u);
        return in + N * 8u;
    }
    else if constexpr (B == 32)
    {
        const unsigned char* ip = in;
        for (unsigned i = 0; i < N; ++i)
        {
            out[i] = loadU32Fast(ip);
            ip += 4u;
        }
        return ip;
    }
    else if constexpr (B == 16)
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
        for (unsigned i = 0; i < N; ++i) out[i] = *ip++;
        return ip;
    }
    else
    {
        // General bitpacked path (non-byte-aligned)
        return unpack64_blocks<B, N, 0u>(in, out);
    }
}

// Tail dispatch: runtime n -> compile-time N
template <unsigned B>
static ABPFOR_INLINE const unsigned char* unpack64_dispatch_n(const unsigned char* in, unsigned n, uint64_t* out)
{
    switch (n)
    {
        // clang-format off
        case  1: return unpack64_n_b<B,  1>(in, out);
        case  2: return unpack64_n_b<B,  2>(in, out);
        case  3: return unpack64_n_b<B,  3>(in, out);
        case  4: return unpack64_n_b<B,  4>(in, out);
        case  5: return unpack64_n_b<B,  5>(in, out);
        case  6: return unpack64_n_b<B,  6>(in, out);
        case  7: return unpack64_n_b<B,  7>(in, out);
        case  8: return unpack64_n_b<B,  8>(in, out);
        case  9: return unpack64_n_b<B,  9>(in, out);
        case 10: return unpack64_n_b<B, 10>(in, out);
        case 11: return unpack64_n_b<B, 11>(in, out);
        case 12: return unpack64_n_b<B, 12>(in, out);
        case 13: return unpack64_n_b<B, 13>(in, out);
        case 14: return unpack64_n_b<B, 14>(in, out);
        case 15: return unpack64_n_b<B, 15>(in, out);
        case 16: return unpack64_n_b<B, 16>(in, out);
        case 17: return unpack64_n_b<B, 17>(in, out);
        case 18: return unpack64_n_b<B, 18>(in, out);
        case 19: return unpack64_n_b<B, 19>(in, out);
        case 20: return unpack64_n_b<B, 20>(in, out);
        case 21: return unpack64_n_b<B, 21>(in, out);
        case 22: return unpack64_n_b<B, 22>(in, out);
        case 23: return unpack64_n_b<B, 23>(in, out);
        case 24: return unpack64_n_b<B, 24>(in, out);
        case 25: return unpack64_n_b<B, 25>(in, out);
        case 26: return unpack64_n_b<B, 26>(in, out);
        case 27: return unpack64_n_b<B, 27>(in, out);
        case 28: return unpack64_n_b<B, 28>(in, out);
        case 29: return unpack64_n_b<B, 29>(in, out);
        case 30: return unpack64_n_b<B, 30>(in, out);
        case 31: return unpack64_n_b<B, 31>(in, out);
    // clang-format on
    default:
        __builtin_unreachable();
    }
}

// Entry point per bit-width
template <unsigned B> static const unsigned char* unpack64_b(const unsigned char* in, unsigned n, uint64_t* out)
{
    const unsigned char* ret = in + ((static_cast<uint64_t>(n) * B + 7u) / 8u);

    if constexpr (B == 64)
    {
        std::memcpy(out, in, n * 8u);
        return ret;
    }
    else
    {
        // Main loop: 32-element blocks
        uint64_t* end = out + (n & ~31u);
        while (out < end)
        {
            in = unpack64_n_b<B, 32>(in, out);
            out += 32;
        }

        n &= 31u;
        if (n == 0u) return ret;
        unpack64_dispatch_n<B>(in, n, out);
        return ret;
    }
}

using Unpack64Fn = const unsigned char* (*)(const unsigned char*, unsigned, uint64_t*);

inline const Unpack64Fn unpack64_table[65] = {
    nullptr,         &unpack64_b<1>,  &unpack64_b<2>,  &unpack64_b<3>,  &unpack64_b<4>,  &unpack64_b<5>,
    &unpack64_b<6>,  &unpack64_b<7>,  &unpack64_b<8>,  &unpack64_b<9>,  &unpack64_b<10>, &unpack64_b<11>,
    &unpack64_b<12>, &unpack64_b<13>, &unpack64_b<14>, &unpack64_b<15>, &unpack64_b<16>, &unpack64_b<17>,
    &unpack64_b<18>, &unpack64_b<19>, &unpack64_b<20>, &unpack64_b<21>, &unpack64_b<22>, &unpack64_b<23>,
    &unpack64_b<24>, &unpack64_b<25>, &unpack64_b<26>, &unpack64_b<27>, &unpack64_b<28>, &unpack64_b<29>,
    &unpack64_b<30>, &unpack64_b<31>, &unpack64_b<32>, &unpack64_b<33>, &unpack64_b<34>, &unpack64_b<35>,
    &unpack64_b<36>, &unpack64_b<37>, &unpack64_b<38>, &unpack64_b<39>, &unpack64_b<40>, &unpack64_b<41>,
    &unpack64_b<42>, &unpack64_b<43>, &unpack64_b<44>, &unpack64_b<45>, &unpack64_b<46>, &unpack64_b<47>,
    &unpack64_b<48>, &unpack64_b<49>, &unpack64_b<50>, &unpack64_b<51>, &unpack64_b<52>, &unpack64_b<53>,
    &unpack64_b<54>, &unpack64_b<55>, &unpack64_b<56>, &unpack64_b<57>, &unpack64_b<58>, &unpack64_b<59>,
    &unpack64_b<60>, &unpack64_b<61>, &unpack64_b<62>, &unpack64_b<63>, &unpack64_b<64>,
};

inline const unsigned char* unpack64(const unsigned char* in, unsigned n, uint64_t* out, unsigned b)
{
    if (b == 0u) [[unlikely]]
    {
        std::fill(out, out + n, uint64_t(0));
        return in;
    }
    return unpack64_table[b](in, n, out);
}

// --- Fused delta + unpack64 ---

// Strategy 1 (B <= 32): word-array preload + fold with delta accumulation
template <unsigned B, unsigned Base, size_t I>
static ABPFOR_INLINE void unpack_delta64_emit_one(const uint64_t* __restrict w, uint64_t* __restrict out, uint64_t& acc)
{
    constexpr unsigned idx = Base + static_cast<unsigned>(I);
    constexpr unsigned bitpos = static_cast<unsigned>(I) * B;
    constexpr unsigned wi = bitpos / 64u;
    constexpr unsigned sh = bitpos % 64u;
    constexpr uint64_t mask = (B == 64u) ? ~0ull : ((1ull << B) - 1ull);

    uint64_t v = w[wi] >> sh;
    if constexpr (sh + B > 64u) v |= w[wi + 1u] << (64u - sh);
    acc += v & mask;
    out[idx] = acc + uint64_t(idx + 1u);
}

template <unsigned B, unsigned Base, size_t... I>
static ABPFOR_INLINE void unpack_delta64_emit(const uint64_t* __restrict w, uint64_t* __restrict out, uint64_t& acc,
                                              std::index_sequence<I...>)
{
    (unpack_delta64_emit_one<B, Base, I>(w, out, acc), ...);
}

template <unsigned B, unsigned K, unsigned Base>
static ABPFOR_INLINE const unsigned char* unpack_delta64_block_word(const unsigned char* __restrict in,
                                                                    uint64_t* __restrict out, uint64_t& acc)
{
    constexpr unsigned total_bits = K * B;
    constexpr unsigned total_bytes = (total_bits + 7u) / 8u;
    constexpr unsigned word_count = (total_bits + 63u) / 64u;

    uint64_t w[word_count];
    for (unsigned i = 0; i < word_count; ++i) w[i] = loadU64Fast(in + i * 8u);
    unpack_delta64_emit<B, Base>(w, out, acc, std::make_index_sequence<K>{});
    return in + total_bytes;
}

// Strategy 2 (33 <= B <= 63): byte-aligned load with delta accumulation
#define ABPFOR_UNPACK_DELTA64_ELEM(IDX)                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        if constexpr ((IDX) < 32u)                                                                                     \
        {                                                                                                              \
            constexpr unsigned bitpos_ = (IDX) * B;                                                                    \
            constexpr unsigned byte_off_ = bitpos_ / 8u;                                                               \
            constexpr unsigned bit_off_ = bitpos_ % 8u;                                                                \
            uint64_t v_;                                                                                               \
            if constexpr (bit_off_ + B <= 64u)                                                                         \
            {                                                                                                          \
                v_ = (loadU64Fast(in + byte_off_) >> bit_off_) & mask;                                                 \
            }                                                                                                          \
            else                                                                                                       \
            {                                                                                                          \
                uint64_t lo = loadU64Fast(in + byte_off_);                                                             \
                uint64_t hi = loadU64Fast(in + byte_off_ + 8u);                                                        \
                v_ = (lo >> bit_off_) | ((hi << (64u - bit_off_)) & mask);                                             \
            }                                                                                                          \
            acc += v_;                                                                                                 \
            out[(IDX)] = acc + uint64_t((IDX) + 1u);                                                                   \
        }                                                                                                              \
    } while (0)

template <unsigned B>
static ABPFOR_INLINE void unpack_delta64_block_interleaved(const unsigned char* __restrict in, uint64_t* __restrict out,
                                                           uint64_t& acc)
{
    static_assert(B > 32u && B < 64u);
    constexpr uint64_t mask = (1ull << B) - 1ull;

    ABPFOR_UNPACK_DELTA64_ELEM(0);
    ABPFOR_UNPACK_DELTA64_ELEM(1);
    ABPFOR_UNPACK_DELTA64_ELEM(2);
    ABPFOR_UNPACK_DELTA64_ELEM(3);
    ABPFOR_UNPACK_DELTA64_ELEM(4);
    ABPFOR_UNPACK_DELTA64_ELEM(5);
    ABPFOR_UNPACK_DELTA64_ELEM(6);
    ABPFOR_UNPACK_DELTA64_ELEM(7);
    ABPFOR_UNPACK_DELTA64_ELEM(8);
    ABPFOR_UNPACK_DELTA64_ELEM(9);
    ABPFOR_UNPACK_DELTA64_ELEM(10);
    ABPFOR_UNPACK_DELTA64_ELEM(11);
    ABPFOR_UNPACK_DELTA64_ELEM(12);
    ABPFOR_UNPACK_DELTA64_ELEM(13);
    ABPFOR_UNPACK_DELTA64_ELEM(14);
    ABPFOR_UNPACK_DELTA64_ELEM(15);
    ABPFOR_UNPACK_DELTA64_ELEM(16);
    ABPFOR_UNPACK_DELTA64_ELEM(17);
    ABPFOR_UNPACK_DELTA64_ELEM(18);
    ABPFOR_UNPACK_DELTA64_ELEM(19);
    ABPFOR_UNPACK_DELTA64_ELEM(20);
    ABPFOR_UNPACK_DELTA64_ELEM(21);
    ABPFOR_UNPACK_DELTA64_ELEM(22);
    ABPFOR_UNPACK_DELTA64_ELEM(23);
    ABPFOR_UNPACK_DELTA64_ELEM(24);
    ABPFOR_UNPACK_DELTA64_ELEM(25);
    ABPFOR_UNPACK_DELTA64_ELEM(26);
    ABPFOR_UNPACK_DELTA64_ELEM(27);
    ABPFOR_UNPACK_DELTA64_ELEM(28);
    ABPFOR_UNPACK_DELTA64_ELEM(29);
    ABPFOR_UNPACK_DELTA64_ELEM(30);
    ABPFOR_UNPACK_DELTA64_ELEM(31);
}

#undef ABPFOR_UNPACK_DELTA64_ELEM

// Top-level per-B: process 32-element blocks, tail uses plain unpack + prefix-sum
template <unsigned B>
static const unsigned char* unpack_delta64_b(const unsigned char* in, unsigned n, uint64_t* out, uint64_t start)
{
    uint64_t acc = start;
    uint64_t* end = out + (n & ~31u);

    if constexpr (B == 64)
    {
        // shortcut: memcpy blocks then prefix-sum; no bit extraction needed
        while (out < end)
        {
            std::memcpy(out, in, 32u * 8u);
            for (unsigned i = 0; i < 32u; ++i)
            {
                acc += out[i] + 1u;
                out[i] = acc;
            }
            in += 32u * 8u;
            out += 32;
        }
    }
    else if constexpr (B == 32)
    {
        while (out < end)
        {
            for (unsigned i = 0; i < 32u; ++i)
            {
                acc += static_cast<uint64_t>(loadU32Fast(in));
                out[i] = acc + (i + 1u);
                in += 4u;
            }
            acc = out[31];
            out += 32;
        }
    }
    else if constexpr (B == 16)
    {
        while (out < end)
        {
            for (unsigned i = 0; i < 32u; ++i)
            {
                acc += static_cast<uint64_t>(loadU16Fast(in));
                out[i] = acc + (i + 1u);
                in += 2u;
            }
            acc = out[31];
            out += 32;
        }
    }
    else if constexpr (B == 8)
    {
        while (out < end)
        {
            for (unsigned i = 0; i < 32u; ++i)
            {
                acc += static_cast<uint64_t>(in[i]);
                out[i] = acc + (i + 1u);
            }
            acc = out[31];
            in += 32u;
            out += 32;
        }
    }
    else if constexpr (B > 32u)
    {
        while (out < end)
        {
            unpack_delta64_block_interleaved<B>(in, out, acc);
            in += (32u * B + 7u) / 8u;
            acc = out[31]; // read back from output after block
            out += 32;
        }
    }
    else
    {
        while (out < end)
        {
            in = unpack_delta64_block_word<B, 32, 0>(in, out, acc);
            acc = out[31]; // read back from output after block
            out += 32;
        }
    }

    n &= 31u;
    if (n == 0u) return in;
    // Tail: unpack normally then prefix-sum
    const unsigned char* tail_end = unpack64_table[B](in, n, out);
    for (unsigned i = 0; i < n; ++i)
    {
        acc += out[i] + 1u;
        out[i] = acc;
    }
    return tail_end;
}

using UnpackDelta64Fn = const unsigned char* (*)(const unsigned char*, unsigned, uint64_t*, uint64_t);

const unsigned char* unpack_delta64(const unsigned char* in, unsigned n, uint64_t* out, unsigned b, uint64_t start);

} // namespace abpfor::detail::bitops
