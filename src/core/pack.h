#pragma once

// abpfor bitpacking engine — public API.
//
// Two optimized implementations:
//   detail::bitops::pack32/unpack32 — unrolled scalar for uint32_t
//   detail::bitops::pack64/unpack64 — streaming accumulator for uint64_t
//
// Public API: pack<T>, unpack<T> — dispatch to the optimized implementations.

#include "pack32.h"
#include "pack64.h"

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace abpfor
{

// ---------------------------------------------------------------------------
// Public API — thin wrappers over detail::bitops::pack32/unpack32/pack64/unpack64
// ---------------------------------------------------------------------------

template <typename T> uint8_t* pack(const T* __restrict in, unsigned n, uint8_t* __restrict out, unsigned bits)
{
    static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>);
    if constexpr (std::is_same_v<T, uint32_t>)
    {
        return reinterpret_cast<uint8_t*>(detail::bitops::pack32(in, n, reinterpret_cast<unsigned char*>(out), bits));
    }
    else
    {
        return reinterpret_cast<uint8_t*>(detail::bitops::pack64(in, n, reinterpret_cast<unsigned char*>(out), bits));
    }
}

template <typename T> const uint8_t* unpack(const uint8_t* __restrict in, unsigned n, T* __restrict out, unsigned bits)
{
    static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>);
    if constexpr (std::is_same_v<T, uint32_t>)
    {
        return reinterpret_cast<const uint8_t*>(
            detail::bitops::unpack32(reinterpret_cast<const unsigned char*>(in), n, out, bits));
    }
    else
    {
        return reinterpret_cast<const uint8_t*>(
            detail::bitops::unpack64(reinterpret_cast<const unsigned char*>(in), n, out, bits));
    }
}

// Fused unpack + delta
inline const uint8_t* unpack_delta(const uint8_t* __restrict in, unsigned n, uint32_t* __restrict out, unsigned bits,
                                   uint32_t start)
{
    return reinterpret_cast<const uint8_t*>(
        detail::bitops::unpack32_delta(reinterpret_cast<const unsigned char*>(in), n, out, bits, start));
}

inline const uint8_t* unpack_delta(const uint8_t* __restrict in, unsigned n, uint64_t* __restrict out, unsigned bits,
                                   uint64_t start)
{
    return reinterpret_cast<const uint8_t*>(
        detail::bitops::unpack_delta64(reinterpret_cast<const unsigned char*>(in), n, out, bits, start));
}

} // namespace abpfor
