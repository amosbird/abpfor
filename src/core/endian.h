#pragma once
// Endian detection and conversion utilities.
// Wire format is little-endian; these convert between LE wire and native host order.

#include <cstdint>
#include <cstring>

namespace abpfor::detail
{

// --- Endian detection ---
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__)
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define ABPFOR_BIG_ENDIAN 1
#else
#define ABPFOR_BIG_ENDIAN 0
#endif
#elif defined(__BIG_ENDIAN__) || defined(__ARMEB__) || defined(__THUMBEB__) || defined(__AARCH64EB__) ||               \
    defined(_MIPSEB) || defined(__MIPSEB) || defined(__MIPSEB__)
#define ABPFOR_BIG_ENDIAN 1
#else
#define ABPFOR_BIG_ENDIAN 0
#endif

// --- Byte swap ---
inline constexpr uint16_t bswap16(uint16_t v)
{
    return __builtin_bswap16(v);
}
inline constexpr uint32_t bswap32(uint32_t v)
{
    return __builtin_bswap32(v);
}
inline constexpr uint64_t bswap64(uint64_t v)
{
    return __builtin_bswap64(v);
}

// --- LE <-> native conversion ---
// On little-endian (including all x86): identity (no-op)
// On big-endian: byte swap
#if ABPFOR_BIG_ENDIAN
inline constexpr uint16_t leToNative(uint16_t v)
{
    return bswap16(v);
}
inline constexpr uint32_t leToNative(uint32_t v)
{
    return bswap32(v);
}
inline constexpr uint64_t leToNative(uint64_t v)
{
    return bswap64(v);
}
#else
inline constexpr uint16_t leToNative(uint16_t v)
{
    return v;
}
inline constexpr uint32_t leToNative(uint32_t v)
{
    return v;
}
inline constexpr uint64_t leToNative(uint64_t v)
{
    return v;
}
#endif
inline constexpr uint16_t nativeToLe(uint16_t v)
{
    return leToNative(v);
}
inline constexpr uint32_t nativeToLe(uint32_t v)
{
    return leToNative(v);
}
inline constexpr uint64_t nativeToLe(uint64_t v)
{
    return leToNative(v);
}

// --- Portable load/store (unaligned, LE wire format) ---
inline uint64_t loadU64(const unsigned char* in)
{
    uint64_t v;
    memcpy(&v, in, 8);
    return leToNative(v);
}
inline void storeU64(unsigned char* out, uint64_t v)
{
    v = nativeToLe(v);
    memcpy(out, &v, 8);
}
inline uint32_t loadU32(const unsigned char* in)
{
    uint32_t v;
    memcpy(&v, in, 4);
    return leToNative(v);
}
inline void storeU32(unsigned char* out, uint32_t v)
{
    v = nativeToLe(v);
    memcpy(out, &v, 4);
}
inline uint16_t loadU16(const unsigned char* in)
{
    uint16_t v;
    memcpy(&v, in, 2);
    return leToNative(v);
}
inline void storeU16(unsigned char* out, uint16_t v)
{
    v = nativeToLe(v);
    memcpy(out, &v, 2);
}

// --- Array copy with LE conversion (for B=32 special case) ---
inline void copyU32ArrayToLe(unsigned char* out, const uint32_t* in, unsigned n)
{
#if ABPFOR_BIG_ENDIAN
    for (unsigned i = 0; i < n; ++i)
    {
        uint32_t v = nativeToLe(in[i]);
        memcpy(out + i * 4u, &v, 4);
    }
#else
    memcpy(out, in, n * 4u);
#endif
}

inline void copyU32ArrayFromLe(uint32_t* out, const unsigned char* in, unsigned n)
{
#if ABPFOR_BIG_ENDIAN
    for (unsigned i = 0; i < n; ++i)
    {
        uint32_t v;
        memcpy(&v, in + i * 4u, 4);
        out[i] = leToNative(v);
    }
#else
    memcpy(out, in, n * 4u);
#endif
}

} // namespace abpfor::detail
