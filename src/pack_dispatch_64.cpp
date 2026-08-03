// Separate TU for 64-bit pack dispatch only
#include "core/pack.h"

namespace abpfor
{

uint8_t* pack64(const uint64_t* in, unsigned n, uint8_t* out, unsigned bits)
{
    return reinterpret_cast<uint8_t*>(detail::bitops::pack64(in, n, reinterpret_cast<unsigned char*>(out), bits));
}

const uint8_t* unpack64(const uint8_t* in, unsigned n, uint64_t* out, unsigned bits)
{
    return reinterpret_cast<const uint8_t*>(
        detail::bitops::unpack64(reinterpret_cast<const unsigned char*>(in), n, out, bits));
}

} // namespace abpfor
