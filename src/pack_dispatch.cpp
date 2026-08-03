// Separate TU for pack/unpack dispatch — prevents compiler from
// folding template-expanded straight-line code back into loops.
#include "core/pack.h"

namespace abpfor
{

uint8_t* pack32(const uint32_t* in, unsigned n, uint8_t* out, unsigned bits)
{
    return reinterpret_cast<uint8_t*>(detail::bitops::pack32(in, n, reinterpret_cast<unsigned char*>(out), bits));
}

const uint8_t* unpack32(const uint8_t* in, unsigned n, uint32_t* out, unsigned bits)
{
    return reinterpret_cast<const uint8_t*>(
        detail::bitops::unpack32(reinterpret_cast<const unsigned char*>(in), n, out, bits));
}

} // namespace abpfor
