// Separate TU for fused delta+unpack32 dispatch table.
// Keeps 32 template instantiations out of every includer's icache.
#include "core/pack32.h"

namespace abpfor::detail::bitops
{

static const UnpackDeltaFn unpack_delta_table[33] = {nullptr,
                                                     &unpack_delta_b<1, true>,
                                                     &unpack_delta_b<2, true>,
                                                     &unpack_delta_b<3, true>,
                                                     &unpack_delta_b<4, true>,
                                                     &unpack_delta_b<5, true>,
                                                     &unpack_delta_b<6, true>,
                                                     &unpack_delta_b<7, true>,
                                                     &unpack_delta_b<8, true>,
                                                     &unpack_delta_b<9, true>,
                                                     &unpack_delta_b<10, true>,
                                                     &unpack_delta_b<11, true>,
                                                     &unpack_delta_b<12, true>,
                                                     &unpack_delta_b<13, true>,
                                                     &unpack_delta_b<14, true>,
                                                     &unpack_delta_b<15, true>,
                                                     &unpack_delta_b<16, true>,
                                                     &unpack_delta_b<17, true>,
                                                     &unpack_delta_b<18, true>,
                                                     &unpack_delta_b<19, true>,
                                                     &unpack_delta_b<20, true>,
                                                     &unpack_delta_b<21, true>,
                                                     &unpack_delta_b<22, true>,
                                                     &unpack_delta_b<23, true>,
                                                     &unpack_delta_b<24, true>,
                                                     &unpack_delta_b<25, true>,
                                                     &unpack_delta_b<26, true>,
                                                     &unpack_delta_b<27, true>,
                                                     &unpack_delta_b<28, true>,
                                                     &unpack_delta_b<29, true>,
                                                     &unpack_delta_b<30, true>,
                                                     &unpack_delta_b<31, true>,
                                                     &unpack_delta_b<32, true>};

const unsigned char* unpack32_delta(const unsigned char* in, unsigned n, uint32_t* out, unsigned b, uint32_t start)
{
    if (b == 0u) [[unlikely]]
    {
        for (unsigned i = 0; i < n; ++i) out[i] = start + i + 1;
        return in;
    }
    return unpack_delta_table[b](in, n, out, start);
}

} // namespace abpfor::detail::bitops
