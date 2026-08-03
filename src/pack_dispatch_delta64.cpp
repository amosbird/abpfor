// Separate TU for fused delta+unpack64 dispatch table.
#include "core/pack64.h"

namespace abpfor::detail::bitops
{

static const UnpackDelta64Fn unpack_delta64_table[65] = {
    nullptr,
    &unpack_delta64_b<1>,
    &unpack_delta64_b<2>,
    &unpack_delta64_b<3>,
    &unpack_delta64_b<4>,
    &unpack_delta64_b<5>,
    &unpack_delta64_b<6>,
    &unpack_delta64_b<7>,
    &unpack_delta64_b<8>,
    &unpack_delta64_b<9>,
    &unpack_delta64_b<10>,
    &unpack_delta64_b<11>,
    &unpack_delta64_b<12>,
    &unpack_delta64_b<13>,
    &unpack_delta64_b<14>,
    &unpack_delta64_b<15>,
    &unpack_delta64_b<16>,
    &unpack_delta64_b<17>,
    &unpack_delta64_b<18>,
    &unpack_delta64_b<19>,
    &unpack_delta64_b<20>,
    &unpack_delta64_b<21>,
    &unpack_delta64_b<22>,
    &unpack_delta64_b<23>,
    &unpack_delta64_b<24>,
    &unpack_delta64_b<25>,
    &unpack_delta64_b<26>,
    &unpack_delta64_b<27>,
    &unpack_delta64_b<28>,
    &unpack_delta64_b<29>,
    &unpack_delta64_b<30>,
    &unpack_delta64_b<31>,
    &unpack_delta64_b<32>,
    &unpack_delta64_b<33>,
    &unpack_delta64_b<34>,
    &unpack_delta64_b<35>,
    &unpack_delta64_b<36>,
    &unpack_delta64_b<37>,
    &unpack_delta64_b<38>,
    &unpack_delta64_b<39>,
    &unpack_delta64_b<40>,
    &unpack_delta64_b<41>,
    &unpack_delta64_b<42>,
    &unpack_delta64_b<43>,
    &unpack_delta64_b<44>,
    &unpack_delta64_b<45>,
    &unpack_delta64_b<46>,
    &unpack_delta64_b<47>,
    &unpack_delta64_b<48>,
    &unpack_delta64_b<49>,
    &unpack_delta64_b<50>,
    &unpack_delta64_b<51>,
    &unpack_delta64_b<52>,
    &unpack_delta64_b<53>,
    &unpack_delta64_b<54>,
    &unpack_delta64_b<55>,
    &unpack_delta64_b<56>,
    &unpack_delta64_b<57>,
    &unpack_delta64_b<58>,
    &unpack_delta64_b<59>,
    &unpack_delta64_b<60>,
    &unpack_delta64_b<61>,
    &unpack_delta64_b<62>,
    &unpack_delta64_b<63>,
    &unpack_delta64_b<64>,
};

const unsigned char* unpack_delta64(const unsigned char* in, unsigned n, uint64_t* out, unsigned b, uint64_t start)
{
    if (b == 0u) [[unlikely]]
    {
        for (unsigned i = 0; i < n; ++i) out[i] = start + i + 1;
        return in;
    }
    return unpack_delta64_table[b](in, n, out, start);
}

} // namespace abpfor::detail::bitops
