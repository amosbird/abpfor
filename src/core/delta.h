#pragma once

// abpfor::delta / abpfor::undelta — delta-1 encode and decode.
//
// delta:   out[i] = in[i] - prev - 1,  prev = in[i]
// undelta: out[i] = in[i] + prev + 1,  prev = out[i]   (prefix sum + 1)
//
// "minus 1" makes sorted sequences (gaps ≥ 1) produce non-negative deltas.

#include "bits.h"

namespace abpfor
{

template <typename T> ABPFOR_INLINE void delta(const T* __restrict in, unsigned n, T* __restrict out, T start)
{
    for (unsigned i = 0; i < n; ++i)
    {
        out[i] = in[i] - start - 1;
        start = in[i];
    }
}

template <typename T> ABPFOR_INLINE void undelta(T* __restrict data, unsigned n, T start)
{
    for (unsigned i = 0; i < n; ++i)
    {
        data[i] += start + 1;
        start = data[i];
    }
}

// delta0 / undelta0 — plain delta (no minus-1), for non-decreasing sequences (gaps >= 0).

template <typename T> ABPFOR_INLINE void delta0(const T* __restrict in, unsigned n, T* __restrict out, T start)
{
    for (unsigned i = 0; i < n; ++i)
    {
        out[i] = in[i] - start;
        start = in[i];
    }
}

template <typename T> ABPFOR_INLINE void undelta0(T* __restrict data, unsigned n, T& carry)
{
    for (unsigned i = 0; i < n; ++i)
    {
        carry += data[i];
        data[i] = carry;
    }
}

} // namespace abpfor
