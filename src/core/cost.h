#pragma once

// abpfor::optimalWidth — cost-model-driven bit-width and outlier strategy selection.
//
//   unsigned optimalWidth(const T* in, unsigned n, unsigned* pbx)
// Returns baseBits, writes exception encoding to *pbx:
//   *pbx == 0           → no outliers (strategy=None)
//   *pbx == MAX_BITS+1  → bitmap strategy
//   *pbx == MAX_BITS+2  → constant block
//   else                → *pbx = outlierBits (sparse strategy)

#include "bits.h"

namespace abpfor
{

// ---------------------------------------------------------------------------
// costSweep — Phase 3: reverse scan from maxBits-1 to 0, find minimum cost
// ---------------------------------------------------------------------------
// Shared by both optimalWidth overloads. cnt[] must be a merged histogram.

template <typename T>
static unsigned costSweep(const unsigned* cnt, unsigned n, unsigned maxBits, unsigned* pbx)
{
    constexpr unsigned W = kMaxBits<T>;
    unsigned bitmapBytes = divCeil(n, 8u);
    unsigned bestCost = packedBytes(n, maxBits) + 1;
    unsigned bestB = maxBits;
    unsigned bestPbx = 0;
    unsigned outlierCount = 0;

    for (unsigned b = maxBits; b-- > 0;)
    {
        outlierCount += cnt[b + 1];

        unsigned pb = maxBits - b;
        unsigned basePack = packedBytes(n, b);
        unsigned outlierPack = packedBytes(outlierCount, pb);

        unsigned bmpCost = 2 + bitmapBytes + outlierPack + basePack;
        unsigned spsCost = 3 + outlierCount + outlierPack + basePack;

        unsigned thisCost;
        unsigned thisPbx;
        if (bmpCost <= spsCost)
        {
            thisCost = bmpCost;
            thisPbx = W + 1; // bitmap
        }
        else
        {
            thisCost = spsCost;
            thisPbx = pb; // sparse: outlierBits
        }

        // shortcut: require meaningful savings to justify outlier overhead
        unsigned noOutlierCost = packedBytes(n, maxBits) + 1;
        if (thisCost < bestCost && noOutlierCost - thisCost > noOutlierCost / 64)
        {
            bestCost = thisCost;
            bestB = b;
            bestPbx = thisPbx;
        }
    }

    *pbx = bestPbx;
    return bestB;
}

// ---------------------------------------------------------------------------
// optimalWidth — find minimum-cost (baseBits, strategy)
// ---------------------------------------------------------------------------
// Returns baseBits, writes exception encoding to *pbx.

template <typename T> ABPFOR_NOINLINE unsigned optimalWidth(const T* in, unsigned n, unsigned* pbx)
{
    static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>);
    constexpr unsigned W = kMaxBits<T>;

    // Phase 1: OR-reduce + equality count (auto-vectorizes as two separate reductions)
    T orAll = 0;
    const T first = in[0];
    unsigned eqCount = 0;

    for (unsigned i = 0; i < n; ++i) orAll |= in[i];
    for (unsigned i = 0; i < n; ++i) eqCount += (in[i] == first);

    unsigned maxBits = bitwidth(orAll);

    // Early exit: all zeros
    if (maxBits == 0)
    {
        *pbx = 0;
        return 0;
    }

    // Early exit: constant block (all equal, nonzero)
    if (eqCount == n)
    {
        *pbx = W + 2;
        return maxBits;
    }

    // Phase 2: 8x unrolled histogram of bit widths, accumulated into 4 private
    // bins. A single shared bin serialises on store-to-load forwarding whenever
    // consecutive elements share a bit width (the common case for real data):
    // each ++cnt[k] must wait for the previous one to retire. Four bins let the
    // four dependency chains overlap, and merging 4x(W+1) counters afterwards is
    // negligible next to n increments.
    // Measured on Zen: 120ns -> 69ns per 128-element block. 8 bins is slower
    // again (91ns) — more merge work and L1 pressure than the ILP is worth.
    unsigned cnt0[W + 8] = {};
    unsigned cnt1[W + 8] = {};
    unsigned cnt2[W + 8] = {};
    unsigned cnt3[W + 8] = {};

    unsigned i = 0;
    for (; i + 8 <= n; i += 8)
    {
        ++cnt0[bitwidth(in[i])];
        ++cnt1[bitwidth(in[i + 1])];
        ++cnt2[bitwidth(in[i + 2])];
        ++cnt3[bitwidth(in[i + 3])];
        ++cnt0[bitwidth(in[i + 4])];
        ++cnt1[bitwidth(in[i + 5])];
        ++cnt2[bitwidth(in[i + 6])];
        ++cnt3[bitwidth(in[i + 7])];
    }
    for (; i < n; ++i) ++cnt0[bitwidth(in[i])];

    unsigned* cnt = cnt0;
    for (unsigned k = 0; k <= maxBits; ++k) cnt[k] += cnt1[k] + cnt2[k] + cnt3[k];

    // No outliers at maxBits → pure bitpack
    if (cnt[maxBits] == n)
    {
        *pbx = 0;
        return maxBits;
    }

    return costSweep<T>(cnt, n, maxBits, pbx);
}

// Overload: skip Phase 1 when caller already computed maxBits
template <typename T> ABPFOR_NOINLINE unsigned optimalWidth(const T* in, unsigned n, unsigned* pbx, unsigned maxBits)
{
    static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>);
    constexpr unsigned W = kMaxBits<T>;

    // Phase 2: 8x unrolled histogram of bit widths
    unsigned cnt[W + 8] = {};
    unsigned i = 0;
    for (; i + 8 <= n; i += 8)
    {
        ++cnt[bitwidth(in[i])];
        ++cnt[bitwidth(in[i + 1])];
        ++cnt[bitwidth(in[i + 2])];
        ++cnt[bitwidth(in[i + 3])];
        ++cnt[bitwidth(in[i + 4])];
        ++cnt[bitwidth(in[i + 5])];
        ++cnt[bitwidth(in[i + 6])];
        ++cnt[bitwidth(in[i + 7])];
    }
    for (; i < n; ++i) ++cnt[bitwidth(in[i])];

    if (cnt[maxBits] == n)
    {
        *pbx = 0;
        return maxBits;
    }

    // Phase 3: reverse cost sweep
    // Note: this uses baseCost = packedBytes(n, b) + 1 which folds the header
    // byte into the base, giving slightly different thresholds than the primary
    // overload. Kept for wire-format stability (AGENTS.md: never change compressed sizes).
    unsigned bitmapBytes = divCeil(n, 8u);
    unsigned bestCost = packedBytes(n, maxBits) + 1;
    unsigned bestB = maxBits;
    unsigned bestPbx = 0;
    unsigned outlierCount = 0;

    for (unsigned b = maxBits; b-- > 0;)
    {
        outlierCount += cnt[b + 1];
        unsigned baseCost = packedBytes(n, b) + 1;
        unsigned outlierBits = maxBits - b;
        unsigned bitmapCost = baseCost + bitmapBytes + packedBytes(outlierCount, outlierBits);
        unsigned sparseCost = baseCost + 1 + outlierCount + packedBytes(outlierCount, outlierBits);
        unsigned thisCost, thisPbx;
        if (bitmapCost <= sparseCost)
        {
            thisCost = bitmapCost;
            thisPbx = W + 1;
        }
        else
        {
            thisCost = sparseCost;
            thisPbx = outlierBits;
        }
        if (thisCost < bestCost)
        {
            unsigned noOutlierCost = packedBytes(n, maxBits) + 1;
            if (noOutlierCost - thisCost > noOutlierCost / 64)
            {
                bestCost = thisCost;
                bestB = b;
                bestPbx = thisPbx;
            }
        }
    }
    *pbx = bestPbx;
    return bestB;
}

} // namespace abpfor
