"""Host proof: NaN-robust fp32 boundary classification via branch inversion.

New scheme (vs proto_boundary_cls.main2):
    gt  := NOT (v <= T)        where T = gtu_threshold(tb + 1)
    tie := (v <= T) AND (v >= lo)   with lo = lower_bound_float(tb) (unchanged)

T construction: predecessor (one fp32 ordered-key step below) of the old
inclusive lower bound, so for every REAL v:  NOT(v <= T)  ==  (v >= bound).
NaN-space bins return T = +inf: reals and +/-inf take the tie-check branch,
NaN (which fails every ordered compare) falls to gt.

Checks, over the adversarial value set + NaN patterns x threshold bins:
  1. finite/inf classification identical to integer coarse_bin compare
     (same benign -0/+0 exception as before; the old +inf-at-inf-bin
     reclassification must DISAPPEAR: inf now ties like the integer path)
  2. every +NaN classifies gt (matches histogram: +NaN keys above +inf)
  3. every -NaN classifies gt (intentional deviation from its bottom-bin
     histogram position; torch treats any NaN as top -- count separately)
"""

import numpy as np

from proto_boundary_cls import (
    COARSE_SHIFT,
    adversarial_values,
    coarse_bin,
    f16_val,
    from_key16,
    from_key32,
    lower_bound_float,
    to_key32,
)

HIST_BITS = 12


def gtu_threshold(b):
    """T such that NOT(v <= T) == (coarse_bin(v) >= b) for real v, and
    True for every NaN.  Mirrors the planned kernel code."""
    key = b << COARSE_SHIFT
    if key <= 0x03FF:
        # unreachable for hi bounds (tb >= 63 => key(tb+1) >= 0x400);
        # qNaN would make everything gt -- return it for safety symmetry
        return np.float32(np.nan)
    if key > 0xFC00:
        # NaN key space: only NaN may be gt
        return np.float32(np.inf)

    def to_val(okey):
        if okey < 0x03FF:
            return -np.inf
        if okey == 0x03FF:
            return -65536.0
        if okey == 0xFC00:
            return 65536.0
        return f16_val(from_key16(okey))

    m = np.float32(0.5) * (np.float32(to_val(key)) + np.float32(to_val(key - 1)))
    hb_hi = from_key16(key)
    if hb_hi & 1:
        # odd parity: old boundary = succ(m); its predecessor is m itself
        return np.float32(m)
    # even parity: old boundary = m; predecessor = one key step down
    mb = np.uint32(to_key32(np.float32(m).view(np.uint32)) - 1)
    return np.uint32(from_key32(int(mb))).view(np.float32)


def main():
    rng = np.random.default_rng(11)
    vals = adversarial_values(rng)
    nan_bits = [0x7FC00000, 0x7F800001, 0x7FFFFFFF, 0xFFC00000, 0xFF800001]
    print(f"{len(vals)} real values + {len(nan_bits)} NaN patterns")
    tbs = sorted(
        set(
            [63, 64, 2047, 2048, 4031, 4032]
            + [coarse_bin(np.uint32(v)) for v in vals[:400]]
            + list(rng.integers(63, 4033, size=60))
        )
    )
    tbs = [t for t in tbs if 63 <= t <= 4032]
    print(f"{len(tbs)} threshold bins under test")
    bad = benign_zero = 0
    for tb in tbs:
        lo = np.float32(lower_bound_float(tb))
        T = gtu_threshold(tb + 1)
        for vb in vals:
            f = np.uint32(vb).view(np.float32)
            b = coarse_bin(np.uint32(vb))
            gt_ref, tie_ref = b > tb, b == tb
            le = bool(f <= T)  # ordered; the kernel branch shape
            gt_new = not le
            tie_new = le and bool(f >= lo)
            if gt_new != gt_ref or tie_new != tie_ref:
                if f == 0.0:
                    benign_zero += 1
                    continue
                bad += 1
                if bad < 10:
                    print(
                        f"MISMATCH tb={tb} x={f} bits={vb:08x} bin={b} "
                        f"T={T} lo={lo} gt={gt_new}/{gt_ref} tie={tie_new}/{tie_ref}"
                    )
        for nb in nan_bits:
            f = np.uint32(nb).view(np.float32)
            le = bool(f <= T)
            if le:  # NaN must never take the tie-check branch
                bad += 1
                print(f"NAN NOT GT: tb={tb} bits={nb:08x} T={T}")
    print(
        "GTU-BOUNDARY:",
        "FAIL" if bad else "ALL EXACT",
        f"({bad} mismatches, {benign_zero} benign +/-0 crossings)",
    )


if __name__ == "__main__":
    main()
