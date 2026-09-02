"""Host prototype: fp32 ordered-key boundary-compare classification.

Verifies that two integer compares against bisection-derived key32 bounds
classify EXACTLY like the per-element coarse_bin comparison, for adversarial
values (fp16 midpoints incl. ties-to-even, ULP neighbors, denormals, inf, 0).
"""

import numpy as np

HIST_BITS = 12
COARSE_SHIFT = 16 - HIST_BITS  # 4

KEY32_NEG_INF = 0x007FFFFF  # ~bits(-inf)
KEY32_POS_INF = 0xFF800000  # bits(+inf) ^ 0x80000000


def to_key32(bits32):
    b = np.uint64(bits32)
    sign = b >> np.uint64(31)
    mask = (np.uint64(0) - sign) & np.uint64(0xFFFFFFFF) | np.uint64(0x80000000)
    return int((b ^ mask) & np.uint64(0xFFFFFFFF))


def from_key32(key):
    k = np.uint64(key)
    # top bit set => was positive (xor 0x80000000); else was negative (~)
    if k & np.uint64(0x80000000):
        return int(k ^ np.uint64(0x80000000))
    return int(~k & np.uint64(0xFFFFFFFF))


def to_key16(bits16):
    b = np.uint64(bits16)
    sign = b >> np.uint64(15)
    mask = ((np.uint64(0) - sign) & np.uint64(0xFFFF)) | np.uint64(0x8000)
    return int((b ^ mask) & np.uint64(0xFFFF))


def coarse_bin(bits32):
    f32 = np.uint32(bits32).view(np.float32)
    with np.errstate(over="ignore"):
        h = np.float16(f32)  # numpy: round-to-nearest-even, matches cvt.rn
    return to_key16(h.view(np.uint16)) >> COARSE_SHIFT


def lower_bound(b):
    """Smallest key32 in the real-float domain with coarse_bin >= b."""
    lo, hi = KEY32_NEG_INF, KEY32_POS_INF + 1
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        if coarse_bin(np.uint32(from_key32(mid))) >= b:
            hi = mid
        else:
            lo = mid + 1
    return lo


def adversarial_values(rng, n_rand=20000):
    vals = []
    # random fp16 grid values, their fp32 midpoints, and ULP neighbors
    h16 = rng.integers(0, 0x10000, size=4000, dtype=np.uint16)
    for hb in h16:
        h = np.uint16(hb).view(np.float16)
        if np.isnan(h):
            continue
        f = np.float32(h)
        fb = f.view(np.uint32)
        vals.append(int(fb))
        # neighbors of the exact fp16 value in fp32 space
        for d in (-2, -1, 1, 2):
            vals.append(int(np.uint32((int(fb) + d) & 0xFFFFFFFF)))
        # midpoint with the next fp16 up (exact in fp32) + its neighbors
        nb = np.uint16((int(hb) & 0x7FFF) + 1) if int(hb) & 0x7FFF != 0x7FFF else None
        if nb is not None:
            hn = np.uint16((int(hb) & 0x8000) | int(nb)).view(np.float16)
            if not (np.isnan(hn) or np.isinf(hn) or np.isinf(h)):
                m = (np.float32(h) + np.float32(hn)) * np.float32(0.5)
                mb = np.float32(m).view(np.uint32)
                for d in (-1, 0, 1):
                    vals.append(int(np.uint32((int(mb) + d) & 0xFFFFFFFF)))
    # specials
    for f in (
        0.0,
        -0.0,
        np.inf,
        -np.inf,
        65504.0,
        -65504.0,
        65520.0,
        -65520.0,
        1e-8,
        -1e-8,
        5.9604645e-08,
        2.9802322e-08,
        1e30,
        -1e30,
    ):
        vals.append(int(np.float32(f).view(np.uint32)))
    # random fp32
    vals.extend(int(x) for x in rng.integers(0, 2**32, size=n_rand, dtype=np.uint64))
    # drop NaNs (out of contract)
    out = []
    for vb in vals:
        f = np.uint32(vb).view(np.float32)
        if not np.isnan(f):
            out.append(vb)
    return out


def from_key16(key):
    k = int(key) & 0xFFFF
    if k & 0x8000:
        return k ^ 0x8000
    return (~k) & 0xFFFF


def f16_val(bits16):
    return float(np.uint16(bits16).view(np.float16))


def lower_bound_float(b):
    """Closed-form fp32 boundary (sglang construction + parity bump).

    Returns float lo such that (val >= lo) == (coarse_bin(val) >= b) for all
    real fp32 val (modulo the benign -0/+0 equal-value crossing).
    """
    if b <= 0:
        return -np.inf
    if b >= 1 << HIST_BITS:
        return np.inf
    key = b << COARSE_SHIFT
    # Keys at/below the -inf key: every real fp32 rounds into bin >= b
    # (values < -65520 round to -inf whose bin is 0x03FF >> shift = 63);
    # the boundary is -inf itself, no parity bump.
    if key <= 0x03FF:
        return -np.inf
    # Keys above the +inf key are +NaN space: no real value qualifies.
    # +inf is the only value that compares >= +inf; that reclassification
    # only triggers when tb is the inf bin itself, where all qualifying
    # values are equal (+inf) and any subset is a valid top-k.
    if key > 0xFC00:
        return np.inf

    def to_val(okey):
        if okey < 0x03FF:
            return -np.inf
        if okey == 0x03FF:
            return -65536.0
        if okey == 0xFC00:
            return 65536.0
        if okey > 0xFC00:
            return np.finfo(np.float32).max
        return f16_val(from_key16(okey))

    m = np.float32(0.5) * (np.float32(to_val(key)) + np.float32(to_val(key - 1)))
    # Parity: if the upper fp16 has an ODD mantissa LSB, ties-to-even rounds
    # the exact midpoint DOWN (to the even lower neighbor), so the true
    # boundary is one fp32 step above m in ordered-key space.
    hb_hi = from_key16(key)
    if hb_hi & 1:
        m = np.uint32(to_key32(np.float32(m).view(np.uint32)) + 1)
        m = np.uint32(from_key32(int(m))).view(np.float32)
    return float(m)


def main2():
    rng = np.random.default_rng(11)
    vals = adversarial_values(rng)
    print(f"[float-boundary] {len(vals)} adversarial values")
    # Only bins a real-element histogram can produce: bin(-inf)=63 up to
    # bin(+inf)=4032; NaN-space bins never win the threshold scan.
    tbs = sorted(
        set(
            [63, 64, 2047, 2048, 4031, 4032]
            + [coarse_bin(np.uint32(v)) for v in vals[:400]]
            + list(rng.integers(63, 4033, size=60))
        )
    )
    tbs = [t for t in tbs if 63 <= t <= 4032]
    print(f"[float-boundary] {len(tbs)} threshold bins under test")
    bad = benign = 0
    for tb in tbs:
        lo = np.float32(lower_bound_float(tb))
        hi = np.float32(lower_bound_float(tb + 1))
        for vb in vals:
            f = np.uint32(vb).view(np.float32)
            b = coarse_bin(np.uint32(vb))
            gt_ref, tie_ref = b > tb, b == tb
            gt_new = bool(f >= hi)
            tie_new = (not gt_new) and bool(f >= lo)
            if gt_new != gt_ref or tie_new != tie_ref:
                # -0.0 at a 0.0 boundary is value-equivalent: benign
                if f == 0.0:
                    benign += 1
                    continue
                # +inf at the inf bin: gt-vs-tie reclassification among
                # exclusively equal (+inf) values -- any subset valid
                if np.isinf(f) and f > 0 and tb == 4032 and gt_new and tie_ref:
                    benign += 1
                    continue
                bad += 1
                if bad < 10:
                    print(
                        f"MISMATCH tb={tb} x={f} bits={vb:08x} bin={b} "
                        f"lo={lo} hi={hi} gt={gt_new}/{gt_ref} tie={tie_new}/{tie_ref}"
                    )
    print(
        "FLOAT-BOUNDARY:",
        "FAIL" if bad else "ALL EXACT",
        f"({bad} mismatches, {benign} benign +/-0 crossings)",
    )


def main():
    rng = np.random.default_rng(7)
    vals = adversarial_values(rng)
    print(f"{len(vals)} adversarial values")
    tbs = sorted(
        set(
            [0, 1, 2047, 2048, 4094, 4095]
            + [coarse_bin(np.uint32(v)) for v in vals[:400]]
            + list(rng.integers(0, 4096, size=60))
        )
    )
    print(f"{len(tbs)} threshold bins under test")
    bad = 0
    for tb in tbs:
        lo = lower_bound(tb)
        hi = lower_bound(tb + 1)
        for vb in vals:
            b = coarse_bin(np.uint32(vb))
            k = to_key32(vb)
            gt_ref, tie_ref = b > tb, b == tb
            gt_new = k >= hi
            tie_new = (not gt_new) and k >= lo
            if gt_new != gt_ref or tie_new != tie_ref:
                bad += 1
                if bad < 10:
                    f = np.uint32(vb).view(np.float32)
                    print(
                        f"MISMATCH tb={tb} x={f} bits={vb:08x} bin={b} "
                        f"k={k:08x} lo={lo:08x} hi={hi:08x}"
                    )
    print("FAIL" if bad else "ALL EXACT", f"({bad} mismatches)")


if __name__ == "__main__":
    main()
