"""
Density-tier suite for the Spiral Sesame Code.

Checks the four tiers (L0 standard .. L3 long-range) end to end: the capacity
table, the tier packed into the version byte, per-tier round trips through the
camera front end, rotation, occlusion and minimum-size tolerance, and -- the
part that must never regress -- that an L0 render is still bit-for-bit what it
was before tiers existed, so codes already printed in the world keep decoding.

Usage:
    python test_tiers.py           # everything
    python test_tiers.py capacity  # only groups whose name matches
"""

import hashlib
import math
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

from spiral_sesame_code import (
    DEFAULT_IMAGE_SIZE, TIER_TABLE, SpiralSesameDecoder, SpiralSesameEncoder,
    get_ring_params, pack_version_byte, tier_capacity, tier_geometry,
    unpack_version_byte,
)

OUTER_R = get_ring_params((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))[0]

# One payload per tier, sized to leave headroom under that tier's capacity but
# long enough to be a realistic URL at the coarse end.
PAYLOADS = {
    0: "https://www.bilibili.com/video/BV1GJ411x7h7/?spm_id_from=333.999.0.0",
    1: "https://b23.tv/BV1xx411c7mD",
    2: "https://x.io/wall",
    3: "https://x.io/screen",
}

# SHA-256 of the raw RGB array of `SpiralSesameEncoder().encode_text(...)`.
# Recorded from the pre-tier codec and re-verified against a HEAD worktree, so
# this is the guarantee that old readers still read new L0 codes.
L0_REF_PAYLOAD = "https://b23.tv/BV1xx411c7mD"
L0_REF_SHA256 = ("02e05a7863925bc5ce8063a6146a88717ce7e71bce11cd56714249a4"
                 "f8d9d119")

# Codes rendered before tiers existed. Tracked in the repo.
LEGACY_PNGS = ("demo_short_url.png", "demo_bilibili.png",
               "demo_xiaohongshu.png")


def paste_disc(img, roll=0.0):
    """Rotate onto a white canvas through a circular alpha mask. Rotating the
    square directly would leave the canvas corners visible just outside the
    seed annulus, and the front end reads that edge as part of the disc."""
    if roll:
        img = img.rotate(roll, resample=Image.BICUBIC,
                         fillcolor=(255, 255, 255))
    rgba = img.convert("RGBA")
    w, h = img.size
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot(xx - w / 2.0, yy - h / 2.0)
    rgba.putalpha(Image.fromarray(
        np.where(d <= OUTER_R + 1, 255, 0).astype(np.uint8)))
    canvas = Image.new("RGB", img.size, (255, 255, 255))
    canvas.paste(rgba, (0, 0), rgba)
    return canvas


def occlude(img, wedge_deg, r_frac=0.72):
    """White out one wedge of the seed ring, the shape a sticker or a hand
    leaves. `wedge_deg` of arc at radius `r_frac * OUTER_R`."""
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    w, h = out.size
    cx, cy = w / 2.0, h / 2.0
    r = r_frac * OUTER_R
    half = math.radians(wedge_deg / 2.0)
    pts = [(cx, cy)]
    for i in range(25):
        a = -half + 2 * half * i / 24.0
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    d.polygon(pts, fill=(255, 255, 255))
    return out


class Report:
    def __init__(self):
        self.failures = []

    def check(self, ok, label, detail=""):
        print(f"[{'OK' if ok else 'FAIL'}] {label}"
              + (f"  {detail}" if detail else ""))
        if not ok:
            self.failures.append(label)
        return ok


def group_capacity(rep):
    """The ladder has to be monotone in density and match the table the README
    publishes; a silently changed cell count breaks every printed code."""
    expected = {0: (1, 5, 30, 139), 1: (2, 3, 20, 40),
                2: (3, 2, 8, 29), 3: (4, 1, 8, 33)}
    for tier in sorted(TIER_TABLE):
        merge, repeat, ec = TIER_TABLE[tier]
        cap = tier_capacity(tier)
        em, er, ee, exp_cap = expected[tier]
        rep.check((merge, repeat, ec, cap) == (em, er, ee, exp_cap),
                  f"capacity L{tier}",
                  f"merge={merge} repeat={repeat} ec={ec} -> {cap} B")
        enc = SpiralSesameEncoder(tier=tier)
        rep.check(enc.capacity_bytes() == cap,
                  f"encoder agrees with tier_capacity L{tier}")
        g = tier_geometry(tier)
        rep.check(g.n_logical * repeat == g.n_physical,
                  f"physical cells L{tier}", f"{g.n_physical}")

    # Cell count has to fall monotonically -- that is the whole point of the
    # ladder. Capacity does not: L3 gives up spatial repeat entirely, and the
    # cells that frees up outweigh the merge, so L3 (33 B) out-holds L2 (29 B).
    cells = [tier_geometry(t).n_physical for t in sorted(TIER_TABLE)]
    rep.check(all(a > b for a, b in zip(cells, cells[1:])),
              "cell count falls as the tier coarsens", str(cells))
    caps = [tier_capacity(t) for t in sorted(TIER_TABLE)]
    rep.check(all(c >= 20 for c in caps),
              "every tier still holds a short URL", str(caps))
    try:
        SpiralSesameEncoder(tier=9)
        rep.check(False, "unknown tier rejected")
    except ValueError:
        rep.check(True, "unknown tier rejected")


def group_version_byte(rep):
    for tier in sorted(TIER_TABLE):
        for version in (1, 2, 3):
            vb = pack_version_byte(tier, version)
            rep.check(unpack_version_byte(vb) == (tier, version),
                      f"pack/unpack L{tier} v{version}", f"byte={vb}")
    # L0 has to pack to the bare version number: that is what makes an L0 code
    # indistinguishable from a pre-tier code at the byte level.
    for version in (1, 2, 3):
        rep.check(pack_version_byte(0, version) == version,
                  f"L0 version byte unchanged (v{version})")
    rep.check(max(pack_version_byte(t, v) for t in TIER_TABLE
                  for v in (1, 2, 3)) < 256, "version byte fits")


def group_l0_identical(rep):
    img = SpiralSesameEncoder().encode_text(L0_REF_PAYLOAD)
    digest = hashlib.sha256(np.array(img).tobytes()).hexdigest()
    rep.check(digest == L0_REF_SHA256,
              "L0 render is bit-identical to the pre-tier codec",
              digest[:16])


def group_roundtrip(rep):
    dec = SpiralSesameDecoder()
    for tier in sorted(TIER_TABLE):
        text = PAYLOADS[tier]
        enc = SpiralSesameEncoder(tier=tier)
        t0 = time.time()
        img = enc.encode_text(text)
        got = dec.decode_image(img)
        dt = time.time() - t0
        rep.check(got == text and dec.tier == tier,
                  f"round trip L{tier}",
                  f"{len(text.encode())} B, detected L{dec.tier}, {dt:.2f}s")

    # One byte over capacity must be refused, not silently truncated.
    for tier in sorted(TIER_TABLE):
        enc = SpiralSesameEncoder(tier=tier)
        over = b"x" * (enc.capacity_bytes() + 1)
        try:
            enc.encode_bytes(over)
            rep.check(False, f"oversize rejected L{tier}")
        except ValueError:
            rep.check(True, f"oversize rejected L{tier}")
        exact = b"x" * enc.capacity_bytes()
        rep.check(dec.decode_bytes(enc.encode_bytes(exact)) == exact,
                  f"exact-capacity round trip L{tier}", f"{len(exact)} B")


def group_legacy(rep):
    dec = SpiralSesameDecoder()
    for name in LEGACY_PNGS:
        t0 = time.time()
        got = dec.decode_image(Image.open(name))
        rep.check(bool(got) and dec.tier == 0,
                  f"pre-tier PNG still decodes: {name}",
                  f"L{dec.tier} {time.time() - t0:.2f}s -> {str(got)[:48]}")


def group_rotation(rep):
    dec = SpiralSesameDecoder()
    for tier in sorted(TIER_TABLE):
        img = SpiralSesameEncoder(tier=tier).encode_text(PAYLOADS[tier])
        for roll in (15.0, 30.0):
            t0 = time.time()
            got = dec.decode_image(paste_disc(img, roll))
            rep.check(got == PAYLOADS[tier],
                      f"roll {roll:.0f} deg L{tier}",
                      f"{time.time() - t0:.2f}s")


def group_occlusion(rep):
    """Occlusion tolerance falls as the tier coarsens, and that is the deal the
    ladder makes: L0 spends 5 copies and 30 RS symbols on redundancy, L3 spends
    its cells on seed area instead. Measured limits are 60 deg of ring for
    L0/L1, 30 deg for L2, 10 deg for L3; these sit one notch inside that."""
    dec = SpiralSesameDecoder()
    budget = {0: 40.0, 1: 40.0, 2: 20.0, 3: 10.0}
    for tier in sorted(TIER_TABLE):
        img = SpiralSesameEncoder(tier=tier).encode_text(PAYLOADS[tier])
        wedge = budget[tier]
        t0 = time.time()
        got = dec.decode_image(occlude(img, wedge))
        rep.check(got == PAYLOADS[tier],
                  f"{wedge:.0f} deg wedge occluded L{tier}",
                  f"{time.time() - t0:.2f}s")


# Rungs at and above each tier's measured floor: 180 px for L0, 220 for L1,
# 120 for L2 and L3.
DIST_SIZES = (700, 400, 260, 220, 180, 150, 120)
DIST_FLOOR = {0: 180, 1: 220, 2: 120, 3: 120}


def group_distance(rep):
    """The payoff side of the ladder: a coarse tier reads at a size where L0
    cannot, which is the case the tiers exist for -- a code on a wall or a
    screen, where L0's 7000 seeds fall under the camera's resolution limit.

    L1 floors above L0 (220 against 180): the merge halves the sample count
    the front end's pose search has to work with, and it loses its bearings
    before the seeds themselves become unreadable. L2 and L3 beat both by a
    wide margin. Every rung at or above a tier's floor is checked, not just
    the floor, because the failure that motivated this group was a single
    missing rung -- L3 decoded at 900, 800, 640 and all the way down to 120,
    and failed at 700 alone, where pose refinement walked out of a flat
    metric's basin."""
    dec = SpiralSesameDecoder()
    for tier in sorted(TIER_TABLE):
        text = PAYLOADS[tier]
        base = SpiralSesameEncoder(tier=tier).encode_text(text)
        rungs = [n for n in DIST_SIZES if n >= DIST_FLOOR[tier]]
        miss = [n for n in rungs
                if dec.decode_image(base.resize((n, n), Image.LANCZOS)) != text]
        rep.check(not miss, f"L{tier} decodes down to {DIST_FLOOR[tier]} px",
                  f"failed at {miss} px" if miss else f"{len(rungs)} rungs")

    # L0 gives up 30 px above where the coarse tiers are still reading.
    tiny = 120
    for tier in (0, 2, 3):
        text = PAYLOADS[tier]
        img = SpiralSesameEncoder(tier=tier).encode_text(text)
        got = dec.decode_image(img.resize((tiny, tiny), Image.LANCZOS))
        want = got == text if tier else got is None
        rep.check(want,
                  f"{tiny} px: L{tier} " + ("reads" if tier else "does not"),
                  f"L{dec.tier}" if got else "no decode")


GROUPS = (
    ("capacity", group_capacity),
    ("version", group_version_byte),
    ("identical", group_l0_identical),
    ("roundtrip", group_roundtrip),
    ("legacy", group_legacy),
    ("rotation", group_rotation),
    ("occlusion", group_occlusion),
    ("distance", group_distance),
)


def main(argv):
    rep = Report()
    for name, fn in GROUPS:
        if argv and not any(a in name for a in argv):
            continue
        print(f"\n--- {name} ---")
        fn(rep)
    print("\n" + "=" * 40)
    if rep.failures:
        print(f"{len(rep.failures)} FAILURES:")
        for f in rep.failures:
            print(f"  {f}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
