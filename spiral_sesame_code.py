"""
螺旋芝麻码 (Spiral Sesame Matrix) v6
=====================================
- 仅使用白芝麻的 有/无 编码: 有芝麻=1, 无芝麻=0(留空)
- 芝麻的"尖"始终朝向贝果图案正中心
- 粉色贝果, 芝麻均匀分布在整个环面上
- 比特白化: 无论载荷多少, 芝麻密度均约50%
- QR码式版本分级: 固定位置元数据标明数据范围
- 五重重复编码 + RS 纠错, 扇区感知交织

容量: 144 bytes (1024px), 支持 B站/小红书完整链接。
"""

import math
import struct
import hashlib
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from reedsolo import RSCodec


# ─── LCG PRNG (matches JS implementation) ─────────────────────────────

class LCG:
    """Linear Congruential Generator matching the JS seededRandom()."""
    def __init__(self, seed):
        self.s = seed & 0x7fffffff

    def next(self):
        self.s = (self.s * 1103515245 + 12345) & 0x7fffffff
        return self.s / 0x7fffffff

    def uniform(self, a, b):
        return a + (b - a) * self.next()


def lcg_shuffle(lst, rng):
    """Fisher-Yates shuffle using LCG PRNG."""
    for i in range(len(lst) - 1, 0, -1):
        j = int(rng.next() * (i + 1))
        lst[i], lst[j] = lst[j], lst[i]


# ─── Reed-Solomon Wrapper ────────────────────────────────────────────────

class ReedSolomon:
    def __init__(self, nsym=30):
        self.nsym = nsym
        self.codec = RSCodec(nsym)

    def encode(self, data: bytes) -> bytes:
        return bytes(self.codec.encode(bytearray(data)))

    def decode(self, received: bytes) -> Optional[bytes]:
        try:
            result = self.codec.decode(bytearray(received))
            return bytes(result[0])
        except Exception:
            return None

    def decode_with_erasures(self, received: bytes, erasure_positions=None) -> Optional[bytes]:
        if not erasure_positions:
            return self.decode(received)
        try:
            result = self.codec.decode(bytearray(received), erase_pos=erasure_positions)
            return bytes(result[0])
        except Exception:
            return self.decode(received)


# ─── Spatial Interleaver ─────────────────────────────────────────────────

class SectorInterleaver:
    """Sector-aware interleaver that places each bit's REPEAT copies
    in maximally separated angular sectors for burst-error resilience."""

    def __init__(self, grid, n_physical: int, repeat: int, seed=42):
        self.repeat = repeat
        self.n_physical = n_physical
        cx = sum(c[0] for c in grid) / len(grid)
        cy = sum(c[1] for c in grid) / len(grid)

        angles = []
        for x, y, ri, ai in grid[:n_physical]:
            angles.append(math.atan2(y - cy, x - cx))

        sector_indices = []
        for a in angles:
            # +1e-6 keeps cells exactly on a sector boundary assigned
            # identically despite 1-ulp trig differences across languages
            s = int((a + math.pi) / (2 * math.pi) * repeat + 1e-6) % repeat
            sector_indices.append(s)

        sectors = [[] for _ in range(repeat)]
        for i, s in enumerate(sector_indices):
            sectors[s].append(i)

        rng = LCG(seed)
        for s in sectors:
            lcg_shuffle(s, rng)

        logical_count = n_physical // repeat
        self.logical_to_phys = [[] for _ in range(logical_count)]

        used = set()
        for bit_idx in range(logical_count):
            assigned = []
            for rep in range(repeat):
                sector = sectors[rep]
                found = False
                start = bit_idx % len(sector)
                for offset in range(len(sector)):
                    candidate = sector[(start + offset) % len(sector)]
                    if candidate not in used:
                        assigned.append(candidate)
                        used.add(candidate)
                        found = True
                        break
                if not found:
                    for p in range(n_physical):
                        if p not in used:
                            assigned.append(p)
                            used.add(p)
                            found = True
                            break
            self.logical_to_phys[bit_idx] = assigned

        self.logical_to_physical = [0] * n_physical
        self.physical_to_logical = [0] * n_physical
        logical_pos = 0
        for bit_idx in range(logical_count):
            for rep in range(repeat):
                phys_pos = self.logical_to_phys[bit_idx][rep]
                self.logical_to_physical[logical_pos] = phys_pos
                self.physical_to_logical[phys_pos] = logical_pos
                logical_pos += 1

    def interleave(self, bits):
        out = [0] * self.n_physical
        for i, b in enumerate(bits):
            if i < self.n_physical:
                out[self.logical_to_physical[i]] = b
        return out

    def deinterleave_with_erasures(self, values):
        out = [None] * self.n_physical
        for i in range(min(self.n_physical, len(values))):
            out[self.physical_to_logical[i]] = values[i]
        return out


# ─── Constants ────────────────────────────────────────────────────────────

DEFAULT_IMAGE_SIZE = 1024
DEFAULT_EC = 30
DEFAULT_SEED = 42
VERSION_HEADER = b'SSM6'
REPEAT = 5
WHITEN_SEED = 7

RING_OUTER_RATIO = 0.46
RING_INNER_RATIO = 0.16
SESAME_SPACING = 9.0

# Version table: version_id -> max payload bytes
VERSION_TABLE = {
    1: 30,    # Short URLs
    2: 70,    # Medium URLs
    3: 144,   # Full URLs (max for 1024px)
}

# Metadata: 1 byte version + 2 bytes payload_length = 3 bytes = 24 bits
# (inside the whitened stream, protected by REPEAT majority vote)
METADATA_BYTES = 3


def get_ring_params(image_size):
    half = min(image_size) // 2
    outer_r = int(half * RING_OUTER_RATIO * 2)
    inner_r = int(half * RING_INNER_RATIO * 2)
    return outer_r, inner_r


def build_polar_grid(cx, cy, inner_r, outer_r, spacing, fixed_count=None):
    cells = []
    r_step = spacing
    n_rings = max(1, int((outer_r - inner_r) / r_step))
    actual_r_step = (outer_r - inner_r) / n_rings

    for ri in range(n_rings):
        r = inner_r + actual_r_step * (ri + 0.5)
        circumference = 2 * math.pi * r
        n_angles = max(1, int(circumference / spacing))
        angle_step = 2 * math.pi / n_angles
        offset = angle_step * 0.5 if ri % 2 else 0
        for ai in range(n_angles):
            angle = ai * angle_step + offset
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            cells.append((x, y, ri, ai))

    if fixed_count is not None:
        if len(cells) >= fixed_count:
            return cells[:fixed_count]
        rng = LCG(12345)
        while len(cells) < fixed_count:
            angle = rng.uniform(0, 2 * math.pi)
            r = inner_r + rng.uniform(0.1, 0.9) * (outer_r - inner_r)
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            cells.append((x, y, 0, 0))
        return cells[:fixed_count]

    return cells


_canon_outer, _canon_inner = get_ring_params((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))
CANONICAL_GRID_COUNT = len(build_polar_grid(
    DEFAULT_IMAGE_SIZE // 2, DEFAULT_IMAGE_SIZE // 2,
    _canon_inner + 4, _canon_outer - 4, SESAME_SPACING))

N_TOTAL_LOGICAL = CANONICAL_GRID_COUNT // REPEAT   # 1457
N_PHYSICAL = N_TOTAL_LOGICAL * REPEAT              # 7285


def whitening_sequence(n, seed=WHITEN_SEED):
    """Deterministic pseudo-random bit sequence for density whitening."""
    rng = LCG(seed)
    return [1 if rng.next() < 0.5 else 0 for _ in range(n)]


def select_version(payload_len: int) -> int:
    for v in sorted(VERSION_TABLE.keys()):
        if payload_len <= VERSION_TABLE[v]:
            return v
    raise ValueError(f"Data too long: {payload_len} bytes")


# ─── Renderer ────────────────────────────────────────────────────────────

class SesameRenderer:
    # White sesame teardrop: tip points toward the bagel center
    TIP_LEN = 6.0
    BODY_A = 4.5
    BODY_B = 2.8

    def create_bagel(self, size, seed=42):
        w, h = size
        cx, cy = w // 2, h // 2
        outer_r, inner_r = get_ring_params(size)
        rng = np.random.RandomState(seed)

        result = np.full((h, w, 3), 255, dtype=np.float32)

        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(np.float32)
        ring = (dist < outer_r) & (dist > inner_r)

        t = np.zeros((h, w), dtype=np.float32)
        valid = ring & (outer_r > inner_r)
        t[valid] = (dist[valid] - inner_r) / (outer_r - inner_r)
        shade = np.clip(0.55 + 0.45 * np.sin(t * math.pi), 0, 1)

        # Deep-pink bagel palette: surface mean brightness stays <= ~183,
        # well below white sesame (~215-235) for reliable presence detection
        bagel_r = 205.0 * shade + 20
        bagel_g = 118.0 * shade + 28
        bagel_b = 138.0 * shade + 32

        for c, ch in enumerate([bagel_r, bagel_g, bagel_b]):
            result[:, :, c] = np.where(ring, ch, result[:, :, c])

        hole = dist <= inner_r
        hole_color = np.array([245, 230, 235], dtype=np.float32)
        for c in range(3):
            result[:, :, c] = np.where(hole, hole_color[c], result[:, :, c])

        tex = rng.normal(0, 6, result.shape)
        result = np.where(ring[..., None], result + tex, result)
        result = np.clip(result, 0, 255).astype(np.uint8)

        img = Image.fromarray(result).filter(ImageFilter.GaussianBlur(1.5))

        draw = ImageDraw.Draw(img)
        for off in range(2):
            draw.ellipse([cx - outer_r + off, cy - outer_r + off,
                          cx + outer_r - off, cy + outer_r - off],
                         outline=(170, 90, 110))
            draw.ellipse([cx - inner_r - off, cy - inner_r - off,
                          cx + inner_r + off, cy + inner_r + off],
                         outline=(170, 90, 110))
        return img

    @staticmethod
    def _teardrop_points(x, y, ux, uy, tip_len, a, b,
                         start_deg=40, end_deg=320, step_deg=20):
        """Teardrop outline: sharp tip along (ux,uy), rounded body behind."""
        vx, vy = -uy, ux
        pts = [(x + ux * tip_len, y + uy * tip_len)]
        deg = start_deg
        while deg <= end_deg:
            ar = math.radians(deg)
            ca, sa = math.cos(ar), math.sin(ar)
            pts.append((x + ux * (a * ca) + vx * (b * sa),
                        y + uy * (a * ca) + vy * (b * sa)))
            deg += step_deg
        return pts

    def draw_sesame(self, draw, x, y, angle_to_center):
        """Draw one white sesame seed at (x,y), tip aimed at the center."""
        ux, uy = math.cos(angle_to_center), math.sin(angle_to_center)
        body = self._teardrop_points(x, y, ux, uy,
                                     self.TIP_LEN, self.BODY_A, self.BODY_B)
        draw.polygon(body, fill=(240, 228, 190), outline=(205, 190, 145))
        hx, hy = x - ux * 1.0, y - uy * 1.0
        hl = self._teardrop_points(hx, hy, ux, uy,
                                   self.TIP_LEN * 0.45,
                                   self.BODY_A * 0.45, self.BODY_B * 0.42)
        draw.polygon(hl, fill=(252, 246, 222))


# ─── Encoder ─────────────────────────────────────────────────────────────

class SpiralSesameEncoder:
    def __init__(self, image_size=(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
                 ec_level=DEFAULT_EC, seed=DEFAULT_SEED):
        self.image_size = image_size
        self.ec_level = ec_level
        self.seed = seed
        self.rs = ReedSolomon(nsym=ec_level)
        self.renderer = SesameRenderer()

        cx, cy = image_size[0] // 2, image_size[1] // 2
        outer_r, inner_r = get_ring_params(image_size)
        self.cx, self.cy = cx, cy
        self.outer_r = outer_r
        self.inner_r = inner_r

        raw_grid = build_polar_grid(cx, cy, inner_r + 4, outer_r - 4, SESAME_SPACING)
        self.grid = raw_grid
        self.grid_count = len(raw_grid)

        self.n_total_logical = self.grid_count // REPEAT
        self.n_physical = self.n_total_logical * REPEAT
        self.whitening = whitening_sequence(self.n_total_logical)

    def capacity_bytes(self):
        total_bytes = self.n_total_logical // 8
        overhead = METADATA_BYTES + len(VERSION_HEADER) + 2 + 4 + self.ec_level
        return max(0, total_bytes - overhead)

    def encode_text(self, text: str) -> Image.Image:
        return self.encode_bytes(text.encode('utf-8'))

    def encode_bytes(self, data: bytes) -> Image.Image:
        cap = self.capacity_bytes()
        if len(data) > cap:
            raise ValueError(f"Data too long: {len(data)} bytes, max {cap}")

        version = select_version(len(data))

        # Metadata: version(1B) + payload_length(2B)
        metadata = struct.pack('>BH', version, len(data))
        meta_bits = []
        for byte in metadata:
            for i in range(7, -1, -1):
                meta_bits.append((byte >> i) & 1)

        # Payload: header + length + data + hash, RS-encoded
        payload = VERSION_HEADER + struct.pack('>H', len(data)) + data
        digest = hashlib.sha256(payload).digest()[:4]
        encoded = self.rs.encode(payload + digest)

        data_bits = []
        for byte in encoded:
            for i in range(7, -1, -1):
                data_bits.append((byte >> i) & 1)

        # Logical stream: metadata + data + zero padding
        logical = meta_bits + data_bits
        while len(logical) < self.n_total_logical:
            logical.append(0)
        logical = logical[:self.n_total_logical]

        # Whiten so sesame density stays ~50% regardless of payload
        whitened = [b ^ w for b, w in zip(logical, self.whitening)]

        # Repeat each bit REPEAT times
        physical_bits = []
        for b in whitened:
            physical_bits.extend([b] * REPEAT)

        # Sector-aware interleaving across the full grid
        interleaver = SectorInterleaver(self.grid, self.n_physical,
                                        REPEAT, seed=self.seed)
        scattered = interleaver.interleave(physical_bits)

        # Render: sesame ONLY where bit==1, tip aimed at the center
        canvas = self.renderer.create_bagel(self.image_size, seed=self.seed)
        draw = ImageDraw.Draw(canvas)

        for i in range(min(self.n_physical, len(self.grid))):
            if scattered[i] != 1:
                continue
            x, y, ri, ai = self.grid[i]
            angle = math.atan2(self.cy - y, self.cx - x)
            self.renderer.draw_sesame(draw, x, y, angle)

        canvas = canvas.filter(ImageFilter.GaussianBlur(0.3))
        return canvas


# ─── Decoder ─────────────────────────────────────────────────────────────

class SpiralSesameDecoder:
    # Presence thresholds: bagel surface patch-mean <= ~183, sesame ~215-235
    SESAME_PRESENT_MIN = 205
    SESAME_ABSENT_MAX = 192

    def __init__(self, ec_level=DEFAULT_EC, seed=DEFAULT_SEED):
        self.ec_level = ec_level
        self.seed = seed
        self.rs = ReedSolomon(nsym=ec_level)
        self.whitening = whitening_sequence(N_TOTAL_LOGICAL)
        self._grid_cache = None
        self._centers = None
        self._interleaver = None
        self._repeat_groups = None

    def decode_image(self, img: Image.Image, try_hard=True) -> Optional[str]:
        result = self.decode_bytes(img, try_hard=try_hard)
        if result is None:
            return None
        try:
            return result.decode('utf-8')
        except UnicodeDecodeError:
            return result.hex()

    def decode_bytes(self, img: Image.Image, try_hard=True) -> Optional[bytes]:
        img = img.convert('RGB')
        arr = np.array(img)

        deferred = []
        for raw in self._pose_candidates(arr):
            result = self._backend(raw, False)
            if result is not None:
                return result
            # The exhaustive backend sweeps every plausible payload length,
            # so it costs seconds per candidate. Only the best-ranked poses
            # are worth that; the rest are geometry the cheap pass already
            # rejected by a wide margin.
            if try_hard and len(deferred) < 4:
                deferred.append(raw)
        for raw in deferred:
            result = self._backend(raw, True)
            if result is not None:
                return result

        circle = self._detect_bagel_circle(arr)
        if circle is None:
            print("[decode] Failed to detect bagel circle")
            return None

        cx, cy, detected_outer_r, detected_inner_r = circle

        # Estimate scale from the seed-ring radii: robust even when the
        # outer rim is damaged/cropped and the edge-based radius is wrong.
        ring_scale = self._estimate_scale_from_rings(
            arr, cx, cy, detected_inner_r)
        if ring_scale is not None:
            scale = ring_scale
        else:
            scale = (get_ring_params((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))[0]
                     / max(detected_outer_r, 1))

        canon_cx = DEFAULT_IMAGE_SIZE // 2
        canon_cy = DEFAULT_IMAGE_SIZE // 2
        canon_outer_r, canon_inner_r = get_ring_params(
            (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))

        normalized = self._normalize_image(img, cx, cy, scale, DEFAULT_IMAGE_SIZE)
        narr = np.array(normalized)

        damage = self._border_white_mask(narr)

        grid = self._grid()

        raw_values = self._read_presence(narr, grid, damage)
        return self._backend(raw_values, try_hard)

    def _backend(self, raw_values, try_hard=True):
        """Deinterleave -> majority vote -> de-whiten -> RS. Shared by both
        front ends so they can be tried in sequence."""
        inter = self._interleave_obj()
        src = np.empty(N_PHYSICAL, dtype=np.int8)
        src[:] = -1
        n = min(N_PHYSICAL, len(raw_values))
        for i in range(n):
            v = raw_values[i]
            if v is not None:
                src[i] = v
        ordered = np.full(N_PHYSICAL, -1, dtype=np.int8)
        ordered[inter.physical_to_logical[:n]] = src[:n]

        groups = ordered.reshape(N_TOTAL_LOGICAL, REPEAT).astype(np.int16)
        ones = (groups == 1).sum(axis=1)
        zeros = (groups == 0).sum(axis=1)
        nones = (groups < 0).sum(axis=1)
        whit = np.fromiter(self.whitening, dtype=np.int16, count=N_TOTAL_LOGICAL)
        b = np.where(ones > zeros, 1, np.where(zeros > ones, 0, whit))
        logical = (b ^ whit).tolist()
        logical_erased = (nones >= 3).tolist()

        result = self._extract_payload(logical, logical_erased)
        if result is not None:
            return result
        if try_hard:
            return self._decode_exhaustive(logical, logical_erased)
        return None

    # ─── Camera-robust front end ────────────────────────────────────────
    #
    # A phone photo differs from a rendered PNG in four ways: the bagel is
    # off-center and small, the view is tilted (circle -> ellipse plus
    # keystone), the in-plane rotation is unknown (the grid has no angular
    # anchor), and illumination is arbitrary. So instead of fixed brightness
    # thresholds this path works on a morphological white top-hat response,
    # which is invariant to smooth lighting and rejects background texture,
    # and fits the full pose: center, scale, aspect, tilt axis, rotation,
    # keystone. Geometry comes from the rim conic of the seed disc; rotation
    # comes from REPEAT-copy agreement (_align_score), because no scoring
    # function based on the samples alone can tell a rotation from an alias
    # of it.

    ANALYZE_MAX_DIM = 560
    R0 = 467.0

    def _grid(self):
        if self._grid_cache is None:
            canon_outer_r, canon_inner_r = get_ring_params(
                (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))
            self._grid_cache = build_polar_grid(
                DEFAULT_IMAGE_SIZE // 2, DEFAULT_IMAGE_SIZE // 2,
                canon_inner_r + 4, canon_outer_r - 4, SESAME_SPACING,
                fixed_count=CANONICAL_GRID_COUNT)
        return self._grid_cache

    def _interleave_obj(self):
        if self._interleaver is None:
            self._interleaver = SectorInterleaver(
                self._grid(), N_PHYSICAL, REPEAT, seed=self.seed)
        return self._interleaver

    def _cell_centers(self):
        if self._centers is None:
            grid = self._grid()
            c = DEFAULT_IMAGE_SIZE / 2.0
            self._centers = np.array(
                [[grid[i][0] - c, grid[i][1] - c] for i in range(N_PHYSICAL)],
                dtype=np.float64)
        return self._centers

    @staticmethod
    def _box_blur(a, rad):
        if rad <= 0:
            return np.asarray(a, dtype=np.float64)
        a = np.asarray(a, dtype=np.float64)
        h, w = a.shape
        c = np.zeros((h + 1, w + 1), dtype=np.float64)
        c[1:, 1:] = np.cumsum(np.cumsum(a, axis=0), axis=1)
        ys = np.arange(h)
        xs = np.arange(w)
        y0 = np.clip(ys - rad, 0, h)
        y1 = np.clip(ys + rad + 1, 0, h)
        x0 = np.clip(xs - rad, 0, w)
        x1 = np.clip(xs + rad + 1, 0, w)
        s = (c[np.ix_(y1, x1)] - c[np.ix_(y0, x1)]
             - c[np.ix_(y1, x0)] + c[np.ix_(y0, x0)])
        n = (y1 - y0)[:, None] * (x1 - x0)[None, :]
        return s / np.maximum(n, 1)

    @staticmethod
    def _mov_extreme(a, k, want_min):
        if k <= 1:
            return np.asarray(a, dtype=np.float64)
        k |= 1
        pad = k // 2
        b = np.pad(a, ((pad, pad), (pad, pad)), mode='edge')
        fn = np.minimum if want_min else np.maximum
        h, w = a.shape
        m = b[:h, :].copy()
        for i in range(1, k):
            m = fn(m, b[i:i + h, :])
        out = m[:, :w].copy()
        for j in range(1, k):
            out = fn(out, m[:, j:j + w])
        return out

    def _tophat(self, a, k):
        """White top-hat (image minus its morphological opening): keeps bright
        blobs smaller than k and cancels broad shading and background texture,
        which a plain high-pass would leave competing with the seeds."""
        a = np.asarray(a, dtype=np.float64)
        if k <= 1:
            return a - self._box_blur(a, 8)
        opened = self._mov_extreme(self._mov_extreme(a, k, True), k, False)
        return a - opened

    def _project(self, prm, pts):
        """Canonical cell coords -> image coords under the fitted pose."""
        cp = math.cos(prm['phi'])
        sp = math.sin(prm['phi'])
        px = pts[:, 0]
        py = pts[:, 1]
        qx = px * cp - py * sp
        qy = px * sp + py * cp
        d = 1.0 + (prm['kx'] * qx + prm['ky'] * qy) / self.R0
        d = np.clip(d, 0.35, 2.5)
        qx = qx / d
        qy = qy / d
        ct = math.cos(prm['theta'])
        st = math.sin(prm['theta'])
        u = qx * ct - qy * st
        v = (qx * st + qy * ct) * prm['aspect']
        x = u * ct + v * st
        y = -u * st + v * ct
        return prm['cx'] + x * prm['scale'], prm['cy'] + y * prm['scale']

    def _sample_cells(self, img, prm, step=1):
        h, w = img.shape
        pts = self._cell_centers()[::step]
        xs, ys = self._project(prm, pts)
        x0 = np.floor(xs)
        y0 = np.floor(ys)
        fx = xs - x0
        fy = ys - y0
        xi = x0.astype(np.int64)
        yi = y0.astype(np.int64)
        valid = (xi >= 0) & (xi < w - 1) & (yi >= 0) & (yi < h - 1)
        xi = np.clip(xi, 0, max(w - 2, 0))
        yi = np.clip(yi, 0, max(h - 2, 0))
        a = img[yi, xi]
        b = img[yi, xi + 1]
        c = img[yi + 1, xi]
        d = img[yi + 1, xi + 1]
        val = (a * (1 - fx) * (1 - fy) + b * fx * (1 - fy)
               + c * (1 - fx) * fy + d * fx * fy)
        return np.where(valid, val, np.nan), xs, ys, valid

    @staticmethod
    def _otsu(v, bins=64):
        """Threshold maximizing between-class variance, plus the normalized
        separability (0..0.25) used as the pose-alignment score."""
        v = v[np.isfinite(v)]
        if v.size < 64:
            return None, -1.0
        lo = float(v.min())
        hi = float(v.max())
        if hi - lo < 1e-9:
            return lo, -1.0
        hist, edges = np.histogram(v, bins=bins, range=(lo, hi))
        centers = (edges[:-1] + edges[1:]) / 2.0
        total = float(hist.sum())
        wt = hist / total
        mu = float((wt * centers).sum())
        best_eta = -1.0
        best_t = centers[0]
        w0 = 0.0
        mu0 = 0.0
        for i in range(bins):
            w0 += wt[i]
            mu0 += wt[i] * centers[i]
            if w0 <= 1e-6 or w0 >= 1 - 1e-6:
                continue
            m0 = mu0 / w0
            m1 = (mu - mu0) / (1 - w0)
            eta = w0 * (1 - w0) * (m0 - m1) ** 2
            if eta > best_eta:
                best_eta = eta
                best_t = centers[i]
        var = float(v.var())
        return float(best_t), (best_eta / var if var > 1e-12 else -1.0)

    def _copies(self):
        """Physical indices of the REPEAT copies of every logical bit."""
        if self._repeat_groups is None:
            self._repeat_groups = np.array(
                self._interleave_obj().logical_to_physical,
                dtype=np.int64).reshape(N_TOTAL_LOGICAL, REPEAT)
        return self._repeat_groups

    def _align_score(self, img, prm):
        """Fraction of logical bits whose REPEAT copies all read the same.

        Sample bimodality cannot see the absolute rotation: turning the grid
        slides each ring tangentially, so every cell still lands on either a
        seed or bare dough and the samples stay just as bimodal -- an aliased
        rotation can even out-score the true one. The copies of a logical bit
        sit in different sectors, so a rotation shifts each of them by a
        different number of cells and their agreement collapses. That makes
        this the score that actually pins the rotation, and it is sharper than
        bimodality on the rest of the pose too.
        """
        vals, _, _, valid = self._sample_cells(img, prm)
        thr, _ = self._otsu(vals)
        if thr is None:
            return -1.0
        bits = np.where(valid, (vals > thr).astype(np.float64), np.nan)
        g = bits[self._copies()]
        g = g[np.all(np.isfinite(g), axis=1)]
        if g.shape[0] < 64:
            return -1.0
        return float(np.mean(g.max(axis=1) == g.min(axis=1)))

    @staticmethod
    def _row_detrend(a, rad):
        n = a.shape[1]
        c = np.zeros((a.shape[0], n + 1), dtype=np.float64)
        c[:, 1:] = np.cumsum(a, axis=1)
        i = np.arange(n)
        i0 = np.clip(i - rad, 0, n)
        i1 = np.clip(i + rad + 1, 0, n)
        s = c[:, i1] - c[:, i0]
        return a - s / np.maximum(i1 - i0, 1)

    def _localize(self, resp):
        """Find the seed disc in the top-hat energy map: its centroid is the
        code center, its RMS radius the scale, and its inertia the tilt axis
        and aspect. Brightness, shading and background texture are already
        gone from `resp`, so the disc is the only large structure left."""
        h, w = resp.shape
        rad = max(3, int(round(min(h, w) / 90.0)))
        e = self._box_blur(resp * resp, rad)
        inside = e > 0.25 * float(e.max())
        outside = ~inside
        if int(inside.sum()) < 300 or not bool(outside.any()):
            return None, -1.0
        contrast = float(e[inside].mean()) / (float(e[outside].mean()) + 1e-9)
        if contrast < 2.5:
            return None, -1.0
        ys, xs = np.nonzero(inside)
        wts = e[inside]
        cx = float(xs.mean())
        cy = float(ys.mean())
        rms = math.sqrt(float(((xs - cx) ** 2 + (ys - cy) ** 2).mean()))
        keep = np.ones_like(xs, dtype=bool)
        for _ in range(3):
            # Re-center on the disc alone: background speckle that survived
            # the threshold would otherwise drag the centroid several pixels,
            # which is enough to flatten the scale landscape later on.
            keep = ((xs - cx) ** 2 + (ys - cy) ** 2) <= (1.5 * rms) ** 2
            if int(keep.sum()) < 300:
                return None, -1.0
            xk, yk, wk = xs[keep], ys[keep], wts[keep]
            sv = float(wk.sum())
            cx = float((xk * wk).sum() / sv)
            cy = float((yk * wk).sum() / sv)
            rms = math.sqrt(float(((xk - cx) ** 2 + (yk - cy) ** 2).mean()))
        xk, yk, wk = xs[keep], ys[keep], wts[keep]
        sv = float(wk.sum())
        dx = xk - cx
        dy = yk - cy
        mu20 = float((wk * dx * dx).sum() / sv)
        mu02 = float((wk * dy * dy).sum() / sv)
        mu11 = float((wk * dx * dy).sum() / sv)
        tr = mu20 + mu02
        det = mu20 * mu02 - mu11 * mu11
        disc = max(tr * tr / 4.0 - det, 0.0)
        l1 = tr / 2.0 + math.sqrt(disc)
        l2 = max(tr / 2.0 - math.sqrt(disc), 1e-9)
        outer_r, inner_r = get_ring_params(
            (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))
        # Area-uniform annulus between inner_r and outer_r: E[r^2]=(R^2+r0^2)/2
        rms_norm = math.sqrt((outer_r ** 2 + inner_r ** 2) / 2.0)
        scale = math.sqrt(max(tr, 1e-9)) / rms_norm
        if scale <= 1e-6:
            return None, -1.0
        aspect = math.sqrt(l2 / l1)
        theta = 0.5 * math.atan2(2 * mu11, mu20 - mu02)
        r_guess = scale * float(outer_r)
        # The edge conic wants an unsmoothed energy map: the box blur that
        # steadies the centroid pushes the steepest-descent edge outward by
        # roughly two thirds of its radius, which is a 2-4% scale error on a
        # small bagel and far outside the alignment capture basin.
        e_edge = self._box_blur(resp * resp, max(1, int(round(min(h, w) / 280.0))))
        outer_c = self._edge_conic(e_edge, cx, cy, 0.60 * r_guess,
                                   1.40 * r_guess, True)
        hole_c = self._edge_conic(e_edge, cx, cy, 0.15 * r_guess,
                                  0.55 * r_guess, False)
        if outer_c is not None:
            rcx, rcy, rmaj, rasp, rpsi = outer_c
            r_scale = rmaj / 468.0
            if 0.6 * scale < r_scale < 1.4 * scale:
                cx, cy, scale, aspect, theta = rcx, rcy, r_scale, rasp, rpsi
            kx = ky = 0.0
            if aspect < 0.985:
                # Only worth the full pose fit under real tilt: the projected
                # grid center stops coinciding with the rim ellipse's center,
                # and the keystone terms are the only thing that can absorb
                # the gap. Frontal shots are better served by the rim conic
                # alone -- the hole conic is small and its edge sits several
                # pixels outside the true hole, so pulling it into the fit
                # inflates the scale past the alignment basin.
                pose = self._fit_pose_conics(
                    outer_c, hole_c,
                    {'cx': cx, 'cy': cy, 'scale': scale, 'aspect': aspect,
                     'theta': theta, 'phi': 0.0, 'kx': 0.0, 'ky': 0.0})
                cx, cy, scale = pose['cx'], pose['cy'], pose['scale']
                aspect, theta = pose['aspect'], pose['theta']
                kx, ky = pose['kx'], pose['ky']
        else:
            kx, ky = 0.0, 0.0
        if aspect > 0.985:
            # Tilt below ~17 degrees is not worth modelling, and any residual
            # squeeze at that level is fit noise: a 1% squeeze along a random
            # axis costs more alignment than it explains.
            aspect, theta = 1.0, 0.0
        return {'cx': cx, 'cy': cy, 'scale': scale,
                'aspect': min(max(aspect, 0.3), 1.0),
                'theta': theta, 'phi': 0.0, 'kx': kx, 'ky': ky}, contrast

    def _edge_conic(self, e, cx, cy, r_lo, r_hi, descent=True):
        """Conic through the steepest radial transition of the seed energy
        between r_lo and r_hi, taken per angle bin: descent finds the outer
        rim, ascent the hole edge."""
        h, w = e.shape
        yy, xx = np.mgrid[0:h, 0:w]
        dx = xx - cx
        dy = yy - cy
        r = np.hypot(dx, dy)
        ang = np.arctan2(dy, dx)
        nb, nsh = 360, 64
        span = r_hi - r_lo
        if span <= 1.0:
            return None
        sh = ((r - r_lo) / span * nsh).astype(np.int64)
        ok = (sh >= 0) & (sh < nsh)
        bi = np.clip(((ang[ok] + math.pi) / (2 * math.pi) * nb).astype(np.int64),
                     0, nb - 1)
        flat = bi * nsh + sh[ok]
        S = np.bincount(flat, weights=e[ok].ravel(),
                        minlength=nb * nsh).reshape(nb, nsh)
        C = np.bincount(flat, minlength=nb * nsh).reshape(nb, nsh)
        P = S / np.maximum(C, 1.0)
        # Normalise each angle bin by its own peak: under tilt the far side is
        # compressed and its gradient is proportionally shallower.
        P = P / np.maximum(P.max(axis=1, keepdims=True), 1e-9)
        kern = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        kern /= kern.sum()
        Ps = np.stack([np.convolve(row, kern, mode='same') for row in P])
        dP = Ps[:, 1:] - Ps[:, :-1]
        j = np.argmin(dP, axis=1) if descent else np.argmax(dP, axis=1)
        depth = np.abs(dP[np.arange(nb), j])
        good = depth > 0.20 * float(np.percentile(depth, 70))
        if int(good.sum()) < 200:
            return None
        if len(np.unique(np.arange(nb)[good] // 30)) < 10:
            return None
        jl = np.clip(j - 1, 0, nsh - 2)
        jh = np.clip(j + 1, 0, nsh - 2)
        d0 = dP[np.arange(nb), jl]
        d1 = dP[np.arange(nb), j]
        d2 = dP[np.arange(nb), jh]
        denom = d0 - 2 * d1 + d2
        sub = np.where(np.abs(denom) > 1e-9,
                       0.5 * (d0 - d2) / np.where(np.abs(denom) > 1e-9,
                                                  denom, 1.0), 0.0)
        sub = np.clip(sub, -1.0, 1.0)
        r_edge = r_lo + span * (j + 0.5 + sub) / nsh
        theta_b = (np.arange(nb) + 0.5) / nb * 2 * math.pi - math.pi
        x = cx + r_edge * np.cos(theta_b)
        y = cy + r_edge * np.sin(theta_b)
        return self._conic_fit(x[good], y[good])

    @staticmethod
    def _conic_fit(x, y):
        """Algebraic ellipse fit with 2.5-sigma clipping, returned as
        (cx, cy, semi-major, aspect, major-axis angle)."""
        cx = cy = 0.0
        for _ in range(3):
            if x.size < 24:
                return None
            A = np.stack([x * x, x * y, y * y, x, y], axis=1)
            sol = np.linalg.lstsq(A, -np.ones_like(x), rcond=None)[0]
            a, b, c, d, f = sol
            m = np.array([[a, b / 2.0], [b / 2.0, c]])
            try:
                cen = np.linalg.solve(2.0 * m, [-d, -f])
            except np.linalg.LinAlgError:
                return None
            val = (cen @ m @ cen) + d * cen[0] + f * cen[1] + 1.0
            if val >= -1e-9:
                return None
            ev, evec = np.linalg.eigh(m / -val)
            if ev[0] <= 1e-12:
                return None
            ax = 1.0 / np.sqrt(ev)
            major = int(np.argmax(ax))
            cx, cy = float(cen[0]), float(cen[1])
            psi = float(math.atan2(evec[1, major], evec[0, major]))
            a_maj = float(ax[major])
            asp = float(ax[1 - major] / ax[major])
            tt = np.arctan2(y - cy, x - cx)
            r_mod = 1.0 / np.sqrt(np.cos(tt - psi) ** 2 / a_maj ** 2
                                  + np.sin(tt - psi) ** 2
                                  / max(a_maj * asp, 1e-9) ** 2)
            resid = np.abs(np.hypot(x - cx, y - cy) - r_mod)
            keep = resid < max(2.5 * float(resid.std()), 1.5)
            if int(keep.sum()) < 24:
                break
            x, y = x[keep], y[keep]
        if x.size < 24:
            return None
        return cx, cy, a_maj, min(max(asp, 0.3), 1.0), psi

    def _conic_of_circle(self, prm, radius):
        t = np.linspace(0.0, 2 * math.pi, 90, endpoint=False)
        pts = np.stack([radius * np.cos(t), radius * np.sin(t)], axis=1)
        x, y = self._project(prm, pts)
        return self._conic_fit(x, y)

    def _fit_pose_conics(self, outer, hole, prm):
        """Two concentric canonical circles (seed rim, hole edge) project to
        the two observed conics. Matching them recovers center, scale, tilt
        and keystone together, including the perspective offset between a
        conic's center and the projected grid center, which is exactly what
        makes a rim-only estimate drift under tilt."""
        targets = []
        if outer is not None:
            targets.append((468.0, outer))
        if hole is not None:
            targets.append((165.0, hole))
        if not targets:
            return prm
        best = dict(prm)

        def cost(p):
            tot = 0.0
            for radius, obs in targets:
                c = self._conic_of_circle(p, radius)
                if c is None:
                    return 1e6
                pcx, pcy, pa, pasp, ppsi = c
                ocx, ocy, oa, oasp, opsi = obs
                tot += ((pcx - ocx) ** 2 + (pcy - ocy) ** 2) / 4.0
                tot += ((pa - oa) / 3.0) ** 2
                # A near-circular conic carries no axis information, and the
                # small hole conic is too noisy for it: take squeeze and its
                # direction only from a clearly eccentric outer rim.
                w = 0.0 if radius < 300.0 else min(max((1.0 - oasp) / 0.05,
                                                       0.0), 1.0)
                tot += w * ((pasp - oasp) / 0.01) ** 2
                dpsi = (ppsi - opsi + math.pi / 2) % math.pi - math.pi / 2
                tot += w * (dpsi / 0.05) ** 2
            return tot

        c0 = cost(best)
        keys = ('cx', 'cy', 'scale', 'aspect', 'theta', 'kx', 'ky')
        x0 = [best[k] for k in keys]
        steps = [4.0, 4.0, 0.02 * best['scale'], 0.03, 0.08, 0.10, 0.10]

        def fvec(v):
            p = dict(best)
            p.update(zip(keys, v))
            p['aspect'] = min(max(p['aspect'], 0.3), 1.0)
            return cost(p)

        # Nelder-Mead: keystone and center trade off against each other, a
        # valley coordinate descent cannot follow.
        sim = [np.array(x0, dtype=np.float64)]
        for i in range(len(keys)):
            v = np.array(x0, dtype=np.float64)
            v[i] += steps[i]
            sim.append(v)
        fv = [fvec(v) for v in sim]
        for _ in range(160):
            order = np.argsort(fv)
            sim = [sim[i] for i in order]
            fv = [fv[i] for i in order]
            if fv[-1] - fv[0] < 1e-7:
                break
            cen = np.mean(sim[:-1], axis=0)
            xr = cen + (cen - sim[-1])
            fr = fvec(xr)
            if fr < fv[0]:
                xe = cen + 2.0 * (cen - sim[-1])
                fe = fvec(xe)
                if fe < fr:
                    sim[-1], fv[-1] = xe, fe
                else:
                    sim[-1], fv[-1] = xr, fr
            elif fr < fv[-2]:
                sim[-1], fv[-1] = xr, fr
            else:
                xc = cen + 0.5 * (sim[-1] - cen)
                fc = fvec(xc)
                if fc < fv[-1]:
                    sim[-1], fv[-1] = xc, fc
                else:
                    for i in range(1, len(keys) + 1):
                        sim[i] = sim[0] + 0.5 * (sim[i] - sim[0])
                        fv[i] = fvec(sim[i])
        order = np.argsort(fv)
        xb = sim[order[0]]
        best.update(zip(keys, xb))
        best['aspect'] = min(max(best['aspect'], 0.3), 1.0)
        return best

    # Copy agreement dies outside roughly +-3 px of center and +-1% of scale,
    # but a bagel pushed toward the frame edge localizes 15 px out, so the
    # rounds start wide and narrow from there.
    COARSE_ROUNDS = [
        (8.0, 4.0, 0.06),
        (4.0, 2.0, 0.04),
        (2.0, 1.0, 0.02),
        (1.0, 0.5, 0.01),
        (0.5, 0.25, 0.004),
    ]

    def _coarse_pose(self, resp_s, prm, k_phi=3):
        """Snap the disc-moment pose onto the alignment optimum.

        Center and scale are searched jointly on a grid rather than one axis
        at a time, because at this resolution a center error can only be paid
        for by a scale error and vice versa.

        Rotation and geometry have to be ordered, and which order works
        depends on which of the two the localization got wrong:

        * Shot straight on, the center lands within a pixel or two, so the
          rotation scan is already meaningful and picks up the roll at once.
        * Pushed toward the frame edge, the center lands several pixels out,
          and then the rotation scan is worse than useless: a couple of pixels
          of center error already moves the agreement argmax tens of degrees,
          so it locks onto an alias and the geometry search that follows has
          no basin left to find.

        So scan rotation first, and only when that lands somewhere poor fall
        back to fixing the geometry at the incoming rotation and scanning
        rotation afterwards.
        """
        s0 = float(prm['scale'])

        def try_set(base, **kw):
            p = dict(base)
            p.update(kw)
            p['aspect'] = min(max(p['aspect'], 0.3), 1.0)
            # The ring grid is radially self-similar, so half/double scale
            # also lands on seeds; the disc radius from _localize rules
            # those harmonics out.
            p['scale'] = min(max(p['scale'], 0.85 * s0), 1.18 * s0)
            return p, self._align_score(resp_s, p)

        def geom_rounds(best, score):
            for i, (crange, cstep, srange) in enumerate(self.COARSE_ROUNDS):
                offs = np.arange(-crange, crange + 1e-9, cstep)
                for dx in offs:
                    for dy in offs:
                        p, s = try_set(best, cx=best['cx'] + float(dx),
                                       cy=best['cy'] + float(dy))
                        if s > score:
                            score, best = s, p
                for ms in np.linspace(1.0 - srange, 1.0 + srange, 9):
                    p, s = try_set(best, scale=best['scale'] * float(ms))
                    if s > score:
                        score, best = s, p
                if i == 1:
                    # Aspect and its axis are coupled: a compression along the
                    # wrong axis scores worse than none at all. Swept once the
                    # widest center round has landed, not before -- at a bad
                    # center a squeeze can only compensate for the offset and
                    # the pair then locks together.
                    for asp in (0.70, 0.80, 0.88, 0.94, 1.0):
                        for th in np.arange(0.0, math.pi, math.pi / 8.0):
                            p, s = try_set(best, aspect=asp, theta=float(th))
                            if s > score:
                                score, best = s, p
            return best, score

        def rotate_then_geom(start):
            best = dict(start)
            score = self._align_score(resp_s, best)
            for _, cand in self._phi_peaks(resp_s, best, 1.0, 0.25,
                                           k_phi, 20.0):
                b, s = geom_rounds(dict(cand), self._align_score(resp_s, cand))
                if s > score:
                    best, score = b, s
            return best, score

        out_best, out_score = rotate_then_geom(prm)
        if out_score < 0.8:
            geom_first, g_score = geom_rounds(dict(prm),
                                              self._align_score(resp_s, prm))
            alt, alt_score = rotate_then_geom(geom_first)
            if g_score > alt_score:
                alt, alt_score = geom_first, g_score
            if alt_score > out_score:
                out_best, out_score = alt, alt_score
        return out_best, out_score

    # Step sizes stay inside the measured tolerances: scale +-0.5%, center
    # +-1.5px, aspect +-0.01, keystone +-0.03, rotation +-0.3 deg. Coarser
    # sweeps drift past the optimum because the alignment score is flat and
    # noisy at that resolution.
    DESCENT_ROUNDS = [
        [('scale', np.linspace(0.994, 1.006, 9), True),
         ('cx', np.arange(-1.5, 1.51, 0.5), False),
         ('cy', np.arange(-1.5, 1.51, 0.5), False),
         ('aspect', np.arange(-0.012, 0.0121, 0.003), False),
         ('theta', np.arange(-0.024, 0.0241, 0.006), False),
         ('kx', np.arange(-0.06, 0.061, 0.015), False),
         ('ky', np.arange(-0.06, 0.061, 0.015), False)],
        [('scale', np.linspace(0.9985, 1.0015, 7), True),
         ('cx', np.arange(-0.4, 0.41, 0.2), False),
         ('cy', np.arange(-0.4, 0.41, 0.2), False),
         ('aspect', np.arange(-0.004, 0.0041, 0.001), False),
         ('theta', np.arange(-0.008, 0.0081, 0.002), False),
         ('kx', np.arange(-0.02, 0.021, 0.005), False),
         ('ky', np.arange(-0.02, 0.021, 0.005), False)],
    ]

    # Residual geometry error the descent can still leave behind; the real
    # decoder arbitrates over these variants.
    GEOM_VARIANTS = [
        {},
        {'scale': 1.0025}, {'scale': 0.9975},
        {'cx': 0.6}, {'cx': -0.6}, {'cy': 0.6}, {'cy': -0.6},
        {'aspect': 0.004}, {'aspect': -0.004},
        {'kx': 0.02}, {'ky': 0.02}, {'kx': -0.02}, {'ky': -0.02},
    ]

    def _descend(self, resp_s, prm, score):
        for spec in self.DESCENT_ROUNDS:
            for key, deltas, mult in spec:
                for dv in deltas:
                    p = dict(prm)
                    p[key] = (prm[key] * float(dv) if mult
                              else prm[key] + float(dv))
                    if key == 'aspect':
                        p[key] = min(max(p[key], 0.3), 1.0)
                    e = self._align_score(resp_s, p)
                    if e > score:
                        score = e
                        prm = p
        return prm, score

    def _phi_scan(self, resp_s, prm, deg_step):
        out = []
        for deg in np.arange(0.0, 360.0, deg_step):
            p = dict(prm)
            p['phi'] = math.radians(float(deg))
            out.append((self._align_score(resp_s, p), float(deg)))
        out.sort(key=lambda item: -item[0])
        return out

    def _phi_peaks(self, resp_s, prm, deg_step, fine_step, k_peaks, sep_deg):
        peaks = []
        chosen = []
        for score, deg in self._phi_scan(resp_s, prm, deg_step):
            if any(min(abs(deg - o), 360 - abs(deg - o)) < sep_deg
                   for o in chosen):
                continue
            local = [(score, deg)]
            for d2 in np.arange(deg - deg_step, deg + deg_step + 1e-9,
                                fine_step):
                p = dict(prm)
                p['phi'] = math.radians(float(d2) % 360.0)
                local.append((self._align_score(resp_s, p), float(d2) % 360.0))
            local.sort(key=lambda item: -item[0])
            score_b, deg_b = local[0]
            p = dict(prm)
            p['phi'] = math.radians(deg_b)
            peaks.append((score_b, p))
            chosen.append(deg_b)
            if len(peaks) >= k_peaks:
                break
        peaks.sort(key=lambda item: -item[0])
        return peaks

    def _refine_pose(self, resp_s, prm, k_peaks=3):
        """Find the in-plane rotation, then tighten the rest of the pose.

        `_coarse_pose` has already put the center and scale inside the
        agreement basin, so the rotation scan is now meaningful: it runs at
        1 deg, coarse enough to bracket the peak since agreement is already
        down to ~0.5 half a degree out, and only then is the geometry
        descended and the rotation rescanned at 0.1 deg. The top peaks are
        all returned, ranked, for the real decoder to arbitrate.
        """
        out = []
        for score, p in self._phi_peaks(resp_s, prm, 1.0, 0.1, k_peaks, 3.0):
            refined, s = self._descend(resp_s, p, score)
            fine = [(s, refined)]
            lo = refined['phi'] - math.radians(0.5)
            for d2 in np.arange(lo, lo + math.radians(1.0) + 1e-12,
                                math.radians(0.05)):
                q = dict(refined)
                q['phi'] = float(d2)
                fine.append((self._align_score(resp_s, q), q))
            fine.sort(key=lambda item: -item[0])
            out.extend(fine[:2])
        out.sort(key=lambda item: -item[0])
        return out[0][1], out

    def _classify(self, vals, valid, hit):
        thresh = self._otsu(vals)[0]
        if thresh is None:
            return None
        fin = np.isfinite(vals)
        below = vals[fin & (vals < thresh)]
        above = vals[fin & (vals >= thresh)]
        if below.size < 100 or above.size < 100:
            return None
        m0 = float(np.median(below))
        m1 = float(np.median(above))
        gap = abs(m1 - m0)
        if gap <= 1e-6:
            return None
        # Erasure band from each mode's own spread (MAD), not from the gap.
        # A gap-proportional band eats the whole tail of the broader mode and
        # skews the vote; MAD scales with the actual noise instead, so a
        # blurry low-contrast shot widens it (more erasures for RS to use)
        # while a clean one keeps almost every cell decisive.
        s0 = float(np.median(np.abs(below - m0)))
        s1 = float(np.median(np.abs(above - m1)))
        floor = 0.05 * gap
        lo_cut = thresh - max(1.3 * s0, floor)
        hi_cut = thresh + max(1.3 * s1, floor)

        out_arr = np.empty(N_PHYSICAL, dtype=object)
        out_arr[:] = None
        usable = valid & (~hit)
        out_arr[usable & (vals > hi_cut)] = 1
        out_arr[usable & (vals < lo_cut)] = 0
        return out_arr.tolist()

    def _pose_candidates(self, arr):
        """Camera-robust front end. Yields cell readings for ranked pose
        candidates; the caller decodes them in order and stops at the first
        payload whose header and SHA-256 check out."""
        gray = np.mean(arr.astype(np.float64), axis=2)
        h, w = gray.shape
        if min(h, w) < 40:
            return

        f = self.ANALYZE_MAX_DIM / float(max(h, w))
        if f < 1.0:
            resized = Image.fromarray(gray.astype(np.uint8)).resize(
                (max(8, int(round(w * f))), max(8, int(round(h * f)))),
                Image.BOX)
            a_gray = np.array(resized).astype(np.float64)
        else:
            f = 1.0
            a_gray = gray

        a_resp = self._tophat(a_gray, 5)
        prm, loc_score = self._localize(a_resp)
        if prm is None:
            print(f"[decode] pose: localization failed (score={loc_score:.3f})")
            return
        prm['cx'] /= f
        prm['cy'] /= f
        prm['scale'] /= f

        th_k = max(3, int(round(12.0 * prm['scale'])))
        resp = self._tophat(gray, th_k)
        # A quarter of the cell pitch: enough to average out grain inside a
        # cell, and it truncates to zero once cells are only a few pixels
        # across, where any blur at all smears seeds into their neighbours.
        sm_rad = int(min(2.25 * prm['scale'], 28.0))
        resp_s = self._box_blur(resp, sm_rad)

        prm, _ = self._coarse_pose(resp_s, prm)
        _, peaks = self._refine_pose(resp_s, prm)

        dmg_rad = max(2, int(round(3.0 * prm['scale'])))
        damage = self._border_white_mask(
            arr, thresh=max(245.0, 0.98 * float(gray.max())))
        damage = self._box_blur(damage.astype(np.float64), dmg_rad) > 0
        dh, dw = damage.shape

        for _, peak in peaks:
            for variant in self.GEOM_VARIANTS:
                p = dict(peak)
                for key, dv in variant.items():
                    p[key] = p[key] * dv if key == 'scale' else p[key] + dv
                    if key == 'aspect':
                        p[key] = min(max(p[key], 0.3), 1.0)
                vals, xs, ys, valid = self._sample_cells(resp_s, p)
                if int(valid.sum()) < 0.35 * N_PHYSICAL:
                    continue
                xi = np.clip(np.round(xs).astype(np.int64), 0, dw - 1)
                yi = np.clip(np.round(ys).astype(np.int64), 0, dh - 1)
                raw = self._classify(vals, valid, damage[yi, xi])
                if raw is not None:
                    yield raw
    def _extract_payload(self, logical, logical_erased, payload_len=None):
        """Metadata-guided decode. If payload_len is None, read it from
        the metadata region of the logical stream."""
        meta_bits = logical[:METADATA_BYTES * 8]
        meta_bytes = bytearray()
        for i in range(METADATA_BYTES):
            val = 0
            for j in range(8):
                val = (val << 1) | meta_bits[i * 8 + j]
            meta_bytes.append(val)

        if payload_len is None:
            version = meta_bytes[0]
            payload_len = struct.unpack('>H', bytes(meta_bytes[1:3]))[0]
            if version not in VERSION_TABLE:
                return None
            if payload_len > VERSION_TABLE[version]:
                return None

        return self._rs_decode_region(logical, logical_erased, payload_len)

    def _rs_decode_region(self, logical, logical_erased, payload_len):
        overhead_bytes = len(VERSION_HEADER) + 2 + 4 + self.ec_level
        data_logical_bytes = overhead_bytes + payload_len
        start = METADATA_BYTES * 8
        end = start + data_logical_bytes * 8
        if end > len(logical):
            return None

        data_bytes = bytearray()
        erasure_byte_positions = []
        for i in range(data_logical_bytes):
            val = 0
            byte_has_erasure = False
            for j in range(8):
                k = start + i * 8 + j
                val = (val << 1) | logical[k]
                if logical_erased[k]:
                    byte_has_erasure = True
            data_bytes.append(val)
            if byte_has_erasure:
                erasure_byte_positions.append(i)

        decoded = self.rs.decode_with_erasures(
            bytes(data_bytes), erasure_byte_positions)
        if decoded is None:
            return None
        if len(decoded) < len(VERSION_HEADER) + 2 + 4:
            return None
        if decoded[:len(VERSION_HEADER)] != VERSION_HEADER:
            return None

        dec_payload_len = struct.unpack(
            '>H', decoded[len(VERSION_HEADER):len(VERSION_HEADER) + 2])[0]
        if dec_payload_len != payload_len:
            return None
        ps = len(VERSION_HEADER) + 2
        pe = ps + dec_payload_len
        if pe + 4 > len(decoded):
            return None

        payload = decoded[ps:pe]
        stored_hash = decoded[pe:pe + 4]
        check = VERSION_HEADER + struct.pack('>H', dec_payload_len) + payload
        expected_hash = hashlib.sha256(check).digest()[:4]

        if stored_hash == expected_hash:
            return bytes(payload)
        return None

    def _decode_exhaustive(self, logical, logical_erased):
        """Fallback: try all payload sizes when metadata is unreadable."""
        overhead_bytes = len(VERSION_HEADER) + 2 + 4 + self.ec_level
        max_payload = N_TOTAL_LOGICAL // 8 - METADATA_BYTES - overhead_bytes

        for payload_guess in range(5, min(250, max_payload) + 1):
            result = self._rs_decode_region(logical, logical_erased, payload_guess)
            if result is not None:
                return result

        print("[decode] All decode attempts failed")
        return None

    def _read_presence(self, arr, grid, damage=None):
        """Sample each grid cell: 1=sesame present, 0=absent, None=erasure.
        Cells that read bright but overlap border-connected white damage
        (torn/cropped regions) are treated as erasures, not as sesame."""
        h, w = arr.shape[:2]
        values = []
        for x, y, ri, ai in grid:
            ix, iy = int(round(x)), int(round(y))
            if 0 <= iy < h and 0 <= ix < w:
                patch = arr[max(0, iy - 2):iy + 3, max(0, ix - 2):ix + 3]
                brightness = float(np.mean(patch))
                if brightness >= self.SESAME_PRESENT_MIN:
                    if damage is not None:
                        dpatch = damage[max(0, iy - 2):iy + 3,
                                        max(0, ix - 2):ix + 3]
                        if dpatch.any():
                            values.append(None)
                            continue
                    values.append(1)
                elif brightness <= self.SESAME_ABSENT_MAX:
                    values.append(0)
                else:
                    values.append(None)
            else:
                values.append(None)
        return values

    @staticmethod
    def _border_white_mask(arr, thresh=250):
        """Flood-fill pure-white regions connected to the image border."""
        gray = np.mean(arr, axis=2)
        h, w = gray.shape
        white = gray >= thresh
        mask = np.zeros((h, w), dtype=bool)
        stack = []
        for x in range(w):
            if white[0, x] and not mask[0, x]:
                mask[0, x] = True
                stack.append((0, x))
            if white[h - 1, x] and not mask[h - 1, x]:
                mask[h - 1, x] = True
                stack.append((h - 1, x))
        for y in range(h):
            if white[y, 0] and not mask[y, 0]:
                mask[y, 0] = True
                stack.append((y, 0))
            if white[y, w - 1] and not mask[y, w - 1]:
                mask[y, w - 1] = True
                stack.append((y, w - 1))
        while stack:
            yy, xx = stack.pop()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = yy + dy, xx + dx
                if 0 <= ny < h and 0 <= nx < w and white[ny, nx] and not mask[ny, nx]:
                    mask[ny, nx] = True
                    stack.append((ny, nx))
        return mask

    def _estimate_scale_from_rings(self, arr, cx, cy, inner_r_est):
        """Find the image scale by matching observed seed-ring radii against
        the canonical ring radii. Unlike the outer-edge estimate this stays
        correct when the rim is cropped or damaged."""
        gray = np.mean(arr, axis=2)
        h, w = gray.shape
        cx, cy = int(round(cx)), int(round(cy))
        ci = int(round(inner_r_est))
        r_max = int(min(w, h) // 2) - 2
        if r_max <= ci + 10:
            return None

        n_angles = 180
        rads = np.arange(ci + 3, r_max)
        counts = np.zeros(len(rads))
        for ai in range(n_angles):
            ang = 2 * math.pi * ai / n_angles
            xs = (cx + rads * math.cos(ang)).astype(int)
            ys = (cy + rads * math.sin(ang)).astype(int)
            ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            b = gray[ys[ok], xs[ok]]
            counts[ok] += (b >= 210)

        peaks = []
        i = 0
        while i < len(rads):
            if counts[i] >= 0.2 * n_angles:
                j = i + 1
                while (j < len(rads) and rads[j] == rads[j - 1] + 1
                       and counts[j] >= 0.2 * n_angles):
                    j += 1
                seg = range(i, j)
                total = float(sum(counts[k] for k in seg))
                if total > 0:
                    peaks.append(sum(rads[k] * counts[k] for k in seg) / total)
                i = j
            else:
                i += 1

        if len(peaks) < 8:
            return None

        canon_outer_r, canon_inner_r = get_ring_params(
            (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))
        n_rings = max(1, int((canon_outer_r - canon_inner_r - 8) / SESAME_SPACING))
        step = (canon_outer_r - canon_inner_r - 8) / n_rings
        ring_r = [canon_inner_r + 4 + step * (k + 0.5) for k in range(n_rings)]

        def score_scale(s):
            score = 0.0
            for p in peaks:
                d = min(abs(p / s - rr) for rr in ring_r)
                if d <= 2.0:
                    score += 1.0 - d / 2.0
            return score

        best_score, best_s = -1.0, 1.0
        for s_int in range(500, 2001, 10):
            s = s_int / 1000.0
            sc = score_scale(s)
            if sc > best_score:
                best_score, best_s = sc, s

        for s_int in range(int(best_s * 1000) - 9, int(best_s * 1000) + 10, 2):
            s = s_int / 1000.0
            sc = score_scale(s)
            if sc > best_score:
                best_score, best_s = sc, s

        if best_score < 0.5 * len(peaks):
            return None
        return best_s

    def _normalize_image(self, img, cx, cy, scale, target_size):
        w, h = img.size
        new_cx = target_size // 2
        new_cy = target_size // 2

        pixels = np.array(img.convert('RGB'))
        out_arr = np.full((target_size, target_size, 3), 255, dtype=np.uint8)

        yy, xx = np.mgrid[:target_size, :target_size]
        src_x = (xx - new_cx) / scale + cx
        src_y = (yy - new_cy) / scale + cy

        valid = (src_x >= 0) & (src_x < w - 1) & (src_y >= 0) & (src_y < h - 1)
        sx_i = np.clip(src_x.astype(int), 0, w - 2)
        sy_i = np.clip(src_y.astype(int), 0, h - 2)

        out_arr[valid] = pixels[sy_i[valid], sx_i[valid]]
        return Image.fromarray(out_arr)

    def _detect_bagel_circle(self, arr):
        gray = np.mean(arr, axis=2)
        h, w = gray.shape

        non_white = gray < 230
        if non_white.sum() < 100:
            return None

        ys, xs = np.where(non_white)

        candidates = [
            ((xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0),
            (w / 2.0, h / 2.0),
        ]

        best_result = None
        best_edge_count = -1
        best_residual = float('inf')

        for cx_init, cy_init in candidates:
            result = self._refine_circle(gray, cx_init, cy_init, h, w)
            if result is None:
                continue
            cx, cy, outer_r, inner_r, n_edge_points, residual = result
            if (n_edge_points > best_edge_count + 10 or
                (abs(n_edge_points - best_edge_count) <= 10 and residual < best_residual)):
                best_edge_count = n_edge_points
                best_residual = residual
                best_result = (cx, cy, outer_r, inner_r)

        return best_result

    def _refine_circle(self, gray, cx_init, cy_init, h, w):
        cx, cy = cx_init, cy_init
        outer_r = None
        inner_edge_points = []

        for iteration in range(5):
            n_angles = 180
            outer_edge_points = []
            inner_edge_points = []

            for ai in range(n_angles):
                angle = 2 * math.pi * ai / n_angles
                dx = math.cos(angle)
                dy = math.sin(angle)

                # Outer edge: dark->bright transition preceded by a long
                # dark run (the bagel body). Short dark runs are sesame
                # seeds, which must not be mistaken for the rim.
                dark_run = 0
                for r in range(5, int(min(w, h) // 2)):
                    px = int(cx + r * dx)
                    py = int(cy + r * dy)
                    if not (0 <= py < h and 0 <= px < w):
                        break
                    bright = gray[py, px]
                    if bright < 220:
                        dark_run += 1
                    else:
                        if dark_run >= 30 and bright > 240:
                            outer_edge_points.append((cx + r * dx, cy + r * dy))
                            break
                        dark_run = 0

                hole_bright = gray[min(max(int(cy), 0), h - 1),
                                   min(max(int(cx), 0), w - 1)]
                for r in range(1, 300):
                    px = int(cx + r * dx)
                    py = int(cy + r * dy)
                    if not (0 <= py < h and 0 <= px < w):
                        break
                    bright = gray[py, px]
                    if bright < hole_bright - 25:
                        inner_edge_points.append((cx + r * dx, cy + r * dy))
                        break

            if len(outer_edge_points) < 6:
                return None

            cx_new, cy_new, outer_r_new = self._fit_circle_robust(outer_edge_points)
            cx, cy = cx_new, cy_new
            outer_r = outer_r_new

        if outer_r is None:
            return None

        if len(inner_edge_points) >= 6:
            _, _, inner_r = self._fit_circle_robust(inner_edge_points)
        else:
            inner_r = max(1, outer_r * RING_INNER_RATIO / RING_OUTER_RATIO)

        dists = [abs(math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2) - outer_r)
                 for p in outer_edge_points]
        residual = sum(dists) / len(dists) if dists else 999

        return (cx, cy, float(outer_r), float(inner_r),
                len(outer_edge_points), residual)

    @staticmethod
    def _fit_circle(points):
        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])

        mx, my = xs.mean(), ys.mean()
        u = xs - mx
        v = ys - my

        Suu = np.sum(u * u)
        Suv = np.sum(u * v)
        Svv = np.sum(v * v)
        Suuu = np.sum(u * u * u)
        Suvv = np.sum(u * v * v)
        Svuu = np.sum(v * u * u)
        Svvv = np.sum(v * v * v)

        A = np.array([[Suu, Suv], [Suv, Svv]])
        b = np.array([0.5 * (Suuu + Suvv), 0.5 * (Svuu + Svvv)])

        try:
            uc, vc = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return mx, my, float(np.sqrt(Suu + Svv))

        cx = mx + uc
        cy = my + vc
        r = float(np.sqrt(uc * uc + vc * vc + (Suu + Svv) / len(points)))
        return cx, cy, r

    @staticmethod
    def _fit_circle_robust(points, max_iter=5):
        pts = list(points)
        for _ in range(max_iter):
            if len(pts) < 6:
                break
            cx, cy, r = SpiralSesameDecoder._fit_circle(pts)
            dists = [abs(math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2) - r)
                     for p in pts]
            median_d = sorted(dists)[len(dists) // 2]
            threshold = max(median_d * 3, 15)
            pts = [p for p, d in zip(pts, dists) if d < threshold]
        if len(pts) < 6:
            return SpiralSesameDecoder._fit_circle(list(points))
        return SpiralSesameDecoder._fit_circle(pts)


# ─── CLI ──────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Spiral Sesame Code v6")
    sub = parser.add_subparsers(dest='cmd')

    enc = sub.add_parser('encode', help='Encode text to image')
    enc.add_argument('text', help='Text to encode')
    enc.add_argument('-o', '--output', default='sesame_code.png')
    enc.add_argument('-s', '--size', type=int, default=DEFAULT_IMAGE_SIZE)
    enc.add_argument('--ec', type=int, default=DEFAULT_EC)
    enc.add_argument('--seed', type=int, default=DEFAULT_SEED)

    dec = sub.add_parser('decode', help='Decode image to text')
    dec.add_argument('image', help='Input image path')
    dec.add_argument('--ec', type=int, default=DEFAULT_EC)
    dec.add_argument('--seed', type=int, default=DEFAULT_SEED)

    info = sub.add_parser('info', help='Show capacity info')
    info.add_argument('-s', '--size', type=int, default=DEFAULT_IMAGE_SIZE)
    info.add_argument('--ec', type=int, default=DEFAULT_EC)

    args = parser.parse_args()

    if args.cmd == 'encode':
        enc_obj = SpiralSesameEncoder(
            image_size=(args.size, args.size),
            ec_level=args.ec, seed=args.seed)
        print(f"Capacity: {enc_obj.capacity_bytes()} bytes")
        print(f"Grid cells: {len(enc_obj.grid)}")
        print(f"Encoding: \"{args.text}\" ({len(args.text.encode('utf-8'))} bytes)")
        img = enc_obj.encode_text(args.text)
        img.save(args.output)
        print(f"Saved: {args.output}")

    elif args.cmd == 'decode':
        dec_obj = SpiralSesameDecoder(ec_level=args.ec, seed=args.seed)
        img = Image.open(args.image)
        print(f"Decoding: {args.image} ({img.size})")
        result = dec_obj.decode_image(img)
        if result:
            print(f"Result: {result}")
        else:
            print("FAILED: Could not decode")

    elif args.cmd == 'info':
        enc_obj = SpiralSesameEncoder(
            image_size=(args.size, args.size), ec_level=args.ec)
        print(f"Image size: {args.size}x{args.size}")
        print(f"EC symbols: {args.ec}")
        print(f"Repeat factor: {REPEAT}")
        print(f"Grid cells: {len(enc_obj.grid)}")
        print(f"Usable capacity: {enc_obj.capacity_bytes()} bytes")
        print(f"Versions: {VERSION_TABLE}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
