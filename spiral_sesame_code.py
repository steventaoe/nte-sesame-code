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

    def decode_image(self, img: Image.Image) -> Optional[str]:
        result = self.decode_bytes(img)
        if result is None:
            return None
        try:
            return result.decode('utf-8')
        except UnicodeDecodeError:
            return result.hex()

    def decode_bytes(self, img: Image.Image) -> Optional[bytes]:
        img = img.convert('RGB')
        arr = np.array(img)

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

        grid = build_polar_grid(canon_cx, canon_cy,
                                canon_inner_r + 4, canon_outer_r - 4,
                                SESAME_SPACING,
                                fixed_count=CANONICAL_GRID_COUNT)

        raw_values = self._read_presence(narr, grid, damage)

        interleaver = SectorInterleaver(grid, N_PHYSICAL, REPEAT, seed=self.seed)
        ordered = interleaver.deinterleave_with_erasures(raw_values[:N_PHYSICAL])

        # Majority vote per REPEAT group, then de-whiten.
        # Fully-erased groups take the whitening bit and are flagged so
        # Reed-Solomon can use erasure decoding (double correction power).
        logical = []
        logical_erased = []
        for g in range(N_TOTAL_LOGICAL):
            group = ordered[g * REPEAT:(g + 1) * REPEAT]
            ones = sum(1 for v in group if v == 1)
            zeros = sum(1 for v in group if v == 0)
            nones = sum(1 for v in group if v is None)
            if ones > zeros:
                b = 1
            elif zeros > ones:
                b = 0
            else:
                b = self.whitening[g]
            logical.append(b ^ self.whitening[g])
            logical_erased.append(nones >= 3)

        result = self._extract_payload(logical, logical_erased)
        if result is not None:
            return result
        return self._decode_exhaustive(logical, logical_erased)

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
