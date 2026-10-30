"""
螺旋芝麻码 (Spiral Sesame Matrix) v5
=====================================
- 粉色贝果, 芝麻均匀分布在整个环面上
- QR码式版本分级: 固定位置元数据标明数据范围
- 数据范围外的芝麻置0(黑芝麻), 保证视觉均匀
- 五重重复编码 + RS 纠错, 扇区感知交织

容量: ~142 bytes (1024px), 支持 B站/小红书完整链接。
"""

import math
import struct
import hashlib
import random
from typing import List, Tuple, Optional

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
            s = int((a + math.pi) / (2 * math.pi) * repeat) % repeat
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
VERSION_HEADER = b'SSM5'
REPEAT = 5

RING_OUTER_RATIO = 0.46
RING_INNER_RATIO = 0.16
SESAME_SPACING = 9.0

# Version table: version_id → max payload bytes
# Similar to QR code versions, each version defines capacity
VERSION_TABLE = {
    1: 30,    # Short URLs
    2: 70,    # Medium URLs
    3: 142,   # Full URLs (max for 1024px)
}

# Metadata region: first N grid cells reserved for version + length info
# Format: 1 byte version + 2 bytes payload_length = 3 bytes = 24 bits
# With REPEAT=5: 24 * 5 = 120 physical cells for metadata
METADATA_BYTES = 3
METADATA_PHYSICAL = METADATA_BYTES * 8 * REPEAT


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


def select_version(payload_len: int) -> int:
    for v in sorted(VERSION_TABLE.keys()):
        if payload_len <= VERSION_TABLE[v]:
            return v
    raise ValueError(f"Data too long: {payload_len} bytes")


# ─── Renderer ────────────────────────────────────────────────────────────

class SesameRenderer:
    def __init__(self):
        self._cache = {}

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

        # Pink bagel palette
        bagel_r = 235.0 * shade + 20
        bagel_g = 155.0 * shade + 40
        bagel_b = 175.0 * shade + 45

        for c, ch in enumerate([bagel_r, bagel_g, bagel_b]):
            result[:, :, c] = np.where(ring, ch, result[:, :, c])

        hole = dist <= inner_r
        hole_color = np.array([245, 230, 235], dtype=np.float32)
        for c in range(3):
            result[:, :, c] = np.where(hole, hole_color[c], result[:, :, c])

        tex = rng.normal(0, 8, result.shape)
        result = np.where(ring[..., None], result + tex, result)
        result = np.clip(result, 0, 255).astype(np.uint8)

        img = Image.fromarray(result).filter(ImageFilter.GaussianBlur(1.5))

        draw = ImageDraw.Draw(img)
        for off in range(2):
            draw.ellipse([cx - outer_r + off, cy - outer_r + off,
                          cx + outer_r - off, cy + outer_r - off],
                         outline=(180, 100, 120))
            draw.ellipse([cx - inner_r - off, cy - inner_r - off,
                          cx + inner_r + off, cy + inner_r + off],
                         outline=(180, 100, 120))
        return img

    def _get_sesame(self, size=11):
        if size in self._cache:
            return self._cache[size]
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        w, h = size, int(size * 0.55)
        ox, oy = (size - w) // 2, (size - h) // 2
        draw.ellipse([ox, oy, ox + w, oy + h], fill=(235, 220, 170, 240))
        draw.ellipse([ox + w // 4, oy + 1, ox + w * 3 // 4, oy + h // 2],
                     fill=(248, 238, 205, 200))
        draw.ellipse([ox, oy, ox + w, oy + h], outline=(185, 165, 110, 130))
        self._cache[size] = img
        return img

    def draw_sesame(self, canvas, x, y, bit, rotation=0.0):
        tpl = self._get_sesame().copy()
        if rotation != 0:
            tpl = tpl.rotate(rotation, resample=Image.BICUBIC, expand=True)

        arr = np.array(tpl, dtype=np.float32)
        if bit == 0:
            arr[:, :, 0] = np.clip(arr[:, :, 0] * 0.35, 0, 255)
            arr[:, :, 1] = np.clip(arr[:, :, 1] * 0.30, 0, 255)
            arr[:, :, 2] = np.clip(arr[:, :, 2] * 0.25, 0, 255)
        else:
            arr[:, :, :3] = np.clip(arr[:, :, :3] + 5, 0, 255)

        tpl = Image.fromarray(arr.astype(np.uint8))
        px = int(x - tpl.width // 2)
        py = int(y - tpl.height // 2)
        canvas.paste(tpl, (px, py), tpl)


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
        self.grid_count = len(raw_grid)
        self.grid = raw_grid

    def capacity_bytes(self):
        total_logical_bits = len(self.grid) // REPEAT
        total_bytes = total_logical_bits // 8
        # Subtract metadata, hash, EC
        overhead = METADATA_BYTES + 4 + self.ec_level
        return max(0, total_bytes - overhead)

    def encode_text(self, text: str) -> Image.Image:
        return self.encode_bytes(text.encode('utf-8'))

    def encode_bytes(self, data: bytes) -> Image.Image:
        cap = self.capacity_bytes()
        if len(data) > cap:
            raise ValueError(f"Data too long: {len(data)} bytes, max {cap}")

        version = select_version(len(data))

        # Metadata: version(1B) + payload_length(2B) — NOT RS-encoded, stored raw with REPEAT
        metadata = struct.pack('>BH', version, len(data))
        meta_bits = []
        for byte in metadata:
            for i in range(7, -1, -1):
                meta_bits.append((byte >> i) & 1)

        # Payload: header + data + hash — RS-encoded
        payload = VERSION_HEADER + struct.pack('>H', len(data)) + data
        digest = hashlib.sha256(payload).digest()[:4]
        full = payload + digest
        encoded = self.rs.encode(full)

        data_bits = []
        for byte in encoded:
            for i in range(7, -1, -1):
                data_bits.append((byte >> i) & 1)

        # Total physical cells needed for metadata + data
        n_meta_physical = len(meta_bits) * REPEAT
        n_data_physical = len(data_bits) * REPEAT
        n_used_physical = n_meta_physical + n_data_physical

        # Build the full bit stream: metadata bits, then data bits, then padding zeros
        all_logical_bits = meta_bits + data_bits
        n_total_logical = len(self.grid) // REPEAT
        # Pad remaining logical bits with 0
        while len(all_logical_bits) < n_total_logical:
            all_logical_bits.append(0)

        # Repeat each bit REPEAT times
        physical_bits = []
        for b in all_logical_bits:
            physical_bits.extend([b] * REPEAT)

        n_physical = len(physical_bits)

        # Sector-aware interleaving across ALL grid cells
        interleaver = SectorInterleaver(self.grid, n_physical, REPEAT, seed=self.seed)
        scattered = interleaver.interleave(physical_bits)

        # Render: draw sesame at EVERY grid position (uniform distribution)
        canvas = self.renderer.create_bagel(self.image_size, seed=self.seed)
        rng = LCG(self.seed + 1)

        for i in range(min(n_physical, len(self.grid))):
            x, y, ri, ai = self.grid[i]
            bit = scattered[i]
            rot = rng.uniform(-60, 60)
            self.renderer.draw_sesame(canvas, x, y, bit, rot)

        canvas = canvas.filter(ImageFilter.GaussianBlur(0.3))
        return canvas


# ─── Decoder ─────────────────────────────────────────────────────────────

class SpiralSesameDecoder:
    # Adjusted thresholds for pink bagel background (~180-220 brightness)
    BLACK_SESAME_MAX = 110
    WHITE_SESAME_MIN = 190

    def __init__(self, ec_level=DEFAULT_EC, seed=DEFAULT_SEED):
        self.ec_level = ec_level
        self.seed = seed
        self.rs = ReedSolomon(nsym=ec_level)

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
        w, h = img.size
        arr = np.array(img)

        circle = self._detect_bagel_circle(arr)
        if circle is None:
            print("[decode] Failed to detect bagel circle")
            return None

        cx, cy, detected_outer_r, detected_inner_r = circle

        canon_cx = DEFAULT_IMAGE_SIZE // 2
        canon_cy = DEFAULT_IMAGE_SIZE // 2
        canon_outer_r, canon_inner_r = get_ring_params((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))

        scale = canon_outer_r / max(detected_outer_r, 1)
        normalized = self._normalize_image(img, cx, cy, scale, DEFAULT_IMAGE_SIZE)
        narr = np.array(normalized)

        grid = build_polar_grid(canon_cx, canon_cy,
                                canon_inner_r + 4, canon_outer_r - 4,
                                SESAME_SPACING,
                                fixed_count=CANONICAL_GRID_COUNT)

        raw_values = self._read_trinary(narr, grid)

        # Use the SAME interleaver size as encoder: full grid
        n_total_logical = CANONICAL_GRID_COUNT // REPEAT
        n_physical = n_total_logical * REPEAT
        full_interleaver = SectorInterleaver(grid, n_physical, REPEAT, seed=self.seed)
        all_ordered = full_interleaver.deinterleave_with_erasures(raw_values[:n_physical])

        # Step 1: Read metadata from the unified deinterleaved stream
        meta_logical_count = METADATA_BYTES * 8
        meta_bits = []
        for g in range(meta_logical_count):
            group = all_ordered[g * REPEAT:(g + 1) * REPEAT]
            ones = sum(1 for v in group if v == 1)
            zeros = sum(1 for v in group if v == 0)
            if ones > zeros:
                meta_bits.append(1)
            elif zeros > ones:
                meta_bits.append(0)
            else:
                meta_bits.append(0)

        meta_bytes = bytearray()
        for i in range(METADATA_BYTES):
            val = 0
            for j in range(8):
                val = (val << 1) | meta_bits[i * 8 + j]
            meta_bytes.append(val)

        version = meta_bytes[0]
        payload_len = struct.unpack('>H', bytes(meta_bytes[1:3]))[0]

        if version in VERSION_TABLE and payload_len <= VERSION_TABLE[version]:
            result = self._decode_payload(all_ordered, payload_len)
            if result is not None:
                return result

        # Fallback: try exhaustive search
        return self._decode_exhaustive(all_ordered)

    def _decode_payload(self, ordered, payload_len):
        """Decode data region from unified deinterleaved stream."""
        overhead_bytes = len(VERSION_HEADER) + 2 + 4 + self.ec_level
        data_logical_bytes = overhead_bytes + payload_len
        data_start_logical = METADATA_BYTES * 8
        data_end_logical = data_start_logical + data_logical_bytes * 8

        if data_end_logical * REPEAT > len(ordered):
            return None

        logical_bits = []
        bit_erased = []
        for g in range(data_start_logical, data_end_logical):
            group = ordered[g * REPEAT:(g + 1) * REPEAT]
            ones = sum(1 for v in group if v == 1)
            zeros = sum(1 for v in group if v == 0)
            nones = sum(1 for v in group if v is None)
            if ones > zeros:
                logical_bits.append(1)
            elif zeros > ones:
                logical_bits.append(0)
            else:
                logical_bits.append(0)
            bit_erased.append(nones == REPEAT)

        try_bytes = len(logical_bits) // 8
        data_bytes = bytearray()
        erasure_byte_positions = []
        for i in range(try_bytes):
            val = 0
            byte_has_erasure = False
            for j in range(8):
                val = (val << 1) | logical_bits[i * 8 + j]
                if bit_erased[i * 8 + j]:
                    byte_has_erasure = True
            data_bytes.append(val)
            if byte_has_erasure:
                erasure_byte_positions.append(i)

        decoded = self.rs.decode_with_erasures(bytes(data_bytes), erasure_byte_positions)
        if decoded is None:
            return None

        if len(decoded) < len(VERSION_HEADER) + 2 + 4:
            return None
        if decoded[:len(VERSION_HEADER)] != VERSION_HEADER:
            return None

        dec_payload_len = struct.unpack('>H',
            decoded[len(VERSION_HEADER):len(VERSION_HEADER) + 2])[0]
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

    def _decode_exhaustive(self, ordered):
        """Fallback: try all payload sizes when metadata is unreadable."""
        overhead_bytes = len(VERSION_HEADER) + 2 + 4 + self.ec_level
        n_total_logical = CANONICAL_GRID_COUNT // REPEAT
        max_payload = n_total_logical // 8 - METADATA_BYTES - overhead_bytes

        for payload_guess in range(5, min(250, max_payload) + 1):
            result = self._decode_payload(ordered, payload_guess)
            if result is not None:
                return result

        print("[decode] All decode attempts failed")
        return None

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

        for iteration in range(5):
            n_angles = 180
            outer_edge_points = []
            inner_edge_points = []

            for ai in range(n_angles):
                angle = 2 * math.pi * ai / n_angles
                dx = math.cos(angle)
                dy = math.sin(angle)

                prev_bright = None
                for r in range(5, int(min(w, h) // 2)):
                    px = int(cx + r * dx)
                    py = int(cy + r * dy)
                    if not (0 <= py < h and 0 <= px < w):
                        break
                    bright = gray[py, px]
                    if prev_bright is not None and prev_bright < 220 and bright > 240:
                        outer_edge_points.append((cx + r * dx, cy + r * dy))
                        break
                    prev_bright = bright

                hole_bright = gray[min(max(int(cy), 0), h-1), min(max(int(cx), 0), w-1)]
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

        dists = [abs(math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2) - outer_r) for p in outer_edge_points]
        residual = sum(dists) / len(dists) if dists else 999

        return (cx, cy, float(outer_r), float(inner_r), len(outer_edge_points), residual)

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
            dists = [abs(math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2) - r) for p in pts]
            median_d = sorted(dists)[len(dists)//2]
            threshold = max(median_d * 3, 15)
            pts = [p for p, d in zip(pts, dists) if d < threshold]
        if len(pts) < 6:
            return SpiralSesameDecoder._fit_circle(list(points))
        return SpiralSesameDecoder._fit_circle(pts)

    def _read_trinary(self, arr, grid):
        h, w = arr.shape[:2]
        values = []
        for x, y, ri, ai in grid:
            ix, iy = int(round(x)), int(round(y))
            if 0 <= iy < h and 0 <= ix < w:
                patch = arr[max(0, iy - 2):iy + 3, max(0, ix - 2):ix + 3]
                brightness = float(np.mean(patch))
                if brightness < self.BLACK_SESAME_MAX:
                    values.append(0)
                elif brightness > self.WHITE_SESAME_MIN:
                    values.append(1)
                else:
                    values.append(None)
            else:
                values.append(None)
        return values


# ─── CLI ──────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Spiral Sesame Code v5")
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
