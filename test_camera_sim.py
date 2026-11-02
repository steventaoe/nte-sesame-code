"""
Camera-condition simulation suite for the Spiral Sesame Code.

Renders a code, then re-projects it through a pinhole camera model onto a
photo-like frame (tilt, roll, distance, background) and applies the usual
phone-camera degradations (exposure, gamma, white balance, defocus, motion
blur, sensor noise, JPEG, glare, shadow, occlusion). Reports which
conditions still decode.

Usage:
    python test_camera_sim.py            # full matrix
    python test_camera_sim.py tilt roll  # only scenarios whose name matches
"""

import io
import math
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from spiral_sesame_code import (
    DEFAULT_IMAGE_SIZE, SpiralSesameDecoder, SpiralSesameEncoder,
    get_ring_params,
)

URL = ("https://www.xiaohongshu.com/explore/682a1b3c000000000b03abc123"
       "?app_platform=ios&app_version=8.56")
DST = (900, 900)
OUTER_R = get_ring_params((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))[0]


# ─── camera model ────────────────────────────────────────────────────────

def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def plane_homography(src_size, dst_size, tilt_x=0.0, tilt_y=0.0, roll=0.0,
                     dist=3.0, focal=1.0, center=None):
    """Source image plane (z=0) -> destination pixels, pinhole model.
    `dist` is the camera distance in units of the code's half-size, so the
    code spans roughly 2*focal*dst_width/dist pixels."""
    c0 = src_size[0] / 2.0
    R = src_size[0] / 2.0
    A = np.array([[1 / R, 0, -c0 / R], [0, 1 / R, -c0 / R], [0, 0, 1.0]])
    rot = _rot_z(roll) @ _rot_x(tilt_x) @ _rot_y(tilt_y)
    # Plane z=0: camera coords = R@(x,y,0) + (0,0,dist), so the third column is
    # the translation, not R's own third column.
    M = np.column_stack([rot[:, 0], rot[:, 1], [0.0, 0.0, dist]])
    f = focal * dst_size[0]
    K = np.array([[f, 0, 0], [0, f, 0], [0, 0, 1.0]])
    cx = dst_size[0] / 2.0 if center is None else center[0]
    cy = dst_size[1] / 2.0 if center is None else center[1]
    T = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1.0]])
    return T @ K @ M @ A


def to_rgba(code_img):
    """Give the rendered code a transparent surround so it can be composited
    onto an arbitrary photo background."""
    rgba = code_img.convert('RGBA')
    size = code_img.size
    yy, xx = np.mgrid[0:size[1], 0:size[0]]
    d = np.hypot(xx - size[0] / 2.0, yy - size[1] / 2.0)
    alpha = np.where(d <= OUTER_R + 1, 255, 0).astype(np.uint8)
    rgba.putalpha(Image.fromarray(alpha))
    return rgba


def make_background(kind, size=DST, rng=None):
    rng = rng or np.random.RandomState(7)
    w, h = size
    if kind == 'white':
        arr = np.full((h, w, 3), 252.0)
    elif kind == 'dark':
        arr = np.full((h, w, 3), 38.0)
    elif kind == 'gray':
        arr = np.full((h, w, 3), 128.0)
    elif kind == 'wood':
        yy, xx = np.mgrid[0:h, 0:w]
        grain = 18 * np.sin(xx / 9.0 + 2.4 * np.sin(yy / 55.0))
        base = np.stack([148 + grain, 104 + grain * 0.7, 66 + grain * 0.5], -1)
        arr = base
    else:  # cluttered desk
        arr = np.full((h, w, 3), 200.0)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(img)
        for _ in range(26):
            x0 = rng.randint(0, w); y0 = rng.randint(0, h)
            d.ellipse([x0, y0, x0 + rng.randint(40, 260), y0 + rng.randint(40, 260)],
                      fill=tuple(int(c) for c in rng.randint(60, 235, 3)))
        arr = np.array(img).astype(np.float64)
    arr = arr + rng.normal(0, 3, arr.shape)
    return np.clip(arr, 0, 255).astype(np.uint8)


def render_photo(code_img, tilt_x=0.0, tilt_y=0.0, roll=0.0, dist=3.0,
                 focal=1.0, center=None, bg='wood', size=DST):
    H = plane_homography(code_img.size, size, tilt_x, tilt_y, roll, dist,
                         focal, center)
    Hi = np.linalg.inv(H)
    Hi = Hi / Hi[2, 2]
    coeffs = [Hi[0, 0], Hi[0, 1], Hi[0, 2],
              Hi[1, 0], Hi[1, 1], Hi[1, 2], Hi[2, 0], Hi[2, 1]]
    src = to_rgba(code_img)
    warped = src.transform(size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    canvas = Image.fromarray(make_background(bg, size))
    canvas.paste(warped, (0, 0), warped.split()[-1])
    return canvas


# ─── degradations ────────────────────────────────────────────────────────

def fx_exposure(img, gain=1.0, gamma=1.0, wb=(1.0, 1.0, 1.0)):
    arr = np.array(img).astype(np.float64)
    arr = 255.0 * np.power(np.clip(arr, 0, 255) / 255.0, gamma) * gain
    for c in range(3):
        arr[:, :, c] *= wb[c]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def fx_noise(img, sigma):
    rng = np.random.RandomState(11)
    arr = np.array(img).astype(np.float64)
    arr += rng.normal(0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def fx_motion_blur(img, length=9, angle_deg=25.0):
    # PIL's ImageFilter.Kernel only accepts a 5x5 kernel, so the line is
    # accumulated from integer-shifted copies instead.
    arr = np.array(img).astype(np.float64)
    rad = math.radians(angle_deg)
    dx, dy = math.cos(rad), math.sin(rad)
    out = np.zeros_like(arr)
    for i in range(length):
        t = i - (length - 1) / 2.0
        out += np.roll(np.roll(arr, int(round(t * dy)), axis=0),
                       int(round(t * dx)), axis=1)
    return Image.fromarray(np.clip(out / length, 0, 255).astype(np.uint8))


def fx_glare(img, strength=0.85, at=(0.35, 0.3)):
    arr = np.array(img).astype(np.float64)
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w * at[0], h * at[1]
    rad = math.hypot(w, h) * 0.28
    fall = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (rad / 2.2) ** 2))
    arr = arr + (255.0 - arr) * (strength * fall)[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def fx_shadow(img, direction='x', depth=0.55):
    arr = np.array(img).astype(np.float64)
    h, w = arr.shape[:2]
    t = np.linspace(0, 1, w if direction == 'x' else h)
    ramp = 1.0 - depth * t
    ramp = ramp[None, :] if direction == 'x' else ramp[:, None]
    arr = arr * ramp[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def fx_occlusion(img, frac=0.15):
    arr = np.array(img).astype(np.float64)
    h, w = arr.shape[:2]
    r = math.sqrt(frac) * min(w, h) * 0.62
    cx, cy = w * 0.62, h * 0.58
    # Blur the finger's own silhouette, not the whole frame: a sigma-2 pass
    # over the photo wipes out seeds that are only ~2px across, which would
    # make this a blur test rather than an occlusion test.
    layer = Image.new('L', (w, h), 0)
    ImageDraw.Draw(layer).ellipse(
        [cx - r, cy - r * 0.7, cx + r, cy + r * 0.7], fill=255)
    mask = layer.filter(ImageFilter.GaussianBlur(2.0))
    finger = Image.new('RGB', (w, h), (58, 44, 40))
    return Image.composite(finger, Image.fromarray(arr.astype(np.uint8)), mask)


def fx_jpeg(img, quality):
    buf = io.BytesIO()
    img.convert('RGB').save(buf, 'JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def fx_resize(img, factor):
    w, h = img.size
    small = img.resize((max(8, int(w * factor)), max(8, int(h * factor))),
                       Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


# ─── scenarios ───────────────────────────────────────────────────────────

def scenarios(code_img):
    """Each entry: (group, name, callable -> photo image)."""
    out = []

    def add(group, name, fn):
        out.append((group, name, fn))

    for dist, label in [(2.0, 'fills-frame ~900px'), (2.6, '~690px'),
                        (4.0, '~450px'), (5.5, '~330px'), (7.0, '~260px'),
                        (10.0, '~180px'), (14.0, '~130px')]:
        add('distance', label,
            lambda d=dist: render_photo(code_img, dist=d, bg='wood'))

    for deg in (10, 20, 30, 40, 50, 60):
        a = math.radians(deg)
        add('tilt', f'{deg} deg about X',
            lambda a=a: render_photo(code_img, tilt_x=a, dist=2.8, bg='gray'))
    for deg in (20, 35, 50):
        a = math.radians(deg)
        add('tilt', f'{deg} deg about X+Y',
            lambda a=a: render_photo(code_img, tilt_x=a, tilt_y=a, dist=2.8,
                                     bg='gray'))

    for deg in (0, 30, 90, 143, 180, 213, 270):
        a = math.radians(deg)
        add('roll', f'in-plane {deg} deg',
            lambda a=a: render_photo(code_img, roll=a, dist=2.8, bg='white'))

    for off in [(0.28, 0.3), (0.72, 0.68), (0.2, 0.75)]:
        add('offset', f'off-center {off}',
            lambda o=off: render_photo(code_img, dist=3.0, center=(o[0] * DST[0],
                                                               o[1] * DST[1]),
                                       bg='clutter'))

    add('light', 'dim (gain 0.40)', lambda: fx_exposure(
        render_photo(code_img, dist=2.8, bg='wood'), gain=0.40))
    add('light', 'dark (gain 0.22)', lambda: fx_exposure(
        render_photo(code_img, dist=2.8, bg='wood'), gain=0.22))
    add('light', 'bright (gain 1.6)', lambda: fx_exposure(
        render_photo(code_img, dist=2.8, bg='wood'), gain=1.6))
    add('light', 'blown out (gain 2.4)', lambda: fx_exposure(
        render_photo(code_img, dist=2.8, bg='wood'), gain=2.4))
    add('light', 'gamma 0.5', lambda: fx_exposure(
        render_photo(code_img, dist=2.8, bg='wood'), gamma=0.5))
    add('light', 'gamma 2.0', lambda: fx_exposure(
        render_photo(code_img, dist=2.8, bg='wood'), gamma=2.0))
    add('light', 'warm WB', lambda: fx_exposure(
        render_photo(code_img, dist=2.8, bg='wood'), wb=(1.25, 1.0, 0.72)))
    add('light', 'cool WB', lambda: fx_exposure(
        render_photo(code_img, dist=2.8, bg='wood'), wb=(0.75, 1.0, 1.3)))

    add('blur', 'defocus sigma 0.6', lambda: render_photo(
        code_img, dist=2.8, bg='gray').filter(ImageFilter.GaussianBlur(0.6)))
    add('blur', 'defocus sigma 1.2', lambda: render_photo(
        code_img, dist=2.8, bg='gray').filter(ImageFilter.GaussianBlur(1.2)))
    add('blur', 'defocus sigma 2.0', lambda: render_photo(
        code_img, dist=2.8, bg='gray').filter(ImageFilter.GaussianBlur(2.0)))
    add('blur', 'motion blur 9px', lambda: fx_motion_blur(
        render_photo(code_img, dist=2.8, bg='gray')))

    for q in (30, 50, 75, 92):
        add('jpeg', f'quality {q}', lambda q=q: fx_jpeg(
            render_photo(code_img, dist=2.8, bg='wood'), q))

    for sig in (4, 10, 20):
        add('noise', f'sensor noise sigma {sig}', lambda s=sig: fx_noise(
            render_photo(code_img, dist=2.8, bg='wood'), s))

    add('glare', 'specular hotspot', lambda: fx_glare(
        render_photo(code_img, dist=2.8, bg='gray')))
    add('shadow', 'half-frame shadow', lambda: fx_shadow(
        render_photo(code_img, dist=2.8, bg='gray')))
    add('occlusion', 'finger covering 15%', lambda: fx_occlusion(
        render_photo(code_img, dist=2.8, bg='gray'), 0.15))
    add('occlusion', 'finger covering 28%', lambda: fx_occlusion(
        render_photo(code_img, dist=2.8, bg='gray'), 0.28))
    add('resample', 'platform downscale 0.45', lambda: fx_resize(
        render_photo(code_img, dist=2.6, bg='wood'), 0.45))

    for bgk in ('white', 'dark', 'gray', 'wood', 'clutter'):
        add('background', bgk, lambda b=bgk: render_photo(
            code_img, dist=2.8, bg=b))

    add('combined', 'realistic handheld', lambda: fx_jpeg(fx_noise(fx_exposure(
        fx_glare(render_photo(code_img, tilt_x=math.radians(22),
                              tilt_y=math.radians(-14),
                              roll=math.radians(37), dist=3.1,
                              center=(0.54 * DST[0], 0.46 * DST[1]),
                              bg='wood').filter(ImageFilter.GaussianBlur(0.7)),
                             0.7), gain=1.12, gamma=0.85,
        wb=(1.12, 1.0, 0.88)), 7), 72))

    add('combined', 'worst case: tilted + dim + far + jpeg50',
        lambda: fx_jpeg(fx_exposure(render_photo(
            code_img, tilt_x=math.radians(38), roll=math.radians(120),
            dist=4.6, bg='dark'), gain=0.45, gamma=1.4), 50))

    return out


def main():
    filters = [a.lower() for a in sys.argv[1:]]
    print(f"Encoding {len(URL.encode())} bytes ...")
    code_img = SpiralSesameEncoder().encode_text(URL)
    decoder = SpiralSesameDecoder()
    cases = scenarios(code_img)

    rows = []
    for group, name, fn in cases:
        if filters and not any(f in group.lower() or f in name.lower()
                               for f in filters):
            continue
        photo = fn()
        t0 = time.time()
        got = decoder.decode_image(photo, try_hard=False)
        fast = time.time() - t0
        hard = None
        if got != URL:
            t1 = time.time()
            got = decoder.decode_image(photo, try_hard=True)
            hard = time.time() - t1
        ok = got == URL
        rows.append((group, name, ok, fast, hard))
        tag = 'PASS' if ok else 'FAIL'
        extra = '' if hard is None else f' (+{hard:.1f}s hard)'
        print(f"[{tag}] {group:<11} {name:<38} {fast:.1f}s{extra}")
        sys.stdout.flush()

    total = len(rows)
    passed = sum(1 for r in rows if r[2])
    print("\n" + "=" * 74)
    print(f"PASS {passed}/{total}")
    by_group = {}
    for group, name, ok, _, _ in rows:
        by_group.setdefault(group, [0, 0])
        by_group[group][1] += 1
        by_group[group][0] += 1 if ok else 0
    for group, (p, n) in by_group.items():
        bar = '#' * int(round(20 * p / n))
        print(f"  {group:<11} {p:>2}/{n:<2} {bar}")
    fails = [f"{g}/{n}" for g, n, ok, _, _ in rows if not ok]
    if fails:
        print("\nFAILED: " + ", ".join(fails))


if __name__ == '__main__':
    main()
