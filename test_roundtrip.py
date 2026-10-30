"""Round-trip test: encode URL -> save -> load -> decode -> verify"""
import sys
sys.path.insert(0, ".")

from spiral_sesame_code import SpiralSesameEncoder, SpiralSesameDecoder
from PIL import Image

TEST_URLS = [
    ("bilibili_short", "https://b23.tv/BV1xx411c7mD"),
    ("bilibili_full", "https://www.bilibili.com/video/BV1GJ411x7h7/?spm_id_from=333.999.0.0"),
    ("xiaohongshu", "https://www.xiaohongshu.com/explore/682a1b3c000000000b03abc123?app_platform=ios&app_version=8.56"),
    ("short_ascii", "Hello Bagel!"),
]

encoder = SpiralSesameEncoder()
decoder = SpiralSesameDecoder()

print(f"Encoder capacity: {encoder.capacity_bytes()} bytes\n")

all_pass = True
for name, url in TEST_URLS:
    url_bytes = url.encode('utf-8')
    print(f"[{name}] {len(url_bytes)} bytes: {url[:60]}...")

    try:
        img = encoder.encode_text(url)
        fname = f"test_{name}.png"
        img.save(fname)

        loaded = Image.open(fname)
        decoded = decoder.decode_image(loaded)

        if decoded == url:
            print(f"  PASS - round-trip OK")
        else:
            print(f"  FAIL - got: {decoded}")
            all_pass = False
    except Exception as e:
        print(f"  ERROR: {e}")
        all_pass = False
    print()

print("=" * 40)
print("ALL PASS" if all_pass else "SOME FAILED")
