"""Generate v3 demo images with realistic bagel appearance."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from spiral_sesame_code import SpiralSesameEncoder, SpiralSesameDecoder
from PIL import Image

encoder = SpiralSesameEncoder()
decoder = SpiralSesameDecoder()
print("Capacity: %d bytes" % encoder.capacity_bytes())

cases = [
    ("bilibili", "https://www.bilibili.com/video/BV1GJ411x7h7/?spm_id_from=333.999.0.0"),
    ("xiaohongshu", "https://www.xiaohongshu.com/explore/682a1b3c000000000b03abc123?app_platform=ios&app_version=8.56"),
    ("short_url", "https://b23.tv/BV1xx411c7mD"),
]

for name, url in cases:
    fname = "demo_%s.png" % name
    img = encoder.encode_text(url)
    img.save(fname)
    loaded = Image.open(fname)
    decoded = decoder.decode_image(loaded)
    status = "OK" if decoded == url else "FAIL"
    print("[%s] %s (%d bytes)" % (status, fname, len(url.encode('utf-8'))))
