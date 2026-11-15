"""Cuts the CareerMetric wordmark and logo out of the title slide as transparent navy PNGs.
    ../../../.venv/bin/python extract_brand.py
"""
from PIL import Image
import numpy as np

NAVY = np.array([3, 57, 130.])          # brand navy sampled from the slide
im = np.asarray(Image.open("careermetric_slide.webp").convert("RGB")).astype(float)

def extract(box, name, gamma=0.55):
    x0, y0, x1, y1 = box
    a = im[y0:y1, x0:x1]
    bg = a[2, 2]                                     # cream paper colour at the crop's corner
    ink = np.clip((bg.mean() - a.mean(axis=2)) / (bg.mean() - NAVY.mean()), 0, 1)
    ink[ink < 0.06] = 0                              # drop paper texture
    alpha = ink ** gamma                             # strengthen thin hairlines so they survive downscaling
    out = np.zeros(a.shape[:2] + (4,), dtype=np.uint8)
    out[..., :3] = NAVY.astype(np.uint8); out[..., 3] = (alpha * 255).astype(np.uint8)
    img = Image.fromarray(out, "RGBA"); img = img.crop(img.getbbox())
    img.save(name, optimize=True); print(name, img.size)

extract((300, 390, 1600, 725), "wordmark.png")
extract((158, 862, 342, 1006), "logo.png")
