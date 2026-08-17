"""Create a labeled blur level grid for meeting presentation."""

# Per-machine paths: see env.sh / SYNC.md at the repo root.  Inlined rather
# than imported from ambient_paths because these scripts run from varying
# depths and cwds, where an import would need sys.path surgery.
import os as _os
AMBIENT_BASE = _os.environ.get("AMBIENT_BASE") or (
    "/data-local/honjar" if _os.path.isdir("/data-local/honjar") else "/data/scratch/honjar"
)

import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random

bucket_dir = f"{AMBIENT_BASE}/celeba_processed/shared_buckets_64"
clean_files = sorted([f for f in os.listdir(bucket_dir) if f.startswith("b0_") and not f.startswith("._")])
random.seed(42)
sample_faces = random.sample(clean_files, 3)

blur_sigmas = [0, 0.5, 1, 2, 3, 4, 5, 8]

# Clear labels: bucket number, role, blur sigma, size
bucket_info = [
    ("Bucket 0", "CLEAN TARGET", "sigma = 0", "~22,825 imgs"),
    ("Bucket 1", "Blur level 1", "sigma = 0.5", "~22,825 imgs"),
    ("Bucket 2", "Blur level 2", "sigma = 1.0", "~22,825 imgs"),
    ("Bucket 3", "Blur level 3", "sigma = 2.0", "~22,825 imgs"),
    ("Bucket 4", "Blur level 4", "sigma = 3.0", "~22,825 imgs"),
    ("Bucket 5", "Blur level 5", "sigma = 4.0", "~22,825 imgs"),
    ("Bucket 6", "Blur level 6", "sigma = 5.0", "~22,825 imgs"),
    ("Bucket 7", "Blur level 7", "sigma = 8.0", "~22,825 imgs"),
]

SCALE = 3  # 64 -> 192 (slightly smaller images so labels stand out)
img_size = 64 * SCALE
padding = 12
label_height = 120  # much more room for labels
cols = len(blur_sigmas)
rows = len(sample_faces)

width = cols * img_size + (cols + 1) * padding
height = rows * img_size + (rows + 1) * padding + label_height + 60  # extra for title

canvas = Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(canvas)

try:
    font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    font_bucket = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    font_role = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    font_detail = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
except:
    font_title = font_bucket = font_role = font_detail = ImageFont.load_default()

# Title
title = "CelebA Gaussian Blur Buckets (64x64) — 182,599 training images + 20K clean holdout"
bbox = draw.textbbox((0, 0), title, font=font_title)
tw = bbox[2] - bbox[0]
draw.text(((width - tw) // 2, 8), title, fill="black", font=font_title)

title_offset = 50

# Column headers
for col, (bname, role, sigma, size) in enumerate(bucket_info):
    x_center = padding + col * (img_size + padding) + img_size // 2
    
    # Bucket name (bold)
    bbox = draw.textbbox((0, 0), bname, font=font_bucket)
    draw.text((x_center - (bbox[2]-bbox[0])//2, title_offset + 5), bname, fill="black", font=font_bucket)
    
    # Role — highlight clean target in green
    color = "green" if col == 0 else "blue"
    bbox = draw.textbbox((0, 0), role, font=font_role)
    draw.text((x_center - (bbox[2]-bbox[0])//2, title_offset + 30), role, fill=color, font=font_role)
    
    # Sigma value
    bbox = draw.textbbox((0, 0), sigma, font=font_detail)
    draw.text((x_center - (bbox[2]-bbox[0])//2, title_offset + 52), sigma, fill="gray", font=font_detail)
    
    # Bucket size
    bbox = draw.textbbox((0, 0), size, font=font_detail)
    draw.text((x_center - (bbox[2]-bbox[0])//2, title_offset + 72), size, fill="gray", font=font_detail)

# Draw images
for row, fname in enumerate(sample_faces):
    clean_img = Image.open(os.path.join(bucket_dir, fname)).convert("RGB")
    for col, sigma in enumerate(blur_sigmas):
        if sigma == 0:
            img = clean_img.copy()
        else:
            img = clean_img.copy().filter(ImageFilter.GaussianBlur(radius=sigma))
        img_up = img.resize((img_size, img_size), Image.NEAREST)
        x = padding + col * (img_size + padding)
        y = title_offset + label_height + padding + row * (img_size + padding)
        # Green border for clean target column
        if col == 0:
            draw.rectangle([x-2, y-2, x+img_size+1, y+img_size+1], outline="green", width=3)
        canvas.paste(img_up, (x, y))

# Footer
footer = "Bucket 0 = always T=0 (clean target).  Buckets 1-7 = supplementary data with varying noise threshold T."
bbox = draw.textbbox((0, 0), footer, font=font_detail)
draw.text((padding, height - 25), footer, fill="black", font=font_detail)

out_path = f"{AMBIENT_BASE}/celeba_processed/blur_grid_presentation.png"
canvas.save(out_path)
print(f"Saved to {out_path} ({width}x{height})")

import shutil
afs_path = os.path.expanduser("~/generated_preview/blur_grid_presentation.png")
os.makedirs(os.path.dirname(afs_path), exist_ok=True)
shutil.copy(out_path, afs_path)
print(f"Copied to {afs_path}")
