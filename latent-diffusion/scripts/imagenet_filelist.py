"""Relative paths of all ImageNet train images in EDM2 dataset_tool order (sorted full paths), one per line.

Index i of this list = index i of the converted dataset (dataset_tool convert sorts the same way and the archive
names img%08d keep that order), so per-image scores computed from the list line up with the training data.
Usage: python imagenet_filelist.py <imagenet_train_dir> <out.txt>
"""
import os, sys
from PIL import Image

root, out = sys.argv[1], sys.argv[2]
ext = set(Image.registered_extensions())          # dataset_tool's is_image_ext uses PIL's extension table
files = sorted(os.path.join(root, d, f) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
               for f in os.listdir(os.path.join(root, d)) if os.path.splitext(f)[1].lower() in ext)
with open(out, 'w') as fh:
    fh.writelines(os.path.relpath(f, root) + '\n' for f in files)
print(len(files), 'images,', len({f.split(os.sep)[-2] for f in files}), 'classes')
