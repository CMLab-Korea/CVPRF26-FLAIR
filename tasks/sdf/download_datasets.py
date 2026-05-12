"""
Download point clouds (.xyz) for training and Stanford watertight meshes (.ply)
for chamfer/IoU evaluation.

Usage:
    python download_datasets.py          # BACON-distributed xyz (training only)
    python download_datasets.py --eval   # + Stanford .ply, then regenerate
                                         #   gt_<scene>.xyz from the .ply so
                                         #   training and eval share a frame
"""

import argparse
import gzip
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request

import gdown

DATA_DIR = './data'

XYZ_FILES = {
    'gt_armadillo.xyz': '1xBo6OCGmyWi0qD74EZW4lc45Gs4HXjWw',
    'gt_dragon.xyz':    '1Pm3WHUvJiMJEKUnnhMjB6mUAnR9qhnxm',
    'gt_lucy.xyz':      '1wE24AZtXS8jbIIc-amYeEUtlxN8dFYCo',
    'gt_thai.xyz':      '1OVw0JNA-NZtDXVmkf57erqwqDjqmF5Mc',
}

# Stanford repo originals (Lucy ~500 MB skipped from default eval download).
PLY_SOURCES = {
    'Armadillo.ply':        ('https://graphics.stanford.edu/pub/3Dscanrep/armadillo/Armadillo.ply.gz',       'gz'),
    'dragon_vrip.ply':      ('https://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz',       'tar:dragon_recon/dragon_vrip.ply'),
    'xyzrgb_statuette.ply': ('https://graphics.stanford.edu/data/3Dscanrep/xyzrgb/xyzrgb_statuette.ply.gz',  'gz'),
}

# Each eval-supported scene: which .ply backs it, and the gt_<scene>.xyz to regenerate.
RESAMPLE_TARGETS = [
    ('armadillo', 'Armadillo.ply'),
    ('dragon',    'dragon_vrip.ply'),
]


def fetch_ply(name, url, kind):
    out = os.path.join(DATA_DIR, name)
    if os.path.exists(out):
        print(f'[skip] {out}')
        return
    print(f'  -> {url}')
    tmp = os.path.join(DATA_DIR, '_tmp_' + os.path.basename(url))
    urllib.request.urlretrieve(url, tmp)
    if kind == 'gz':
        with gzip.open(tmp, 'rb') as f_in, open(out, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    elif kind.startswith('tar:'):
        member = kind.split(':', 1)[1]
        with tarfile.open(tmp, 'r:gz') as tf:
            tf.extract(member, path=DATA_DIR)
        shutil.move(os.path.join(DATA_DIR, member), out)
    else:
        raise ValueError(kind)
    os.remove(tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval', action='store_true',
                    help='also download Stanford .ply meshes (for chamfer/IoU eval)')
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    print('Downloading SDF point clouds (.xyz)')
    for fname, gid in XYZ_FILES.items():
        out = os.path.join(DATA_DIR, fname)
        if os.path.exists(out):
            print(f'[skip] {out}')
            continue
        gdown.download(f'https://drive.google.com/uc?id={gid}', out, quiet=False)

    if args.eval:
        print('\nDownloading Stanford ground-truth meshes (.ply)')
        for name, (url, kind) in PLY_SOURCES.items():
            fetch_ply(name, url, kind)

        print('\nRegenerating training xyz from Stanford .ply (aligned for eval)')
        sample_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'sample_xyz_from_ply.py')
        for scene, ply in RESAMPLE_TARGETS:
            subprocess.check_call([
                sys.executable, sample_script,
                '--ply', os.path.join(DATA_DIR, ply),
                '--out', os.path.join(DATA_DIR, f'gt_{scene}.xyz'),
            ])

    print('Done.')


if __name__ == '__main__':
    main()
