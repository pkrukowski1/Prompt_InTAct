import argparse
import os
import random

ROOT = './datasets/image_list'
# domain order = task order of the loaders in datasets/*.py
LISTS = {
    'domainnet': ['real', 'quickdraw', 'painting', 'sketch', 'infograph', 'clipart'],
    'imagenet-r': ['art', 'cartoon', 'deviantart', 'graffiti', 'embroidery', 'graphic', 'origami', 'painting',
                   'misc', 'sticker', 'sculpture', 'sketch', 'tattoo', 'toy', 'videogame'],
    'imagenet-c': ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'contrast', 'elastic_transform',
                   'jpeg_compression', 'pixelate', 'spatter', 'speckle_noise', 'gaussian_noise', 'shot_noise',
                   'fog', 'frost', 'snow'],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fraction', type=float, default=0.1, help='fraction of each train list held out (default 0.1)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    assert 0 < args.fraction < 1

    for name, domains in LISTS.items():
        src, dst = os.path.join(ROOT, f'{name}_list'), os.path.join(ROOT, f'{name}_val')
        if not os.path.isdir(src):
            raise SystemExit(f'{src} not found (run from the repository root)')
        os.makedirs(dst, exist_ok=True)
        rng = random.Random(args.seed)
        n_tr = n_val = 0
        for d in domains:
            lines = [l for l in open(os.path.join(src, f'{d}_train.txt')).read().splitlines() if l.strip()]
            rng.shuffle(lines)
            k = max(1, int(round(args.fraction * len(lines))))
            val, tr = lines[:k], lines[k:]
            with open(os.path.join(dst, f'{d}_train.txt'), 'w') as f:
                f.write('\n'.join(tr) + '\n')
            with open(os.path.join(dst, f'{d}_test.txt'), 'w') as f:
                f.write('\n'.join(val) + '\n')
            n_tr += len(tr)
            n_val += len(val)
        print(f'{dst}: {len(domains)} domains, {n_tr} train / {n_val} validation images')


if __name__ == '__main__':
    main()
