import os


def list_root(official_root: str) -> str:
    split = os.environ.get('KA_SPLIT', 'official')
    if split == 'official':
        return official_root
    if split != 'val':
        raise ValueError(f"KA_SPLIT={split!r}: expected 'official' or 'val'")
    root = official_root.rstrip('/')
    if not root.endswith('_list'):
        raise ValueError(f'unexpected image-list folder {official_root!r}')
    val_root = root[:-len('_list')] + '_val'
    if not os.path.isdir(val_root):
        raise FileNotFoundError(f'{val_root} not found: run `python tools/make_val_lists.py` from the repository root')
    return val_root
