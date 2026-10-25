import argparse
import hashlib
import sys

import torch

TIMM_REFERENCES = [
    'vit_base_patch16_224.augreg_in21k',          # IN-21k only (AugReg)              -> what KA-Prompt's arch name suggests
    'vit_base_patch16_224.orig_in21k',            # IN-21k only (original ViT, = HF google/vit-base-patch16-224-in21k)
    'vit_base_patch16_224.augreg_in21k_ft_in1k',  # IN-21k -> IN-1k   (timm<=0.6 default of vit_base_patch16_224)
    'vit_base_patch16_224.augreg2_in21k_ft_in1k', # IN-21k -> IN-1k   (timm>=0.9 default of vit_base_patch16_224)
    'vit_base_patch16_224.orig_in21k_ft_in1k',    # IN-21k -> IN-1k   (original ViT)
]
PROBE_KEYS = ['cls_token', 'pos_embed', 'patch_embed.proj.weight', 'blocks.0.attn.qkv.weight',
              'blocks.11.mlp.fc2.weight', 'norm.weight']


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def load_state_dict(path):
    sd = torch.load(path, map_location='cpu')
    for k in ('state_dict', 'model'):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
    if not isinstance(sd, dict):
        raise SystemExit(f'{path} does not contain a state_dict (got {type(sd)})')
    return sd


def key_format(sd):
    keys = list(sd.keys())
    if any(k.startswith(('embeddings.', 'vit.embeddings.', 'encoder.layer.', 'vit.encoder.layer.')) for k in keys):
        return 'hf-transformers'
    if 'patch_embed.proj.weight' in sd and 'blocks.0.attn.qkv.weight' in sd:
        return 'timm'
    return 'unknown'


def hf_to_timm(sd, depth=12):
    """Convert a transformers ViTModel / ViTForImageClassification state_dict to timm VisionTransformer keys."""
    sd = {k[len('vit.'):] if k.startswith('vit.') else k: v for k, v in sd.items()}
    out = {
        'cls_token': sd['embeddings.cls_token'],
        'pos_embed': sd['embeddings.position_embeddings'],
        'patch_embed.proj.weight': sd['embeddings.patch_embeddings.projection.weight'],
        'patch_embed.proj.bias': sd['embeddings.patch_embeddings.projection.bias'],
        'norm.weight': sd['layernorm.weight'],
        'norm.bias': sd['layernorm.bias'],
    }
    for i in range(depth):
        p = f'encoder.layer.{i}.'
        q = f'blocks.{i}.'
        out[q + 'norm1.weight'] = sd[p + 'layernorm_before.weight']
        out[q + 'norm1.bias'] = sd[p + 'layernorm_before.bias']
        out[q + 'attn.qkv.weight'] = torch.cat([sd[p + f'attention.attention.{n}.weight'] for n in ('query', 'key', 'value')], 0)
        out[q + 'attn.qkv.bias'] = torch.cat([sd[p + f'attention.attention.{n}.bias'] for n in ('query', 'key', 'value')], 0)
        out[q + 'attn.proj.weight'] = sd[p + 'attention.output.dense.weight']
        out[q + 'attn.proj.bias'] = sd[p + 'attention.output.dense.bias']
        out[q + 'norm2.weight'] = sd[p + 'layernorm_after.weight']
        out[q + 'norm2.bias'] = sd[p + 'layernorm_after.bias']
        out[q + 'mlp.fc1.weight'] = sd[p + 'intermediate.dense.weight']
        out[q + 'mlp.fc1.bias'] = sd[p + 'intermediate.dense.bias']
        out[q + 'mlp.fc2.weight'] = sd[p + 'output.dense.weight']
        out[q + 'mlp.fc2.bias'] = sd[p + 'output.dense.bias']
    return out   # pooler (pre-logits) and the zeroed 21k head of the original release are dropped


def ka_style_load_test(sd):
    """Replicates KA-Prompt's datasets/*.py:get_backbone() loading and reports what was actually loaded."""
    import timm
    model = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=False, num_classes=21843)
    res = model.load_state_dict(sd, strict=False)
    n_model = len(model.state_dict())
    n_loaded = n_model - len(res.missing_keys)
    print(f'[KA-style load] timm {timm.__version__}: {n_loaded}/{n_model} tensors loaded, '
          f'{len(res.missing_keys)} missing, {len(res.unexpected_keys)} unexpected')
    if res.missing_keys:
        print('   missing (first 5):', res.missing_keys[:5])
    if res.unexpected_keys:
        print('   unexpected (first 5):', res.unexpected_keys[:5])
    backbone_missing = [k for k in res.missing_keys if not k.startswith('head')]
    if backbone_missing:
        print('   !!! backbone weights were NOT loaded -> KA-Prompt would run on a randomly initialised ViT')
    else:
        print('   OK: every backbone tensor was loaded')


def compare_with_timm(sd):
    import timm
    available = set(timm.list_pretrained('vit_base_patch16_224*'))
    best = None
    for name in TIMM_REFERENCES:
        if name not in available:
            print(f'[compare] {name}: not available in timm {timm.__version__}')
            continue
        try:
            ref = timm.create_model(name, pretrained=True).state_dict()
        except Exception as e:  # no internet / no HF access
            print(f'[compare] {name}: could not download ({type(e).__name__})')
            continue
        diffs = []
        for k in PROBE_KEYS:
            if k in sd and k in ref and sd[k].shape == ref[k].shape:
                diffs.append((sd[k].float() - ref[k].float()).abs().max().item())
            else:
                diffs.append(float('inf'))
        worst = max(diffs)
        print(f'[compare] {name}: max |diff| over probe tensors = {worst:.3e}')
        if best is None or worst < best[1]:
            best = (name, worst)
    if best is not None:
        verdict = 'IDENTICAL' if best[1] < 1e-5 else 'closest (not identical)'
        print(f'[compare] => {verdict}: {best[0]}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--compare', action='store_true', help='compare with timm reference checkpoints (needs HF hub)')
    ap.add_argument('--convert-hf-to-timm', metavar='OUT', default=None)
    args = ap.parse_args()

    sd = load_state_dict(args.path)
    fmt = key_format(sd)
    print(f'file   : {args.path}\nsha256 : {sha256(args.path)}\nkeys   : {len(sd)}  format: {fmt}')
    if 'head.weight' in sd:
        hw = sd['head.weight']
        print(f'head   : {tuple(hw.shape)}  (21843 -> IN-21k head, 1000 -> IN-1k head)  |head| = {hw.abs().sum().item():.3e}')
    if any(k.startswith('pre_logits') for k in sd):
        print('note   : contains pre_logits (original 2020 IN-21k release)')

    print('-- loading the file exactly as the KA-Prompt code does:')
    ka_style_load_test(sd)
    if fmt == 'hf-transformers':
        print('!!! Hugging Face key names: KA-Prompt\'s load_state_dict(..., strict=False) loads no backbone weight.')
        sd = hf_to_timm(sd)
        print('-- after converting HF -> timm key names:')
        ka_style_load_test(sd)
        if args.convert_hf_to_timm:
            torch.save(sd, args.convert_hf_to_timm)
            print(f'converted timm-format state_dict written to {args.convert_hf_to_timm}')
    if args.compare:
        compare_with_timm(sd)


if __name__ == '__main__':
    sys.exit(main())
