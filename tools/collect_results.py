#!/usr/bin/env python
import argparse
import math
import os
import re
import sys

N_TASKS = {'DomainNet': 6, 'ImageNet-R': 15, 'ImageNet-C': 15, 'ImageNet-Mix': 30}
GRID_SEED = '3407'   # seed of the grid scripts; the best configuration is chosen by this run
SCRIPT = {'DomainNet': 'domainnet', 'ImageNet-R': 'imagenet_r', 'ImageNet-C': 'imagenet_c', 'ImageNet-Mix': 'imagenet_cr'}
NAME = re.compile(r'^(?P<variant>intact_ema_raw|intact_ema|intact_noema)_var(?P<var>[^_]+)_drift(?P<drift>[^_]+)'
                  r'_feat(?P<feat>[^_]+)_align(?P<align>[^_]+)$')
FLOAT = re.compile(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?')
GRID_LINE = re.compile(r'^(VARS|DRIFTS|FEATS|ALIGNS)="([^"]*)"', re.M)
GRID_FILE = {'intact_ema': '', 'intact_ema_raw': '_raw', 'intact_noema': '_noema'}
ACC_LINE = re.compile(r'Accuracy for \d+ task\(s\):\s*([-+]?[\d.]+)')


def read_task_file(path):
    """(per-domain accuracies, printed average accuracy) after one task, from results/<t>.txt."""
    with open(path) as f:
        lines = f.read().splitlines()
    line = lines[1]
    if '([' in line:             # (list, list) -> first list, as in cal_mean_std_af.py
        line = line.split(']')[0]
    accs = [float(x) for x in FLOAT.findall(line)]
    m = next((ACC_LINE.search(l) for l in lines[2:] if ACC_LINE.search(l)), None)
    return accs, (float(m.group(1)) if m else round(sum(accs) / len(accs), 2))


def read_run(run_dir, n_tasks):
    """-> (tasks finished, final AA, forgetting) of one seed folder; AA and F are None if unfinished."""
    stages, printed = [], []
    for t in range(n_tasks):
        p = os.path.join(run_dir, 'results', f'{t}.txt')
        if not os.path.isfile(p):
            break
        accs, avg = read_task_file(p)
        stages.append(accs)
        printed.append(avg)
    if len(stages) < n_tasks:
        return len(stages), None, None
    final = stages[-1]
    forget = [max(stages[s][d] for s in range(d, n_tasks - 1)) - final[d] for d in range(n_tasks - 1)]
    return n_tasks, printed[-1], round(sum(forget) / len(forget), 2)


def grid_configs(dataset, variant):
    # (var, drift, feat, align) of experiments/grid_<dataset><suffix>.sh; None if the script is not next to tools/
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'experiments',
                        f'grid_{SCRIPT[dataset]}{GRID_FILE[variant]}.sh')
    if not os.path.isfile(path):
        return None
    v = dict(GRID_LINE.findall(open(path).read()))
    return {(a, b, c, d) for a in v['VARS'].split() for b in v['DRIFTS'].split() for c in v['FEATS'].split()
            for d in v['ALIGNS'].split()}


def mean_std(xs):
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def fmt(x):
    return '-' if x is None else f'{x:.2f}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folder', help='<ROOT>/<Dataset>/val or <ROOT>/<Dataset>/test')
    ap.add_argument('--top', type=int, default=0, help='show only the best N InTAct configurations')
    args = ap.parse_args()

    folder = os.path.abspath(args.folder)
    split, dataset = os.path.basename(folder), os.path.basename(os.path.dirname(folder))
    if dataset not in N_TASKS:
        sys.exit(f'expected <ROOT>/<Dataset>/<val|test> with Dataset in {sorted(N_TASKS)}, got {args.folder}')
    n_tasks = N_TASKS[dataset]

    rows = []
    for name in sorted(os.listdir(folder)):
        cdir = os.path.join(folder, name)
        m = NAME.match(name)
        if not os.path.isdir(cdir) or not (m or name in ('ka', 'ka_noema')):
            continue
        seeds = sorted((d[len('seed-'):] for d in os.listdir(cdir) if d.startswith('seed-')), key=lambda s: int(s))
        runs = {s: read_run(os.path.join(cdir, f'seed-{s}'), n_tasks) for s in seeds}
        aas = [r[1] for r in runs.values() if r[1] is not None]
        fs = [r[2] for r in runs.values() if r[2] is not None]
        aa, sd = mean_std(aas)
        rows.append(dict(name=name, params=m.groupdict() if m else None, aa=aa, sd=sd, f=mean_std(fs)[0],
                         grid=runs[GRID_SEED][1] if GRID_SEED in runs else None,
                         done=len(aas), seeds=len(seeds),
                         unfinished=[f'seed {s}: {r[0]}/{n_tasks}' for s, r in runs.items() if r[1] is None]))
    if not rows:
        sys.exit(f'no runs in {folder}')

    base = {b: next((r for r in rows if r['name'] == b and r['done'] and not r['unfinished']), None)
            for b in ('ka', 'ka_noema')}
    grids = {v: grid_configs(dataset, v) for v in GRID_FILE}
    for r in rows:
        p = r['params']
        r['in_grid'] = not p or grids[p['variant']] is None or \
            (p['var'], p['drift'], p['feat'], p['align']) in grids[p['variant']]
        r['label'] = r['name'] if r['in_grid'] else r['name'] + ' *'
    rows.sort(key=lambda r: (r['aa'] is None, -(r['aa'] or 0)))
    print(f'{dataset} | {"validation split" if split == "val" else "official test lists"} | {n_tasks} tasks | {folder}')
    w = max(len('configuration'), max(len(r['label']) for r in rows))
    hdr = (f"{'#':>3}  {'configuration':<{w}} {'AA':>6} {'std':>5} {'F':>6} {'AA ' + GRID_SEED:>7} {'vs ka':>6} {'vs ka_noema':>11}"
           f"  seeds  unfinished")
    print(hdr + '\n' + '-' * len(hdr))
    shown = 0
    for i, r in enumerate(rows, 1):
        if r['params']:
            shown += 1
            if args.top and shown > args.top:
                continue
        gain = {b: '-' if (base[b] is None or r['aa'] is None or r is base[b]) else f"{r['aa'] - base[b]['aa']:+.2f}"
                for b in base}
        print(f"{i:>3}  {r['label']:<{w}} {fmt(r['aa']):>6} {fmt(r['sd']):>5} {fmt(r['f']):>6} {fmt(r['grid']):>7} {gain['ka']:>6} "
              f"{gain['ka_noema']:>11}  {r['done']}/{r['seeds']:<4} {', '.join(r['unfinished'])}")

    with open(os.path.join(folder, 'summary.tsv'), 'w') as f:
        f.write(f'configuration\tAA\tstd\tF\tseeds_finished\tseeds\tAA_seed{GRID_SEED}\n')
        for r in rows:
            f.write(f"{r['name']}\t{fmt(r['aa'])}\t{fmt(r['sd'])}\t{fmt(r['f'])}\t{r['done']}\t{r['seeds']}\t{fmt(r['grid'])}\n")
    if not all(r['in_grid'] for r in rows):
        print(f'* not in the current grid (experiments/grid_{SCRIPT[dataset]}*.sh), so never chosen as the best configuration')
    print(f"\nwrote {os.path.join(folder, 'summary.tsv')}")

    best = {}
    for variant, suffix, tail in (('intact_ema', '', 'EMA=1; ANCHOR=ema'), ('intact_ema_raw', '', 'EMA=1; ANCHOR=raw'),
                                  ('intact_noema', '_noema', 'EMA=0')):
        cands = [r for r in rows if r['params'] and r['params']['variant'] == variant and r['grid'] is not None
                 and r['in_grid']]
        if cands:
            b = best[variant] = max(cands, key=lambda r: r['grid'])
            p = b['params']
            print(f"\nbest {variant} configuration by its seed-{GRID_SEED} run ({len(cands)} configurations finished): "
                  f"{b['name']} (AA {b['grid']:.2f})")
            print(f'put this line into experiments/run_{SCRIPT[dataset]}{suffix}.sh:')
            print(f'VAR="{p["var"]}"; DRIFT="{p["drift"]}"; FEAT="{p["feat"]}"; ALIGN="{p["align"]}"; {tail}')
    if 'intact_ema' in best and 'intact_ema_raw' in best:
        e, r = best['intact_ema']['grid'], best['intact_ema_raw']['grid']
        print(f"\nwith KA-Prompt's EMA classifier (seed {GRID_SEED}): EMA anchor {e:.2f}, raw anchor {r:.2f} "
              f"-> use the {'raw' if r > e else 'EMA'} anchor line in experiments/run_{SCRIPT[dataset]}.sh")


if __name__ == '__main__':
    main()
