#!/usr/bin/env python
import argparse
import math
import os
import re
import sys

N_TASKS = {'DomainNet': 6, 'ImageNet-R': 15, 'ImageNet-C': 15, 'ImageNet-Mix': 30}
SCRIPT = {'DomainNet': 'domainnet', 'ImageNet-R': 'imagenet_r', 'ImageNet-C': 'imagenet_c', 'ImageNet-Mix': 'imagenet_cr'}
NAME = re.compile(r'^(?P<variant>intact_ema|intact_noema)_var(?P<var>[^_]+)_drift(?P<drift>[^_]+)'
                  r'_feat(?P<feat>[^_]+)_align(?P<align>[^_]+)$')
FLOAT = re.compile(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?')
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
                         done=len(aas), seeds=len(seeds),
                         unfinished=[f'seed {s}: {r[0]}/{n_tasks}' for s, r in runs.items() if r[1] is None]))
    if not rows:
        sys.exit(f'no runs in {folder}')

    ka = next((r for r in rows if r['name'] == 'ka' and r['done'] and not r['unfinished']), None)
    rows.sort(key=lambda r: (r['aa'] is None, -(r['aa'] or 0)))
    print(f'{dataset} | {"validation split" if split == "val" else "official test lists"} | {n_tasks} tasks | {folder}')
    w = max(len('configuration'), max(len(r['name']) for r in rows))
    hdr = f"{'#':>3}  {'configuration':<{w}} {'AA':>6} {'std':>5} {'F':>6} {'vs ka':>6}  seeds  unfinished"
    print(hdr + '\n' + '-' * len(hdr))
    shown = 0
    for i, r in enumerate(rows, 1):
        if r['params']:
            shown += 1
            if args.top and shown > args.top:
                continue
        gain = '-' if (ka is None or r['aa'] is None or r is ka) else f"{r['aa'] - ka['aa']:+.2f}"
        print(f"{i:>3}  {r['name']:<{w}} {fmt(r['aa']):>6} {fmt(r['sd']):>5} {fmt(r['f']):>6} {gain:>6}  "
              f"{r['done']}/{r['seeds']:<4} {', '.join(r['unfinished'])}")

    with open(os.path.join(folder, 'summary.tsv'), 'w') as f:
        f.write('configuration\tAA\tstd\tF\tseeds_finished\tseeds\n')
        for r in rows:
            f.write(f"{r['name']}\t{fmt(r['aa'])}\t{fmt(r['sd'])}\t{fmt(r['f'])}\t{r['done']}\t{r['seeds']}\n")
    print(f"\nwrote {os.path.join(folder, 'summary.tsv')}")

    best = next((r for r in rows if r['params'] and r['done'] and not r['unfinished']), None)
    if split == 'val' and best:
        p = best['params']
        print(f"\nbest configuration on the validation split: {best['name']} (AA {best['aa']:.2f})")
        print(f'put this line into experiments/run_{SCRIPT[dataset]}.sh:')
        print(f'VAR="{p["var"]}"; DRIFT="{p["drift"]}"; FEAT="{p["feat"]}"; ALIGN="{p["align"]}"; '
              f'EMA={1 if p["variant"] == "intact_ema" else 0}')


if __name__ == '__main__':
    main()
