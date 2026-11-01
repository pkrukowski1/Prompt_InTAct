import argparse
import importlib
import json
import os
import statistics
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets import get_dataset
from models import get_model
from utils.conf import set_random_seed

GIB = 1024 ** 3


def parse(argv=None):
    cost = argparse.ArgumentParser(add_help=False)
    cost.add_argument('--cost_tasks', type=int, default=2, help='number of tasks to run (at least 2 for InTAct)')
    cost.add_argument('--cost_epochs', type=int, default=2, help='epochs per task; the time per batch does not depend on it')
    cost.add_argument('--cost_label', type=str, default='', help='name of the configuration in the output')
    cost.add_argument('--cost_out', type=str, default=None, help='JSON file for the measurements')
    copts, rest = cost.parse_known_args(argv)
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument('--model', type=str, default='kaprompt_vit')
    name = pre.parse_known_args(rest)[0].model
    args = importlib.import_module('models.' + name).get_parser().parse_args(rest)
    return copts, args


def count_parameters(model):
    """All parameters of the model (backbone, head, prompt pool, general prompts) and those the optimizer updates."""
    tensors = {id(p): p for p in model.net.parameters()}
    for layer in list(model.pool.key_list) + list(model.pool.prompt_list):
        for p in layer:
            tensors[id(p)] = p
    for p in model.general_prompt:
        tensors[id(p)] = p
    trainable = {id(p): p for g in model.opt.param_groups for p in g['params'] if p.requires_grad}
    return sum(p.numel() for p in tensors.values()), sum(p.numel() for p in trainable.values())


def reset_memory(cuda):
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def peak_memory(cuda):
    if not cuda:
        return None, None
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / GIB, torch.cuda.max_memory_reserved() / GIB


def train_task(model, dataset, args, t, epochs, device, cuda):
    model.net.train()
    loader = dataset.get_data_loaders(task_id=t)
    model.init_opt(args, t, loader)
    total, trainable = count_parameters(model)
    reset_memory(cuda)
    times = []
    for _ in range(epochs):
        start = time.perf_counter()
        for data in loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            model.observe(inputs, labels, dataset, t)
            if cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
            if args.adapt_ema_c:
                model.cal_g_change()
            if args.use_ema_c and args.update_ema_c == 'batch':
                model.ema_c.update()
            if args.use_ema and args.update_ema_g == 'batch':
                model.update_ema_g()
            start = time.perf_counter()
        if args.use_ema_c and args.update_ema_c == 'epoch':
            model.ema_c.update()
        if args.use_ema and args.update_ema_g == 'epoch':
            model.update_ema_g()
    if args.use_ema_c and args.update_ema_c == 'task':
        model.ema_c.update()
    if args.use_ema and args.update_ema_g == 'task':
        model.update_ema_g()
    alloc, reserved = peak_memory(cuda)
    return dict(total_params=total, trainable_params=trainable, train_batches=len(times),
                train_time_mean=statistics.mean(times), train_time_median=statistics.median(times),
                train_mem_allocated_gib=alloc, train_mem_reserved_gib=reserved)


def evaluate_task(model, dataset, args, t, device, cuda):
    """evaluate_cprompt with timing; like the trainer, it first collects the current task's training activations
    that define the next InTAct hypercube (not timed)."""
    status = model.net.training
    model.net.eval()
    ema = (args.use_ema or args.use_ema_c) and not (args.ema_after_domain_2 and t == 0)
    if ema:
        model.ema_before_eval(t)
    if getattr(model, 'interval_penalization', None) is not None:
        model.collect_interval_activations(dataset.get_data_loaders(task_id=t, eval=True))
    reset_memory(cuda)
    times, accs = [], []
    with torch.no_grad():
        for loader in dataset.get_test_dataloaders(t):
            correct = total = 0
            start = time.perf_counter()
            for data in loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs, _ = model.forward_model(inputs)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.shape[0]
                if cuda:
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
                start = time.perf_counter()
            accs.append(100.0 * correct / total)
    alloc, reserved = peak_memory(cuda)
    model.net.train(status)
    if ema:
        model.ema_after_eval()
    return dict(eval_batches=len(times), eval_time_mean=statistics.mean(times), eval_time_median=statistics.median(times),
                eval_mem_allocated_gib=alloc, eval_mem_reserved_gib=reserved, accuracy=sum(accs) / len(accs))


def main(argv=None, dataset=None):
    copts, args = parse(argv)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.seed is not None:
        set_random_seed(args.seed)
    dataset = dataset or get_dataset(args)
    model = get_model(args, dataset.get_backbone(), dataset.get_loss(), dataset.get_transform())
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    model.device = device
    model.net.to(device)

    n_tasks = min(copts.cost_tasks, dataset.N_TASKS)
    tasks = []
    for t in range(n_tasks):
        rec = dict(task=t + 1)
        rec.update(train_task(model, dataset, args, t, copts.cost_epochs, device, cuda))
        if args.ema_after_domain_2 and t == 0:
            model.re_init_ema()
        rec.update(evaluate_task(model, dataset, args, t, device, cuda))
        if t == 0 and args.freeze_g_prompt:
            model.set_g_prompt_not_update()
        tasks.append(rec)
        print(json.dumps(rec), flush=True)

    later = tasks[1:] or tasks
    mean = lambda key, rows: sum(r[key] for r in rows) / len(rows)
    out = dict(label=copts.cost_label, gpu=torch.cuda.get_device_name() if cuda else 'cpu',
               batch_size=args.batch_size, intact=bool(args.use_interval_activation), tasks=tasks,
               total_params=max(r['total_params'] for r in tasks), trainable_params=max(r['trainable_params'] for r in tasks),
               train_time_per_batch=mean('train_time_mean', later), eval_time_per_batch=mean('eval_time_mean', later))
    if cuda:
        out['train_mem_reserved_gib'] = max(r['train_mem_reserved_gib'] for r in tasks)
        out['train_mem_allocated_gib'] = max(r['train_mem_allocated_gib'] for r in tasks)
        out['eval_mem_reserved_gib'] = max(r['eval_mem_reserved_gib'] for r in tasks)
    print(f"\n[{copts.cost_label or 'cost'}] {out['gpu']}, batch size {args.batch_size} (test {4 * args.batch_size}), "
          f"InTAct {'on' if out['intact'] else 'off'}")
    print(f"  total parameters      {out['total_params']:,}")
    print(f"  trainable parameters  {out['trainable_params']:,}")
    if cuda:
        print(f"  peak GPU memory       {out['train_mem_reserved_gib']:.2f} GiB reserved / "
              f"{out['train_mem_allocated_gib']:.2f} GiB allocated (training), "
              f"{out['eval_mem_reserved_gib']:.2f} GiB reserved (evaluation)")
    out['train_time_per_batch_median'] = statistics.median(r['train_time_median'] for r in later)
    which = 'task 1' if n_tasks == 1 else ('task 2' if n_tasks == 2 else f'tasks 2-{n_tasks}')
    print(f"  time per batch        {out['train_time_per_batch']:.3f} s training (median "
          f"{out['train_time_per_batch_median']:.3f} s), {out['eval_time_per_batch']:.3f} s evaluation ({which})")
    if copts.cost_out:
        os.makedirs(os.path.dirname(os.path.abspath(copts.cost_out)), exist_ok=True)
        with open(copts.cost_out, 'w') as f:
            json.dump(out, f, indent=1)
        print(f'  wrote {copts.cost_out}')
    return out


if __name__ == '__main__':
    main()
