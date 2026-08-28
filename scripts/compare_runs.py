import os
import re
import sys
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
RE_VAL = re.compile("'recall@20': ([0-9.]+)")
RE_NDCG = re.compile("'ndcg@20': ([0-9.]+)")
RE_GAP = re.compile("'gen_gap': (-?[0-9.]+)")
RE_EPOCH = re.compile('^Epoch (\\d+)/\\d+', re.M)
RE_BEST = re.compile('\\[Best\\] Recall@20=([0-9.]+) at epoch (\\d+)')
RE_STOP = re.compile('Stopping at epoch (\\d+)\\. Best Recall@20=([0-9.]+) at epoch (\\d+)')
RE_OMEGA_U = re.compile('omega_user: ([0-9.]+)')
RE_OMEGA_I = re.compile('omega_item: ([0-9.]+)')

def parse(path):
    with open(path, encoding='utf-8', errors='replace') as fh:
        text = fh.read()
    stop = RE_STOP.search(text)
    best = RE_BEST.findall(text)
    return {'epochs_started': [int(e) for e in RE_EPOCH.findall(text)], 'recall': [float(x) for x in RE_VAL.findall(text)], 'ndcg': [float(x) for x in RE_NDCG.findall(text)], 'gap': [float(x) for x in RE_GAP.findall(text)], 'omega_u': [float(x) for x in RE_OMEGA_U.findall(text)], 'omega_i': [float(x) for x in RE_OMEGA_I.findall(text)], 'best': (float(best[-1][0]), int(best[-1][1])) if best else None, 'stopped': (int(stop.group(1)), float(stop.group(2)), int(stop.group(3))) if stop else None}

def main():
    names = sys.argv[1:] or ['full', 'wo_llm', 'lowreg']
    runs = {}
    for n in names:
        p = os.path.join(LOG_DIR, f'{n}.log')
        if os.path.exists(p):
            runs[n] = parse(p)
        else:
            print(f'[skip] {p} not found')
    if not runs:
        sys.exit('no logs found')
    print('=' * 74)
    print('STATUS')
    print('=' * 74)
    for n, r in runs.items():
        cur = max(r['epochs_started']) if r['epochs_started'] else 0
        if r['stopped']:
            ep, best, bep = r['stopped']
            print(f'  {n:<8} DONE  stopped@{ep:<4} best val R@20={best:.4f} @ep{bep}')
        else:
            b = f"{r['best'][0]:.4f}@ep{r['best'][1]}" if r['best'] else '—'
            print(f'  {n:<8} running  epoch {cur:<5} best so far {b}')
    depth = min((len(r['recall']) for r in runs.values()))
    if depth >= 2:
        print()
        print('=' * 74)
        print(f'MATCHED-EPOCH COMPARISON (first {depth} evaluations, val Recall@20)')
        print('=' * 74)
        header = '  eval  ' + ''.join((f'{n:>12}' for n in runs))
        print(header)
        for i in range(depth):
            row = f'  {i + 1:<6}'
            vals = [runs[n]['recall'][i] for n in runs]
            top = max(vals)
            for v in vals:
                mark = '*' if v == top else ' '
                row += f'{v:>11.4f}{mark}'
            print(row)
        print('  (* = best at that evaluation; this is the fair comparison)')
        print()
        print('  gen_gap at the same evaluations (val_ce - train_ce):')
        for n, r in runs.items():
            g = r['gap'][:depth]
            if g:
                print(f'    {n:<8} ' + ' '.join((f'{v:6.2f}' for v in g[-6:])))
        print('    NOTE: the pre-fix run reached gen_gap ~8 late in training. If the')
        print('    fixed runs stay well below that, the gap was largely an artifact')
        print('    of the biased DropEdge operator rather than genuine overfitting.')
    if any((r['omega_u'] for r in runs.values())):
        print()
        print('=' * 74)
        print('LEARNED LLM SCALE (init: user 0.3 / item 0.2)')
        print('=' * 74)
        for n, r in runs.items():
            if r['omega_u']:
                print(f"  {n:<8} omega_user {r['omega_u'][-1]:.4f}   omega_item {(r['omega_i'][-1] if r['omega_i'] else float('nan')):.4f}")
    done = [n for n, r in runs.items() if r['stopped']]
    print()
    if len(done) == len(runs):
        print('All runs finished. Next:  bash run_ablations.sh eval')
        print('Then update the paper from logs/eval_*.log (test-set numbers).')
    else:
        print(f'{len(done)}/{len(runs)} finished. Re-run this script later.')
if __name__ == '__main__':
    main()
