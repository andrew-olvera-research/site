"""Explicit, audited retention cleanup, run inside the training container."""
import argparse
import json
from pathlib import Path
import time
import h5py

ROOT = Path('/workspace/outputs/checkpoints').resolve()
OUT = Path('/workspace/outputs/diagnostics/collection-next-20260929')

def retired_shards(directory):
    shards = sorted((directory/'dagger-replay').glob('round-*.h5'))
    keep = set(shards[-3:])
    for p in shards:
        with h5py.File(p, 'r') as h:
            if 'permanent' in h and int(h['permanent'].attrs.get('rows', 0)) > 0:
                keep.add(p)
    return [p for p in shards if p not in keep]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--run', help='Only a completed queue run')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    candidates, changed_runs, manifests = [], [], []
    directories = [ROOT/args.run] if args.run else sorted(p for p in ROOT.iterdir() if p.is_dir())
    # Never remove anything used by a currently running trainer.
    active = set()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try: argv = (proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except (OSError, ProcessLookupError): continue
        if any(a.endswith('train_privileged_racing.py') for a in argv) and '--config' in argv:
            cfg = json.loads(Path(argv[argv.index('--config')+1]).read_text())
            active.add(cfg['dagger']['run_name'])
    for directory in directories:
        if directory.name in active or (not args.run and directory.name.startswith('starscream-collection-next-')):
            continue
        if not directory.resolve().is_relative_to(ROOT) or directory.is_symlink():
            raise ValueError(f'Unsafe target {directory}')
        shards = retired_shards(directory)
        candidates.extend(shards)
        if shards: changed_runs.append(directory.name)
        # Preserve all ranked best/latest of pretrain and RL runs. Old ablation
        # duplicates can be reduced to the single highest-ranked best.
        manifest = directory/'top-k.json'
        if manifest.exists() and ('mini-slalom' in directory.name or 'collection-next-' in directory.name):
            data = json.loads(manifest.read_text())
            ranked = data.get('checkpoints', [])
            if ranked:
                ranked = sorted(ranked, key=lambda x:x['score'], reverse=data.get('mode','max')=='max')
                keep_name = ranked[0]['path']
                candidates.extend(directory/x['path'] for x in ranked[1:] if x['path'] != keep_name and (directory/x['path']).is_file())
                data.update(top_k=1, checkpoints=ranked[:1])
                manifests.append((manifest, data))
        for pattern in ('early-stopped*.pt', 'early_stopped*.pt', 'round-*.pt', 'epoch-*.pt', '*.tmp'):
            candidates.extend(directory.glob(pattern))
    candidates = sorted(set(candidates))
    for p in candidates:
        if p.is_symlink() or not p.resolve().is_relative_to(ROOT) or not p.is_file():
            raise ValueError(f'Unsafe deletion {p}')
        if p.name == 'latest.pt': raise AssertionError('latest must survive')
    report = dict(applied=args.apply, created=time.time(), active_skipped=sorted(active),
        bytes=sum(p.stat().st_size for p in candidates), files=[dict(path=str(p),bytes=p.stat().st_size) for p in candidates],
        replay_resume_incomplete_runs=changed_runs,
        note='Best/latest weights preserved. Retired replay keeps last three and permanent-containing shards; exact resume may be unavailable.')
    stem = 'cleanup-'+(args.run or 'old-runs')
    (OUT/f'{stem}-plan.json').write_text(json.dumps(report, indent=2)+'\n')
    if args.apply:
        for p in candidates: p.unlink()
        for p, data in manifests: p.write_text(json.dumps(data, indent=2)+'\n')
        for name in changed_runs:
            (ROOT/name/'REPLAY_PRUNED.json').write_text(json.dumps(dict(date='2026-09-29', retention='last three + permanent shards', exact_replay_resume=False, audit=str(OUT/f'{stem}-plan.json')), indent=2)+'\n')
        (OUT/f'{stem}-complete.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(applied=args.apply, files=len(candidates), gib=report['bytes']/2**30, affected_replay_runs=len(changed_runs), active_skipped=sorted(active))))

if __name__ == '__main__': main()
