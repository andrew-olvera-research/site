"""Explicit file-only deletion manifest; preserve useful policy and codec anchors."""
import argparse,json
from pathlib import Path

ROOT=Path('/workspace/outputs')
FULL={
 'starscream-v5.5-swift-c-modern-h3-r8',
 'starscream-v6.0-swift-directional-tube-dagger-r80',
 'starscream-v6.0-swift-exact-control-dagger-r80',
 'starscream-v6.9-policy-compass-ppo-12m',
 'starscream-v6.11-policy-transition-ppo-3m',
}

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
    plan=ROOT/'audits/artifact-prune-20260908.json'
    if not a.execute:
        keep=set()
        for folder in (ROOT/'checkpoints').iterdir():
            if not folder.is_dir():continue
            if folder.name in FULL:keep.update(x.resolve() for x in folder.glob('*.pt'));continue
            # One selected best from every completed ManiPPO v6 branch.
            if folder.name.startswith('starscream-v6.') and 'aborted' not in folder.name:
                top=folder/'top-k.json'
                if top.exists():
                    data=json.loads(top.read_text());rows=data.get('checkpoints',[])
                    if rows:
                        best=sorted(rows,key=lambda x:x['score'],reverse=data.get('mode','max')=='max')[0]
                        keep.add((folder/best['path']).resolve())
        candidates=list((ROOT/'checkpoints').rglob('*.pt'))+list((ROOT/'models/retained').rglob('*.pt'))
        # Retain scientific codec arms and all fitted navigation tools; drop smoke weights only.
        candidates.extend(x for x in (ROOT/'course-model').rglob('*.pt') if any('smoke' in k for k in x.relative_to(ROOT/'course-model').parts))
        rows=[]
        for x in sorted(set(candidates)):
            resolved=x.resolve()
            if x.is_symlink() or not resolved.is_relative_to(ROOT) or resolved in keep:continue
            rows.append(dict(path=str(resolved),bytes=x.stat().st_size,mtime_ns=x.stat().st_mtime_ns))
        report=dict(kept=sorted(map(str,keep)),delete=rows,total_bytes=sum(x['bytes'] for x in rows),policy='user-authorized checkpoint pruning; preserve logs/configs, full selected anchors, best per v6 branch, all non-smoke codecs/navigation tools')
        plan.write_text(json.dumps(report,indent=2));print('files',len(rows),'GiB',report['total_bytes']/2**30,'kept',len(keep));return
    report=json.loads(plan.read_text());removed=[]
    for row in report['delete']:
        x=Path(row['path'])
        if x.is_symlink() or not x.resolve().is_relative_to(ROOT) or x.suffix!='.pt':raise RuntimeError('Unsafe target')
        st=x.stat()
        if st.st_size!=row['bytes'] or st.st_mtime_ns!=row['mtime_ns']:raise RuntimeError('Artifact changed since inventory')
    for row in report['delete']:
        Path(row['path']).unlink();removed.append(row['path'])
    report.update(executed=True,removed=removed);plan.write_text(json.dumps(report,indent=2));print('deleted_bytes',report['total_bytes'],'files',len(removed))

if __name__=='__main__':main()
