"""Stream a debug corpus, report independent/reference coverage, export examples."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from collections import Counter
import h5py
import numpy as np
from scipy.spatial.distance import cdist
from starscream.course_model.schema import pack,unpack,static_reasons,VERSION
from starscream.course_model.generation import proposal,FAMILIES,PRIMITIVES
from starscream.env.tracks import load_track
from starscream.env.procedural_tracks import save_track_yaml

REFS=('swift_champion_2022_exact','multigp_cdra_2026_reconstructed','a2rl_s2_2026_source_consistent_v2')
DESCRIPTORS=('gate_count','polyline_length_m','extent_x_m','extent_y_m','vertical_range_m',
    'spacing_min_m','spacing_max_m','spacing_cv','turn_mean_rad','turn_max_rad','vertical_travel_fraction','aperture_mean_m')


def describe(g,loop):
    p=g[:,:3];d=np.diff(np.r_[p,p[:1]] if loop else p,axis=0);s=np.linalg.norm(d,axis=1)
    u=d/np.maximum(s[:,None],1e-6);turn=np.arccos(np.clip((u[1:]*u[:-1]).sum(1),-1,1))
    return np.array([len(g),s.sum(),*np.ptp(p,axis=0),s.min(),s.max(),s.std()/s.mean(),turn.mean(),turn.max(),
                     abs(d[:,2]).sum()/s.sum(),np.exp(g[:,9:11]).mean()],np.float32)


def signature(g,loop):
    # Ordered arc-length center polyline, not a flown/time-optimal trajectory.
    p=g[:,:3];p=np.r_[p,p[:1]] if loop else p
    s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    t=np.linspace(0,s[-1],64,endpoint=not loop)
    return np.stack([np.interp(t,s,p[:,i]) for i in range(3)],1).ravel()


def build(args):
    propose=proposal
    independent=getattr(args,'variant','legacy')=='independent_v5'
    if independent:
        from starscream.course_model.generation_v5 import proposal as propose
    root=args.output;root.mkdir(parents=True,exist_ok=True)
    destination=root/'courses.h5';partial=root/'courses.partial.h5'
    if destination.exists() or partial.exists():raise RuntimeError('Refusing existing dataset; choose a new output')
    refs=[load_track(Path('starscream/assets/tracks')/(name+'.yaml')) for name in REFS]
    started=time.perf_counter();rejections=Counter();offset=0;seen=set()
    signatures=[];descriptors=[];ref_origin=[];family_ids=[];primitive_ids=[]
    with h5py.File(partial,'w') as f:
        f.attrs.update(schema=VERSION,complete=False,seed=args.seed,reference_names=json.dumps(REFS),
            family_names=json.dumps(FAMILIES+REFS),descriptor_names=json.dumps(DESCRIPTORS),
            feasibility_label_contract='-1 unknown; no course in this corpus is automatically RL admitted')
        for k,shape,dtype,chunks in [('gates',(args.count*32,17),'f4',(256,17)),
            ('context',(args.count,19),'f4',(min(256,args.count),19)),
            ('offsets',(args.count+1,),'i8',(min(256,args.count+1),))]:
            f.create_dataset(k,shape=shape,maxshape=(None,)+shape[1:],dtype=dtype,chunks=chunks,compression='lzf',shuffle=True)
        for k,dt in [('parent_id','i4'),('family','i2'),('reference','i2'),('split','i1'),
                     ('primitive','i1'),('attempt','i2'),('strength','f4'),('ground_lift_m','f4'),('feasibility','i1')]:
            f.create_dataset(k,(args.count,),dtype=dt,chunks=(min(256,args.count),),compression='lzf',shuffle=True)
        f.create_dataset('fingerprint',(args.count,),dtype='S64',compression='lzf')
        for i in range(args.count):
            for attempt in range(150):
                track,meta=propose(i,args.seed,refs,attempt)
                reasons=static_reasons(track)
                if reasons:rejections.update(reasons);continue
                g,c,_=pack(track)
                if independent:
                    # Common canonical arena, not a per-course bounding-box ID.
                    c[13:]=[-150,150,-150,150,0,50]
                    if static_reasons(unpack(g,c)):
                        rejections['canonical_arena']+=1;continue
                digest=hashlib.sha256(g.tobytes()+c.tobytes()).hexdigest()
                if digest in seen:rejections['duplicate']+=1;continue
                seen.add(digest);break
            else:raise RuntimeError(f'Unable to fill index {i}: {dict(rejections)}')
            f['gates'][offset:offset+len(g)]=g;f['offsets'][i]=offset;offset+=len(g)
            f['context'][i]=c;f['fingerprint'][i]=digest.encode()
            for k,v in meta.items():f[k][i]=v
            f['feasibility'][i]=-1
            descriptors.append(describe(g,track.loop));signatures.append(signature(g,track.loop));ref_origin.append(meta['reference'])
            family_ids.append(meta['family']);primitive_ids.append(meta['primitive'])
            if i<20:save_track_yaml(track,root/'examples'/f'course_{i:05d}.yaml')
            if (i+1)%1000==0:print(f'courses={i+1} elapsed={time.perf_counter()-started:.1f}s',flush=True)
        f['offsets'][args.count]=offset;f['gates'].resize((offset,17))
        f.create_dataset('descriptors',data=np.array(descriptors),compression='lzf',shuffle=True)
        f.create_dataset('shape_signature',data=np.array(signatures,dtype=np.float32),compression='lzf',shuffle=True)
        reference_group=f.create_group('references')
        for ref in refs:
            g,c,transform=pack(ref);r=reference_group.create_group(ref.name)
            r.create_dataset('gates',data=g);r.create_dataset('context',data=c);r.attrs['transform']=json.dumps(transform)
        f.attrs['complete']=True;f.flush()
    partial.rename(destination)
    report={'schema':VERSION,'count':args.count,'gates':offset,'size_bytes':destination.stat().st_size,
        'elapsed_seconds':time.perf_counter()-started,'rejections':dict(rejections),
        'family_counts':dict(Counter((FAMILIES+REFS)[i] for i in family_ids)),
        'primitive_counts':dict(Counter(PRIMITIVES[i] for i in primitive_ids)),
        'descriptor_quantiles':{k:np.quantile(np.array(descriptors)[:,j],[0,.05,.5,.95,1]).tolist() for j,k in enumerate(DESCRIPTORS)},
        'coverage':{},'limitations':['static proposal validation, not MPCC feasibility',
        'human region is labeled reference proximity, not a learned or exhaustive human-track manifold',
        'reference-neighborhood proximity is designed into the corpus; independent coverage reported separately',
        'no obstacle geometry, timing demand, or policy competence labels',
        'gate center polyline is a geometric signature, not a racing trajectory']}
    desc=np.array(descriptors);sig=np.array(signatures);orig=np.array(ref_origin)
    scale=np.maximum(np.std(desc[orig<0],axis=0),1)
    with h5py.File(destination,'r') as f:
        for ref in refs:
            g,c,_=pack(ref);rd=describe(g,ref.loop);rs=signature(g,ref.loop)
            # Count/length plus ordered spatial shape: explicit units, no claim
            # these distances are proven transfer predictors.
            shape_distance=np.sqrt(np.mean((sig-rs)**2,axis=1))
            report['coverage'][ref.name]={}
            for label,select in [('independent',orig<0),('reference_derived',orig>=0)]:
                if not select.any():
                    report['coverage'][ref.name][label]={'count':0};continue
                ids=np.flatnonzero(select);ranks=shape_distance[ids]+abs(desc[ids,0]-len(g))*.5
                nearest=int(ids[np.argmin(ranks)])
                a,b=f['offsets'][nearest:nearest+2];ng=f['gates'][a:b];nc=f['context'][nearest]
                save_track_yaml(unpack(ng,nc,f'{ref.name}_{label}_nearest'),root/'nearest'/f'{ref.name}_{label}.yaml')
                row={'index':nearest,'family':(FAMILIES+REFS)[family_ids[nearest]],'gate_count':len(ng),
                     'ordered_xyz_rmse_m':float(shape_distance[nearest]),
                     'descriptor_rms_z':float(np.sqrt(np.mean(((desc[nearest]-rd)/scale)**2))),
                     'length_relative_error':float(abs(desc[nearest,1]-rd[1])/rd[1])}
                if len(ng)==len(g):
                    row['gate_position_rmse_m']=float(np.sqrt(np.mean((ng[:,:3]-g[:,:3])**2)))
                    row['frame6_rmse']=float(np.sqrt(np.mean((ng[:,3:9]-g[:,3:9])**2)))
                report['coverage'][ref.name][label]=row
        # Sibling groups must never cross train/validation.
        parents=f['parent_id'][:];splits=f['split'][:]
        assert all(len(set(splits[parents==p]))==1 for p in np.unique(parents))
        report['split_counts']={'train':int((splits==0).sum()),'validation':int((splits==1).sum())}
    (root/'audit.json').write_text(json.dumps(report,indent=2))
    plot(root,destination,refs)
    print(json.dumps(report,indent=2),flush=True)


def plot(root,path,refs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(12,11))
    audit=json.loads((root/'audit.json').read_text())
    with h5py.File(path,'r') as f:
        for r,ref in enumerate(refs):
            g,_,_=pack(ref)
            for j,label in enumerate(['reference','independent','reference_derived']):
                if j:
                    if audit['coverage'][ref.name][label].get('count')==0:
                        axes[r,j].set_visible(False);continue
                    index=audit['coverage'][ref.name][label]['index'];a,b=f['offsets'][index:index+2];g=f['gates'][a:b]
                ax=axes[r,j];p=np.r_[g[:,:3],g[:1,:3]]
                ax.plot(p[:,0],p[:,1],alpha=.6);ax.scatter(g[:,0],g[:,1],c=g[:,2],cmap='viridis')
                for k,point in enumerate(g):ax.annotate(str(k+1),point[:2],fontsize=7)
                ax.set_aspect('equal');ax.set_title(ref.name.replace('_reconstructed','').replace('_source_consistent_v2','')+'\n'+label,fontsize=8)
                ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
    fig.suptitle('Canonical ordered gate-center paths; colors = height. No flight feasibility claim.')
    fig.tight_layout();fig.savefig(root/'coverage.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--count',type=int,default=20000)
    parser.add_argument('--seed',type=int,default=2026090501);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--variant',choices=('legacy','independent_v5'),default='legacy')
    args=parser.parse_args()
    if args.count<100 or args.count%100:parser.error('count must be a multiple of 100')
    build(args)
