"""Extend the same generator stream; freeze original validation and parent splits."""
from pathlib import Path
import hashlib
import json
import h5py
import numpy as np
from starscream.course_model.generation_v5 import proposal
from starscream.course_model.schema import pack,unpack,static_reasons
from scripts.experiments.build_course_vae_dataset import describe,signature


def main():
    src=Path('outputs/course-model/v5/independent20k/courses.h5')
    dst=Path('outputs/course-model/decoder-v2/scaled-courses.h5')
    if dst.exists():raise RuntimeError('Refusing existing scaled corpus')
    ratio=62563985/22596241
    with h5py.File(src) as f:
        seed=int(f.attrs['seed']);base_train=int((f['split'][:]==0).sum())
        target=round(base_train*ratio/5)*5
        arrays={k:v[:] for k,v in f.items() if isinstance(v,h5py.Dataset)}
        attrs=dict(f.attrs)
    seen=set(arrays['fingerprint'].tolist());extra={k:[] for k in arrays if k not in ('gates','offsets')};gate_rows=[]
    i=len(arrays['context']);added=0
    while base_train+added<target:
        parent=i//5
        if np.random.default_rng(np.random.SeedSequence([seed,parent,901])).integers(10)==0:
            i=(parent+1)*5;continue
        for attempt in range(150):
            track,meta=proposal(i,seed,[],attempt)
            if static_reasons(track):continue
            g,c,_=pack(track);c[13:]=[-150,150,-150,150,0,50]
            if static_reasons(unpack(g,c)):continue
            digest=hashlib.sha256(g.tobytes()+c.tobytes()).hexdigest().encode()
            if digest in seen:continue
            break
        else:raise RuntimeError(f'Cannot qualify index {i}')
        seen.add(digest);gate_rows.append(g)
        values=dict(meta,context=c,fingerprint=digest,feasibility=-1,
                    descriptors=describe(g,track.loop),shape_signature=signature(g,track.loop))
        assert values['split']==0
        for k in extra:extra[k].append(values[k])
        added+=1;i+=1
        if added%1000==0:print(f'added={added} target={target-base_train}',flush=True)
    temp=dst.with_suffix('.partial.h5')
    with h5py.File(src) as source,h5py.File(temp,'w') as f:
        f.attrs.update(attrs);f.attrs['complete']=False
        f.create_dataset('gates',data=np.concatenate([arrays['gates'],*gate_rows]),compression='lzf',shuffle=True)
        f.create_dataset('offsets',data=np.r_[arrays['offsets'],arrays['offsets'][-1]+np.cumsum([len(g) for g in gate_rows])])
        for k,rows in extra.items():
            f.create_dataset(k,data=np.concatenate([arrays[k],np.asarray(rows,dtype=arrays[k].dtype)]),compression='lzf',shuffle=True)
        source.copy('references',f)
        parents=f['parent_id'][:];splits=f['split'][:]
        assert not set(parents[splits==0])&set(parents[splits==1])
        assert np.array_equal(f['fingerprint'][:len(arrays['context'])],arrays['fingerprint'])
        assert len(set(f['fingerprint'][:]))==len(f['context'])
        f.attrs['complete']=True;f.attrs['scaled_from']=str(src)
    temp.replace(dst)
    report=dict(parameters_ratio=ratio,base_train=base_train,scaled_train=target,
        train_sample_ratio=target/base_train,epochs=200,base_exposures=base_train*200,
        scaled_exposures=target*200,validation_courses=int((arrays['split']==1).sum()),
        validation_unchanged=True,parent_disjoint=True,duplicates=0,
        static_checks=True,MPCC_feasibility='unknown',bytes=dst.stat().st_size)
    dst.with_suffix('.audit.json').write_text(json.dumps(report,indent=2));print(report,flush=True)


if __name__=='__main__':main()
