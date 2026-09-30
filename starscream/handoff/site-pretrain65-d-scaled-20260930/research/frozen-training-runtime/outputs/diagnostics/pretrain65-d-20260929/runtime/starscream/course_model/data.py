"""Packed compressed HDF5, process-local lazy reads, per-batch padding only."""
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader,Sampler
from .schema import VERSION,GATE_DIM


class CourseDataset(Dataset):
    def __init__(self,path,split=None,cache=False):
        self.path=str(path);self._file=None;self._pid=None
        self.cached=None
        with h5py.File(path,'r') as f:
            if f.attrs['schema']!=VERSION or not f.attrs.get('complete',False):raise ValueError('invalid/incomplete dataset')
            self.indices=np.arange(len(f['context'])) if split is None else np.flatnonzero(f['split'][:]==split)
            self.lengths=np.diff(f['offsets'][:])[self.indices]
            self.families=f['family'][:][self.indices]
            self.references=f['reference'][:][self.indices]
            if cache:
                self.cached={k:f[k][:] for k in ('gates','context','offsets','parent_id')}
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        if self.cached is None and self._pid!=os.getpid():
            self.close();self._file=h5py.File(self.path,'r',rdcc_nbytes=2*1024**2);self._pid=os.getpid()
        f=self.cached if self.cached is not None else self._file
        i=int(self.indices[index]);a,b=f['offsets'][i:i+2]
        return {'gates':torch.from_numpy(f['gates'][a:b]),'context':torch.from_numpy(f['context'][i]),
                'index':i,'parent_id':int(f['parent_id'][i]),
                'family':int(self.families[index]),'reference':int(self.references[index])}
    def close(self):
        if self._file is not None:self._file.close()
        self._file=None;self._pid=None
    def __getstate__(self):
        state=self.__dict__.copy();state['_file']=None;state['_pid']=None;return state
    def __del__(self):self.close()


def collate_courses(items):
    lengths=torch.tensor([len(x['gates']) for x in items]);n=int(lengths.max())
    gates=torch.zeros(len(items),n,GATE_DIM)
    for i,x in enumerate(items):gates[i,:len(x['gates'])]=x['gates']
    return {'gates':gates,'mask':torch.arange(n)[None]<lengths[:,None],
            'context':torch.stack([x['context'] for x in items]),'indices':torch.tensor([x['index'] for x in items])}


def collate_uniform(items):
    """No padding: preserve all observations; fail loudly on mixed counts."""
    if len({len(x['gates']) for x in items})!=1:raise ValueError('mixed count in uniform batch')
    return {'gates':torch.stack([x['gates'] for x in items]),
            'context':torch.stack([x['context'] for x in items]),
            'indices':torch.tensor([x['index'] for x in items]),
            'family':torch.tensor([x['family'] for x in items]),
            'reference':torch.tensor([x['reference'] for x in items])}


class CountBatchSampler(Sampler):
    """Shuffle within counts AND across batches; visit each course once/epoch.

    Remainders are kept. This groups computation, not family oversampling.
    """
    def __init__(self,lengths,batch_size,shuffle=True,seed=0):
        if batch_size<1:raise ValueError('batch size must be positive')
        self.groups=[np.flatnonzero(lengths==n) for n in np.unique(lengths)]
        self.batch_size=batch_size;self.shuffle=shuffle;self.seed=seed;self.epoch=0
    def set_epoch(self,epoch):self.epoch=epoch
    def __len__(self):return sum((len(g)+self.batch_size-1)//self.batch_size for g in self.groups)
    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch);batches=[]
        for group in self.groups:
            group=rng.permutation(group) if self.shuffle else group
            batches.extend(group[i:i+self.batch_size].tolist() for i in range(0,len(group),self.batch_size))
        if self.shuffle:rng.shuffle(batches)
        yield from batches


def make_loader(path,split,batch_size=256,workers=0,cache=True,seed=0):
    ds=CourseDataset(path,split,cache=cache)
    sampler=CountBatchSampler(ds.lengths,batch_size,shuffle=split==0,seed=seed)
    extra={'persistent_workers':True,'prefetch_factor':2,'multiprocessing_context':'spawn'} if workers else {}
    return DataLoader(ds,batch_sampler=sampler,num_workers=workers,pin_memory=True,
                      collate_fn=collate_uniform,**extra)
