"""Compact causal transition shards for fixed-corpus teacher pretraining pilots."""
from pathlib import Path
import os
import h5py
import numpy as np
from torch.utils.data import Dataset


def causal_indices(episode_start, index, history):
    if history<1 or index<episode_start:raise ValueError('invalid causal sequence')
    return np.maximum(np.arange(index-history+1,index+1),episode_start)


class TeacherTransitionDataset(Dataset):
    """Worker-local HDF5 handles. Dynamics remain raw; normalize in the trainer.

    Features are already normalized. Never concatenate across resets, and never
    confuse teacher labels with the actions executed to produce the history.
    """
    def __init__(self,path,history=3,indices=None):
        self.path=str(Path(path).resolve());self.history=history;self._file=None;self._pid=None
        with h5py.File(self.path) as f:
            if f.attrs['schema']!='offline-teacher-transitions-v1':raise ValueError('wrong teacher corpus schema')
            self.indices=np.flatnonzero(f['label_valid'][:]) if indices is None else np.asarray(indices)
    def __len__(self):return len(self.indices)
    def _open(self):
        if self._pid!=os.getpid():
            if self._file:self._file.close()
            self._file=h5py.File(self.path,'r',rdcc_nbytes=32*1024*1024);self._pid=os.getpid()
        return self._file
    def __getitem__(self,index):
        self._open()
        f=self._file;i=int(self.indices[index]);start=int(f['episode_start'][i]);ix=causal_indices(start,i,self.history)
        # A slice, then local duplicate indexing: h5py fancy indices forbid repeats.
        first=int(ix[0]);x=f['features'][first:i+1][ix-first].astype(np.float32)
        return dict(history=x,action=f['expert_actions'][i],executed_action=f['executed_actions'][i],
                    previous_action=f['previous_actions'][i],
                    speed_command=np.float32(f['speed_command'][i]),
                    dynamics=f['dynamics'][i],dynamics_valid=bool(f['dynamics_valid'][i]),
                    track=f['track_id'][i],episode=f['episode_id'][i],gate=f['gate_index'][i])
    def __getitems__(self,indices):
        """Batch HDF5 reads; preserve sampler order and repeated sampled rows."""
        if not len(indices):return []
        f=self._open();rows=self.indices[np.asarray(indices)]
        unique,inverse=np.unique(rows,return_inverse=True)
        starts=f['episode_start'][unique][inverse]
        windows=np.maximum(rows[:,None]+np.arange(1-self.history,1),starts[:,None])
        feature_rows,feature_inverse=np.unique(windows,return_inverse=True)
        x=f['features'][feature_rows][feature_inverse].reshape(len(rows),self.history,-1).astype(np.float32)
        fields=dict(action='expert_actions',executed_action='executed_actions',previous_action='previous_actions',dynamics='dynamics',
                    dynamics_valid='dynamics_valid',speed_command='speed_command',track='track_id',episode='episode_id',gate='gate_index')
        batch={k:f[name][unique][inverse] for k,name in fields.items()}
        batch['dynamics_valid']=[bool(v) for v in batch['dynamics_valid']]
        return [dict(history=x[j],**{k:v[j] for k,v in batch.items()}) for j in range(len(rows))]
    def __getstate__(self):
        state=dict(self.__dict__);state['_file']=None;state['_pid']=None;return state


def hierarchical_weights(track,family,regime,gate,valid):
    """Equal family -> track -> occupancy regime -> active gate -> transition."""
    keys=np.stack([family,track,regime,gate],1);out=np.zeros(len(track),np.float64)
    def visit(ids,depth,mass):
        if depth==4:out[ids]=mass/len(ids);return
        groups=np.unique(keys[ids,depth])
        for group in groups:visit(ids[keys[ids,depth]==group],depth+1,mass/len(groups))
    ids=np.flatnonzero(valid)
    if len(ids):visit(ids,0,1.)
    return out
