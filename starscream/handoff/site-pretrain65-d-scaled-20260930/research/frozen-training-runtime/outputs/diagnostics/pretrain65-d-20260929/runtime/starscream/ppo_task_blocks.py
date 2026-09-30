"""Deterministic overlapping task blocks, independent of evaluation results."""
def task_block(tracks, cycle, *, size=4, stride=2, updates_per_block=2):
    tracks=tuple(tracks)
    if (not tracks or len(set(tracks))!=len(tracks) or
        not 1<=stride<=size<=len(tracks) or cycle<1 or updates_per_block<1):
        raise ValueError('invalid PPO task block schedule')
    start=((cycle-1)//updates_per_block*stride)%len(tracks)
    return tuple(tracks[(start+i)%len(tracks)] for i in range(size))
