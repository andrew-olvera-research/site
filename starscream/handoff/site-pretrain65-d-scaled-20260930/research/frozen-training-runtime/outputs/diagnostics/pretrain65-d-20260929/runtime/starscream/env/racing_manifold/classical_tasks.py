"""Small classical task interventions; no learned embeddings or task scores."""
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=8)
def _reset_bank(path):
    with np.load(path, allow_pickle=False) as data:
        states=data['states'].copy(); actions=data['previous_actions'].copy()
        gate=int(data['gate_index']); fingerprint=str(data['geometry_fingerprint'])
    if states.ndim!=2 or states.shape[1]!=25 or actions.shape!=(len(states),4) or len(states)==0:
        raise ValueError('invalid archived-state bank shape')
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError('nonfinite archived state')
    return states,actions,gate,fingerprint


def archived_task_reset(track, seed, episode_index):
    """Reinitialize from realized expert states, not an exact simulator snapshot.

    Simulator/estimator randomization is applied normally; history follows the
    established repeated-reset convention. All remaining gates stay visible.
    """
    from ..procedural_tracks import geometry_fingerprint
    cyclic=(track.metadata or {}).get('cyclic_expert_reset_bank')
    if cyclic is not None:
        return cyclic_expert_reset(track, seed, episode_index, cyclic)
    path=(track.metadata or {}).get('curriculum_reset_bank')
    if path is None: return None
    states,actions,gate,fingerprint=_reset_bank(str(path))
    if geometry_fingerprint(track)!=fingerprint or not 0<=gate<len(track.gates):
        raise ValueError('archived reset bank geometry/gate mismatch')
    index=int(np.random.default_rng(int(seed)+104729*int(episode_index)).integers(len(states)))
    return dict(state=states[index].copy(),previous_action=actions[index].copy(),
        gate_index=gate,route_plan_total_gates=len(track.gates)-gate,
        spawn=dict(sampler='realized-expert-state-reset-v1',bank_index=index,gate_index=gate))


@lru_cache(maxsize=64)
def _cyclic_bank(path):
    with np.load(path, allow_pickle=False) as data:
        return {k:data[k].copy() for k in data.files}


def cyclic_expert_reset(track, seed, episode_index, spec):
    """Gate-balanced realized starts; always require a complete cyclic lap.

    This is a physical-state reset, NOT a simulator snapshot. Actuator/history
    initialization follows the environment's explicit previous-action contract.
    """
    from ..procedural_tracks import geometry_fingerprint
    rng=np.random.default_rng(int(seed)+104729*int(episode_index))
    if rng.random() >= float(spec.get('probability', .5)): return None
    data=_cyclic_bank(str(spec['path']))
    states=data['states']; actions=data['previous_actions']; gates=data['gate_indices']
    if (not track.loop or str(data['geometry_fingerprint']) != geometry_fingerprint(track)
        or states.ndim!=2 or states.shape[1]!=25 or actions.shape!=(len(states),4)
        or gates.shape!=(len(states),) or len(states)==0
        or not np.isfinite(states).all() or not np.isfinite(actions).all()
        or np.any(gates<0) or np.any(gates>=len(track.gates))):
        raise ValueError('invalid cyclic expert reset bank')
    gate=int(rng.choice(np.unique(gates)))
    index=int(rng.choice(np.flatnonzero(gates==gate)))
    state=states[index].copy()
    state[:3]+=rng.uniform(-.08,.08,3)
    state[7:10]+=rng.uniform(-.15,.15,3)
    return dict(state=state,previous_action=actions[index].copy(),gate_index=gate,
        route_plan_total_gates=len(track.gates),
        spawn=dict(sampler='cyclic-realized-expert-v1',bank_index=index,gate_index=gate))


def classical_ladder_choice(*, source_success, frontier_success, candidate_success,
                            candidate_episodes, candidate_fraction,
                            source_floor=.75, mastery=.60, candidate_floor=.20):
    """Advance only the next explicit goal; two-pass hysteresis lives in manager."""
    if source_success<source_floor: return 'rehearse'
    if frontier_success<mastery: return 'hold'
    if candidate_episodes<24 or candidate_success<candidate_floor or candidate_fraction<.55:
        return 'hold'
    return 'expand'
