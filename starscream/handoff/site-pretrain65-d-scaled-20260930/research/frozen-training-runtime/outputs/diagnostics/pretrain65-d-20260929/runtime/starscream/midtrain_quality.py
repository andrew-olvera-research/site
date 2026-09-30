"""Course-relative admission of complete training trajectories during DAgger."""
import hashlib
import math


def reference_steps(reference_seconds, gate_count, start_gate, laps=1, hz=130):
    if not math.isfinite(reference_seconds) or reference_seconds<=0 or gate_count<1 or laps<1:
        raise ValueError('invalid midtrain reference')
    remaining=gate_count*laps-max(0,int(start_gate))
    if remaining<1:raise ValueError('start gate is beyond the route')
    # This is only a training admission estimate for gate-local resets; no
    # paper/evaluation claim uses a partial-route reference as an optimum.
    return reference_seconds*hz*remaining/gate_count


def admission_probability(method, successful, elapsed_steps, reference, hz=130):
    if not successful or elapsed_steps<1:return 0.
    if method=='hard_timed':
        return float(elapsed_steps<=math.floor(max(1.5*reference,reference+3*hz)))
    if method=='soft_time':
        return max(.2,min(1.,reference/elapsed_steps))
    raise ValueError('unknown trajectory admission method')


def admit(method, successful, elapsed_steps, reference, course, episode_index, seed_base):
    probability=admission_probability(method,successful,elapsed_steps,reference)
    if probability in (0.,1.):return bool(probability)
    key=f'{course}|{episode_index}|{seed_base}'.encode()
    draw=int.from_bytes(hashlib.sha256(key).digest()[:8],'big')/2**64
    return draw<probability
