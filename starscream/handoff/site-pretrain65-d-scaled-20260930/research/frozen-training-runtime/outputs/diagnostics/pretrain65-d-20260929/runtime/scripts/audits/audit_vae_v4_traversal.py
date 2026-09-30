"""Read-only geometry/history audit of the completed v4 traversal ablation.

Writes derived evidence only; no policy updates or generator modifications.
Run in /workspace with python -m scripts.audits.audit_vae_v4_traversal.
"""
import json
from pathlib import Path
import numpy as np
import torch
import wandb
from starscream.env.tracks import load_track
from starscream.course_model.schema import pack
from starscream.course_model.directed import ordered_curve
from starscream.course_model.training import atomic_json


def curve(track):
    return ordered_curve(torch.tensor(pack(track)[0][None, :, :3])).numpy().ravel()


def descriptors(track):
    p = np.array([g.position for g in track.gates])
    lengths = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
    return dict(gate_count=len(p), center_polyline_length_m=float(lengths.sum()),
                minimum_spacing_m=float(lengths.min()), height_range_m=float(np.ptp(p[:, 2])))


def main():
    source = load_track('swift_champion_2022_exact')
    target = load_track('multigp_cdra_2026_reconstructed')
    a, b = curve(source), curve(target)
    residual = b-a
    rows = []
    for row in json.loads(Path('outputs/course-model/v4/rays/all.json').read_text())['records']:
        track = load_track(row['path'])
        c = curve(track)
        delta = c-a
        rows.append(dict(name=row['name'], static_valid=row['static_valid'],
            reasons=row['reasons'], **descriptors(track),
            target_curve_rms_m=float(np.linalg.norm(b-c)/8),
            proxy_distance_reduction=float(1-np.linalg.norm(b-c)/np.linalg.norm(residual)),
            proxy_direction_cosine=float(delta@residual/(np.linalg.norm(delta)*np.linalg.norm(residual))),
            all_gate_rotations_preserved=all(np.allclose(g.rotation,h.rotation)
                                           for g,h in zip(track.gates,source.gates))))
    api=wandb.Api(timeout=30)
    histories={}
    for arm in ('hyper-fixed','online_vae','online_classical'):
        root=Path('outputs/checkpoints')/f'starscream-vae-v4-{arm}-ppo-524k'
        checkpoint=torch.load(root/'latest.pt',map_location='cpu',weights_only=False)
        rid=json.loads((root/'wandb-run.json').read_text())['run_id']
        run=api.run('andrewolvera/starscream/'+rid)
        keys=['_step']+[k for k in run.summary.keys() if '/probe/' in k and 'x12/' in k
                      and ('full_course_success' in k or 'mean_gates' in k)]
        histories[arm]=dict(trace=checkpoint['online_manifold_curriculum_state']['trace'],
                            next_rung_history=list(run.scan_history(keys=keys,page_size=100)))
    atomic_json(Path('outputs/audits/vae-v4-traversal-diagnosis.json'),dict(
        source=descriptors(source), target=descriptors(target),
        base_curve_rms_m=float(np.linalg.norm(residual)/8), geometry=rows, histories=histories,
        caveat='Curve RMS/cosine are canonical center-polyline proxies, not intrinsic manifold or flown-trajectory metrics.'))
    print('Saved outputs/audits/vae-v4-traversal-diagnosis.json')


if __name__=='__main__':
    main()
