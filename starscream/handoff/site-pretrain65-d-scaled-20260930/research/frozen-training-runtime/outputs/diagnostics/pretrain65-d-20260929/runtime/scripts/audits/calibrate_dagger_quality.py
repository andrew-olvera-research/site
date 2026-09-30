"""Derive candidate course bounds exclusively from clean complete expert probes.

Missing clean references stay missing. This does not certify teacher reliability;
the resulting bounds still require independent plant seeds and DART validation.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--minimum-clean', type=int, default=2)
    args = p.parse_args()
    if args.minimum_clean < 1:
        p.error('minimum clean episodes must be positive')
    episodes, clean, sources = {}, {}, []
    for filename in args.input:
        report = json.loads((filename.parent/'report.json').read_text())
        if not report['quality'].get('audit_only') or any(r['beta'] != 1 for r in report['results']):
            raise ValueError('calibration requires unmodified expert-only collection')
        source = dict(path=str(filename), speed_scale=report.get('speed_scale', 1.))
        sources.append(source)
        with np.load(filename) as a:
            data, tracks, success = a['trajectory'], a['tracks'], a['successful']
            if 'planned_plane_crossing' not in a['fields']:
                raise ValueError('calibration requires unclipped distance and reference-aware crossing telemetry')
            for i, name in enumerate(a['track_names']):
                name = str(name)
                x, passed = data[tracks == i], success[tracks == i]
                for seed in np.unique(x[:,12]):
                    mask = x[:,12] == seed
                    z = x[mask]
                    # Avoid treating repeated artifact imports as extra evidence.
                    key = (name, int(seed), source['speed_scale'])
                    if key in episodes:
                        raise ValueError(f'duplicate calibration episode {key}')
                    good = bool(passed[mask].all() and not z[:,1].any() and not z[:,16].any()
                                and np.isfinite(z[:,4]).all())
                    episodes[key] = dict(track=name, seed=int(seed), success=bool(passed[mask].all()),
                        misses=int(z[:,1].sum()), internal_recovery_rows=int(z[:,16].sum()),
                        clean=good, speed_scale=source['speed_scale'])
                    if good:
                        clean.setdefault(name, []).append((z, source['speed_scale']))
    bounds, missing = {}, []
    for name in sorted({k[0] for k in episodes}):
        rows = clean.get(name, [])
        if len(rows) < args.minimum_clean:
            missing.append(name)
            continue
        scales = {scale for _, scale in rows}
        if len(scales) != 1:
            raise ValueError('freeze one teacher speed per course before combining calibration evidence')
        z = np.concatenate([r for r, _ in rows])
        peak, dwell = float(z[:,4].max()), float(z[:,3].max())
        intervention = max(1.5, peak*1.25+.25)
        bounds[name] = dict(nominal_distance=max(.35,float(np.quantile(z[:,4],.99))*1.1+.1),
            intervention_distance=intervention, maximum_distance=max(5., intervention*1.5),
            maximum_gate_seconds=max(3., dwell*1.5+.5), recovery_seconds=max(3., min(8., dwell*3)))
    result = dict(schema='dagger-quality-calibration-v1', sources=sources,
        minimum_clean_episodes=args.minimum_clean, track_bounds=bounds, missing=missing,
        episodes=list(episodes.values()), status='candidate' if not missing else 'incomplete',
        limitation='Empirical training-only reference envelope; independent seed/DART confirmation required.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output), tracks_with_bounds=len(bounds), missing=missing,
        clean_episodes=sum(r['clean'] for r in episodes.values()), episodes=len(episodes)), indent=2))


if __name__ == '__main__':
    main()
