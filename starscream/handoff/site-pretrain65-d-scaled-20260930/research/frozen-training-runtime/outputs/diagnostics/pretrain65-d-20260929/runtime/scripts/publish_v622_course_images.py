#!/usr/bin/env python3
"""Render and publish one geometry image per frozen v6.22 course.

The W&B run/artifact flow intentionally mirrors ``publish_video_review.py``.
Images are geometry-only; no policy score or actor rollout is used.
"""
from collections import Counter
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401: registers 3D projection
from matplotlib.colors import Normalize
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import wandb

from starscream.wandb import load_project_env
from starscream.env.procedural_tracks import geometry_fingerprint
from starscream.env.tracks import load_track


ROOT = Path(__file__).resolve().parents[1]


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=ROOT/'outputs/course-images/v622')
    p.add_argument('--project', default='starscream')
    p.add_argument('--entity')
    p.add_argument('--run-name', default='v622-course-geometry-review')
    p.add_argument('--manifest', type=Path, action='append', default=None)
    p.add_argument('--offline', action='store_true', help='render files without creating a W&B run')
    return p.parse_args()


def resolve(path):
    path = Path(path)
    if path.is_absolute(): return path
    candidates = [ROOT/path, Path.cwd()/path]
    return next(p.resolve() for p in candidates if p.is_file())


def image_for(track, record, destination):
    positions = np.asarray([g.position for g in track.gates], dtype=float)
    altitude = positions[:, 2]
    norm = Normalize(vmin=min(-0.1, float(altitude.min())), vmax=max(9., float(altitude.max())))
    cmap = plt.cm.viridis
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection='3d')
    closed = np.concatenate((positions, positions[:1]))
    ax.plot(closed[:,0], closed[:,1], closed[:,2], color='.60', lw=.9, zorder=0)
    for index, gate in enumerate(track.gates):
        color = cmap(norm(gate.position[2]))
        if gate.render:
            # Draw the exact oriented rectangular aperture, not a projected
            # chord. Width/height are taken directly from the serialized gate.
            corners = [gate.position + sx*gate.size[0]/2*gate.lateral + sz*gate.size[1]/2*gate.up
                       for sx, sz in ((-1,-1),(1,-1),(1,1),(-1,1),(-1,-1))]
            corners = np.asarray(corners)
            ax.plot(corners[:,0], corners[:,1], corners[:,2], color=color, lw=2.5, zorder=3)
        else:
            ax.scatter(*gate.position, facecolors='none', edgecolors=[color], s=45, lw=1.4, zorder=2)
        ax.text(*(gate.position + np.array([.12,.12,.12])), str(index), fontsize=7)
        normal = np.asarray(gate.normal)
        ax.quiver(gate.position[0], gate.position[1], gate.position[2], normal[0], normal[1], normal[2],
                  length=.9, normalize=True, color=color, arrow_length_ratio=.25, linewidth=.8)
    # These are the serialized simulator envelope limits, not a padded crop.
    # set_box_aspect preserves the true x:y:z meter ratios in the rendered view.
    bounds = np.asarray(track.bounds, dtype=float)
    ax.set_xlim(*bounds[0]); ax.set_ylim(*bounds[1]); ax.set_zlim(*bounds[2])
    ax.set_box_aspect(bounds[:,1] - bounds[:,0])
    ax.view_init(elev=25, azim=-60)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
    ax.grid(alpha=.22)
    source = record.get('source', record.get('prior_exposure', 'unknown'))
    exposure = record.get('prior_exposure', 'unknown')
    fig.suptitle(f"{record['suite']} | {record['family']} | {exposure}\n{track.name} | physical {sum(g.render for g in track.gates)}/{len(track.gates)} route checkpoints | source {source}\nExact simulator bounds: x [{bounds[0,0]:.1f}, {bounds[0,1]:.1f}]  y [{bounds[1,0]:.1f}, {bounds[1,1]:.1f}]  z [{bounds[2,0]:.1f}, {bounds[2,1]:.1f}] m", fontsize=10)
    fig.text(.5, .015, 'Rectangles = exact physical aperture planes; hollow points = virtual checkpoints; arrows = directed normals; color = altitude', ha='center', fontsize=8)
    fig.savefig(destination, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    args = args_parser(); load_project_env()
    manifests = args.manifest or [ROOT/'configs/eval/v6_22_real60.manifest.json', ROOT/'configs/eval/v6_22_real100_hard.manifest.json']
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for manifest_path in manifests:
        payload = json.loads(manifest_path.read_text())
        split = 'real60' if payload['records'][0]['suite'] == 'public_reference' and 'real60' in manifest_path.name else 'real100-hard'
        # The first record is public in both manifests; use the filename as the split authority.
        split = 'real100-hard' if 'real100_hard' in manifest_path.name else 'real60'
        folder = args.output_dir/split; folder.mkdir(parents=True, exist_ok=True)
        for record in payload['records']:
            track = load_track(resolve(record['path']))
            fingerprint = geometry_fingerprint(track)
            if fingerprint != record['fingerprint']: raise ValueError(f'fingerprint drift: {record["name"]}')
            destination = folder/f'{record["name"]}.png'
            image_for(track, record, destination)
            records.append(dict(split=split, name=record['name'], suite=record['suite'], family=record['family'],
                exposure=record.get('prior_exposure'), fingerprint=fingerprint, path=str(destination.relative_to(ROOT))))
    index = dict(schema='starscream-v622-course-image-review-v1', image_count=len(records),
        suites=dict(real60=60, real100_hard=100), shared_public_references=8,
        images=records, note='Geometry-only render; no actor metrics. Hollow checkpoints are virtual; bars are physical apertures.')
    index_path = args.output_dir/'index.json'; index_path.write_text(json.dumps(index, indent=2)+'\n')
    # A single PDF makes local review easy while the individual PNGs are the
    # durable per-course W&B media and artifact entries.
    for split in ('real60','real100-hard'):
        paths = [Path(r['path']) for r in records if r['split'] == split]
        with PdfPages(args.output_dir/f'{split}-all-courses.pdf') as pdf:
            for path in paths:
                figure = plt.figure(figsize=(13,6)); image = plt.imread(ROOT/path)
                plt.imshow(image); plt.axis('off'); pdf.savefig(figure, bbox_inches='tight'); plt.close(figure)
    if args.offline:
        print(json.dumps(dict(index=str(index_path), images=len(records), uploaded=False), indent=2)); return
    run = wandb.init(project=args.project, entity=args.entity, name=args.run_name,
        job_type='course-geometry-review', tags=['v622','geometry','real60','real100-hard'],
        config={'image_count':len(records), 'source_manifests':[str(p) for p in manifests], 'geometry_only':True})
    artifact = wandb.Artifact(f'{args.run_name}-images', type='course-geometry-images',
        description='One geometry image per real60 and real100-hard course, including labeled public references.')
    for index_number, record in enumerate(records):
        path = ROOT/record['path']
        caption = f"{record['split']} | {record['family']} | {record['suite']} | {record['name']}"
        run.log({f"course/{record['split']}/{record['name']}": wandb.Image(str(path), caption=caption)}, step=index_number)
        artifact.add_file(str(path), name=f"{record['split']}/{path.name}")
    artifact.add_file(str(index_path), name='index.json')
    for split in ('real60','real100-hard'): artifact.add_file(str(args.output_dir/f'{split}-all-courses.pdf'), name=f'{split}-all-courses.pdf')
    run.log_artifact(artifact)
    run.summary.update({'images':len(records), 'real60_images':sum(r['split']=='real60' for r in records),
        'real100_hard_images':sum(r['split']=='real100-hard' for r in records), 'shared_reference_images':160-52-92})
    run_url = getattr(run, 'url', None)
    if not run_url and hasattr(run, 'get_url'):
        run_url = run.get_url()
    report = dict(index=index, run_url=run_url, project=args.project, run_name=args.run_name,
        artifact=f'{args.run_name}-images', uploaded=True)
    (args.output_dir/'wandb-publish.json').write_text(json.dumps(report, indent=2)+'\n')
    run.finish(); print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
