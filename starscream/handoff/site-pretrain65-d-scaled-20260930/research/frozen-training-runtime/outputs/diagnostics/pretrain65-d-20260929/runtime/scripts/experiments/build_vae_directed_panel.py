"""Freeze a small source/target/random/directed candidate panel for audits."""
import argparse
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from starscream.course_model.model import CourseVAE,VAEConfig
from starscream.course_model.schema import pack,unpack,static_reasons
from starscream.course_model.directed import curve_distance,ordered_curve,project_ball
from starscream.course_model.geometry_objective import geometry_terms
from starscream.course_model.training import atomic_json
from starscream.env.tracks import load_track
from starscream.env.procedural_tracks import save_track_yaml,geometry_fingerprint


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=2026090613)
    p.add_argument('--world-frame',action='store_true',help='invert canonicalization for policy use')
    p.add_argument('--radii',type=float,nargs='+',default=[.25,.5,1.,2.])
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('choose new panel output')
    args.output.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(args.seed)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=CourseVAE(VAEConfig(**saved['config'])).cuda().eval();model.load_state_dict(saved['model'])
    for parameter in model.parameters():parameter.requires_grad_(False)
    source=load_track('swift_champion_2022_exact');target=load_track('multigp_cdra_2026_reconstructed')
    sg,sc,source_transform=pack(source);tg,tc,target_transform=pack(target)
    g=torch.from_numpy(sg[None]).cuda();c=torch.from_numpy(sc[None]).cuda();target_g=torch.from_numpy(tg[None]).cuda()
    n=len(sg);counts=torch.tensor([n],device='cuda');mask=torch.ones(1,n,dtype=torch.bool,device='cuda')
    with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
        z0=model.forward_uniform(g,c,sample=False)['mu']
        baseline,_=model.decode(z0,counts,c,_length=n)
    rows=[]
    def record(name,raw,ctx,meta):
        raw=raw.detach().float();ctx=ctx.detach().float()
        arr=raw[0].cpu().numpy().copy();arr[:,15:17]=raw[0,:,15:17].sigmoid().cpu().numpy() if meta.get('decoded') else arr[:,15:17]
        try:
            if (abs(arr[:,9:11])>5).any():raise ValueError('aperture range')
            reference=target if meta['method']=='exact_target' else source
            transform=target_transform if meta['method']=='exact_target' else source_transform
            track=unpack(arr,ctx[0].cpu().numpy(),name=name,
                         transform=transform if args.world_frame else None)
            if args.world_frame:
                # Restore original bounds and reset metadata, not inflated
                # canonical AABB transformed back to another inflated AABB.
                track=replace(track,bounds=reference.bounds.copy(),metadata=dict(reference.metadata))
            reasons=static_reasons(track)
            if any(np.any(gate.position<track.bounds[:,0]) or np.any(gate.position>track.bounds[:,1]) for gate in track.gates):reasons.append('center_bounds')
        except ValueError as error:
            rows.append(dict(name=name,static_valid=False,reasons=[str(error)],**meta));return
        path=save_track_yaml(track,args.output/'tracks'/(name+'.yaml')).resolve()
        value=dict(name=name,path=str(path),fingerprint=geometry_fingerprint(track),static_valid=not reasons,reasons=reasons,
            target_curve_distance_m=float(curve_distance(raw[...,:3],target_g[...,:3])),
            source_curve_distance_m=float(curve_distance(raw[...,:3],g[...,:3])),flight_feasibility='unknown',**meta)
        if len(arr)==n:value['max_source_gate_displacement_m']=float((raw[0,:,:3]-g[0,:,:3]).norm(dim=-1).max())
        rows.append(value)
    record('source_exact_canonical',g,c,dict(method='exact_source',decoded=False))
    record('target_exact_canonical',target_g,torch.from_numpy(tc[None]).cuda(),dict(method='exact_target',decoded=False))
    record('source_decoded',baseline,c,dict(method='decoded_source',decoded=True,radius=0.))
    target_curve=ordered_curve(target_g[...,:3])
    with sdpa_kernel(SDPBackend.MATH):
        for radius in args.radii:
            if radius<=0:raise ValueError('radius must be positive')
            direction=torch.randn_like(z0);direction=direction/direction.norm()
            zr=z0+radius*direction
            with torch.no_grad():random_raw,_=model.decode(zr,counts,c,_length=n)
            record(f'random_r{radius}',random_raw,c,dict(method='random_latent',decoded=True,radius=radius))
            z=z0.detach().clone().requires_grad_(True);opt=torch.optim.Adam([z],lr=.025)
            best_z=z.detach().clone();best_objective=float('inf')
            for iteration in range(150):
                opt.zero_grad();raw,_=model.decode(z,counts,c,_length=n)
                shape=((ordered_curve(raw[...,:3])-target_curve)/20).square().mean()
                fidelity=geometry_terms(raw,g,mask,c)
                # Keep apertures near source: enlarging gates is not progress.
                size=(raw[...,9:11]-g[...,9:11]).square().mean()
                loss=shape+.05*fidelity['floor']+.1*size
                if float(loss.detach())<best_objective:best_objective=float(loss.detach());best_z=z.detach().clone()
                loss.backward();opt.step()
                with torch.no_grad():z.copy_(project_ball(z,z0,radius))
            with torch.no_grad():value,_=model.decode(best_z,counts,c,_length=n)
            record(f'directed_r{radius}',value,c,dict(method='decoded_geometry_descent',decoded=True,radius=radius,objective=best_objective))
            with torch.no_grad():
                origin_raw,_=model.decode(z0,counts,c,_length=n)
                anchored=g.clone()
                anchored[...,:3]+=value[...,:3]-origin_raw[...,:3]
            record(f'anchored_r{radius}',anchored,c,dict(method='source_anchored_position_delta',decoded=False,radius=radius,objective=best_objective,restriction='source rotations apertures semantics retained; positions only'))
            # Matched physical displacement, not merely equal latent radius.
            displacement=(anchored[...,:3]-g[...,:3]).norm()
            random_delta=random_raw[...,:3].detach()-baseline[...,:3]
            random_anchor=g.clone()
            random_anchor[...,:3]+=random_delta*displacement/random_delta.norm().clamp_min(1e-8)
            record(f'anchored_random_r{radius}',random_anchor,c,dict(method='matched_source_anchored_random',decoded=False,radius=radius))
            positions=g[...,:3].detach().clone().requires_grad_(True)
            classical_loss=(ordered_curve(positions)-target_curve).square().mean()
            gradient=torch.autograd.grad(classical_loss,positions)[0]
            classical=g.clone()
            classical[...,:3]-=gradient*displacement/gradient.norm().clamp_min(1e-8)
            record(f'classical_r{radius}',classical,c,dict(method='matched_direct_geometry_gradient',decoded=False,radius=radius))
    report=dict(schema='vae-directed-fixed-count-panel-v1',checkpoint=args.checkpoint,seed=args.seed,
        context='fixed source context; no structural gate-count transition',world_frame=args.world_frame,
        metric='ordered normalized-arc center polyline, not flight/control utility',records=rows)
    atomic_json(args.output/'panel.json',report)
    atomic_json(args.output/'manifest.json',{'records':[r for r in rows if r.get('static_valid')]})
    print(rows)


if __name__=='__main__':main()
