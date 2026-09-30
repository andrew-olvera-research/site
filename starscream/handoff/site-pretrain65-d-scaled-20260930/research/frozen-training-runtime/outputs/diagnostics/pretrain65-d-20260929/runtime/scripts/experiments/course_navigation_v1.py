"""Generate matched codec configs and train the separately frozen local tools."""
import argparse
from copy import deepcopy
from pathlib import Path
import yaml
from starscream.course_navigation.experiment import run_modules,fit_response
from starscream.course_model.training import atomic_json


def prepare(config):
    root=Path(config['output']);root.mkdir(parents=True,exist_ok=True)
    base=yaml.safe_load(Path(config['codec_base_config']).read_text())
    for arm in ['control','local_edit']:
        c=deepcopy(base);c.update(name=f'starscream-navigation-v1-{arm}',output=str(root/arm),
            initialize_from=config['codec_source'],allow_navigation_objective_change=True)
        c['training'].update(epochs=config['codec_epochs'],lr=.0001,min_lr=.00001,warmup_steps=40)
        c['evaluation']['every_epochs']=10
        if arm=='local_edit':c['objective']['navigation']=dict(samples=32,position_std_m=.35,yaw_std_rad=.08,
            edit_weight=1.,anchor_weight=.1,edge_weight=.1)
        Path(f'configs/exp/course_model/navigation_v1_{arm}.yaml').write_text(yaml.safe_dump(c,sort_keys=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/exp/course_model/navigation_v1.yaml')
    p.add_argument('--prepare',action='store_true');p.add_argument('--modules',choices=['control','local_edit','local_edit_isolated'])
    p.add_argument('--smoke',action='store_true');p.add_argument('--output');p.add_argument('--response',action='store_true')
    a=p.parse_args();c=yaml.safe_load(Path(a.config).read_text())
    if a.prepare:prepare(c)
    if a.modules:
        dest=a.output or f"{c['output']}/{a.modules}_modules"
        try:run_modules(c,f"{c['output']}/{a.modules}/latest.pt",dest,smoke=a.smoke)
        except FileExistsError:raise
        except BaseException as exc:
            Path(dest).mkdir(parents=True,exist_ok=True)
            atomic_json(Path(dest)/'status.json',dict(state='failed',error=repr(exc)))
            raise
    if a.response:fit_response(c['response_root'],a.output or f"{c['output']}/response")


if __name__=='__main__':main()
