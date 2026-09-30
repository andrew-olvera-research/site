"""Clean full-split reconstruction and bounded conditional-prior audit."""
import argparse
from pathlib import Path
import torch
from starscream.course_model.training import setup,evaluate,atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True);args=p.parse_args()
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    config=saved['training_config'];model,(_,loader)=setup(config)
    model.load_state_dict(saved['model'])
    result=evaluate(model,loader,config,geometry=True)
    result.update(checkpoint=str(args.checkpoint),epoch=saved['epoch'])
    destination=Path(args.output);destination.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(destination,result);print(result)


if __name__=='__main__':main()
