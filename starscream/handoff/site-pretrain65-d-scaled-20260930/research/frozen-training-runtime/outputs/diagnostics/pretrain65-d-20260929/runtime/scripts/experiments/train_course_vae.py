"""Launch a bounded course-VAE SSL run or resume an epoch checkpoint."""
import argparse
import yaml
from starscream.course_model.training import train


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--output');p.add_argument('--resume');p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    with open(args.config) as f:config=yaml.safe_load(f)
    if args.smoke:
        config['training']['epochs']=2;config['evaluation']['every_epochs']=1
        config['wandb']['enabled']=False
        if not args.output:p.error('--smoke requires a separate --output')
    train(config,args.output or config['output'],args.resume)


if __name__=='__main__':main()
