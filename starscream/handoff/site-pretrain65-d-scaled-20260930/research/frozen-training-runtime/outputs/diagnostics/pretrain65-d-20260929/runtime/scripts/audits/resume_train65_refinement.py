"""Resume the same refinement evidence with a different CPU worker count."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.audits import refine_v62111_train65_teacher as refinement


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    if not 1<=args.workers<=8:
        raise ValueError('Use between one and eight CPU workers')
    def pool_factory(**kwargs):
        return ProcessPoolExecutor(**{**kwargs,'max_workers':args.workers})
    # Scheduling only: candidate generation, seeds and numerical code are the
    # exact same imported functions covered by the frozen refinement contract.
    refinement.ProcessPoolExecutor=pool_factory
    print(f'refinement_workers={args.workers}',flush=True)
    refinement.main()


if __name__=='__main__':main()
