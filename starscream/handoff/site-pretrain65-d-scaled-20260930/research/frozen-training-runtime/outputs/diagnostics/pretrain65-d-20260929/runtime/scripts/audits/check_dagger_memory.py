"""Run inside the training container after a WSL memory change; no mutations."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from starscream.memory_pressure import memory_pressure_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--minimum-total-gib',type=float,default=0.)
    args = parser.parse_args()
    result = memory_pressure_snapshot()
    print(json.dumps(result,indent=2))
    if result.get('memory_effective_total_gib',0.) < args.minimum_total_gib:
        raise SystemExit('Effective WSL/container memory is below the requested threshold')


if __name__ == '__main__':
    main()
