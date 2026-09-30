"""Build the v6.21.1 update-density correction from the frozen pretrain65 run.

Rounds 5--43 measured 725,360.282 valid labels/round for v6.21.1 versus
289,240.846 for v6.21.  Scaling by course count alone therefore diluted both
optimizer updates and retained replay rows per collected label.  This arm keeps
the v6.21.1 corpus and collection recipe fixed and changes only those budgets.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'configs/exp/v6.21.1/pretrain65_dagger.yaml'
OUT = ROOT / 'configs/exp/v6.21.1/update_fix_dagger.yaml'

OLD_LABELS_PER_ROUND = 289240.8461538461
NEW_LABELS_PER_ROUND = 725360.282051282
LABEL_RATIO = NEW_LABELS_PER_ROUND / OLD_LABELS_PER_ROUND


def main():
    config = json.loads(SOURCE.read_text())
    settings = config['dagger']
    run = 'starscream-v6.21.1.update-fix-pretrain65-dagger'
    settings['run_name'] = run
    settings['seed'] = 2026092112
    settings['mpcc_build_root'] = '/tmp/starscream-v6211-update-fix-pretrain65'
    settings['racing_line_cache'] = '/workspace/outputs/course-pools/v6211-pretrain/racing-lines-update-fix'
    settings['updates_per_round'] = math.ceil(2026 * LABEL_RATIO)
    settings['dagger_online_replay_rows_per_round'] = math.ceil(27001 * LABEL_RATIO)
    settings['online_replay_capacity'] = math.ceil(1575001 * LABEL_RATIO)
    settings['dagger_permanent_expert_capacity'] = math.ceil(675000 * LABEL_RATIO)
    settings['tags'] = list(settings['tags']) + ['update-density-fix', 'measured-label-scaling']
    config['checkpoint']['run_name'] = run
    config['wandb']['run_name'] = run
    config['wandb']['group'] = 'v6.21.1-update-fix'
    config['wandb']['local_event_path'] = f'/workspace/outputs/logs/{run}.events.jsonl'
    config['experiment_notes']['update_fix'] = {
        'control_run': 'starscream-v6.21.1-pretrain65-dagger',
        'calibration_rounds': '5-43',
        'v621_labels_per_round': OLD_LABELS_PER_ROUND,
        'v6211_labels_per_round': NEW_LABELS_PER_ROUND,
        'label_ratio': LABEL_RATIO,
        'contract': 'match v6.21 optimizer updates, retained rows, and replay capacities per collected valid label',
    }
    OUT.write_text(json.dumps(config, indent=2) + '\n')
    print(json.dumps({
        'config': str(OUT), 'run_name': run, 'label_ratio': LABEL_RATIO,
        'updates_per_round': settings['updates_per_round'],
        'online_rows_per_round': settings['dagger_online_replay_rows_per_round'],
        'online_capacity': settings['online_replay_capacity'],
        'permanent_capacity': settings['dagger_permanent_expert_capacity'],
    }, indent=2))


if __name__ == '__main__':
    main()
