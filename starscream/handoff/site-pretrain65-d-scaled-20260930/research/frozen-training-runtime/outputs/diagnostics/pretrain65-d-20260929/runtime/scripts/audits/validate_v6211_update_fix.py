"""Fail-closed preflight for the v6.21.1 update-density correction."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / 'configs/exp/v6.21.1/pretrain65_dagger.yaml'
CONFIG = ROOT / 'configs/exp/v6.21.1/update_fix_dagger.yaml'
ALLOWED = {
    'run_name', 'seed', 'mpcc_build_root', 'racing_line_cache', 'updates_per_round',
    'dagger_online_replay_rows_per_round', 'online_replay_capacity',
    'dagger_permanent_expert_capacity', 'tags', 'resume_checkpoint',
}


def main():
    control = json.loads(CONTROL.read_text())
    candidate = json.loads(CONFIG.read_text())
    a, b = control['dagger'], candidate['dagger']
    changed = {key for key in set(a) | set(b) if a.get(key) != b.get(key)}
    assert changed == ALLOWED, (changed, ALLOWED)
    ratio = candidate['experiment_notes']['update_fix']['label_ratio']
    assert math.isclose(ratio, 725360.282051282 / 289240.8461538461)
    assert b['updates_per_round'] == math.ceil(2026 * ratio)
    assert b['dagger_online_replay_rows_per_round'] == math.ceil(27001 * ratio)
    assert b['online_replay_capacity'] == math.ceil(1575001 * ratio)
    assert b['dagger_permanent_expert_capacity'] == math.ceil(675000 * ratio)
    assert b['rounds'] == a['rounds'] == 218
    assert b['resume_checkpoint'] == (
        '/workspace/outputs/checkpoints/'
        'starscream-v6.21.1.update-fix-pretrain65-dagger/latest.pt'
    )
    assert b['episodes_per_round'] == a['episodes_per_round'] == 520
    update_density = b['updates_per_round'] / 725360.282051282
    old_update_density = 2026 / 289240.8461538461
    retention_density = b['dagger_online_replay_rows_per_round'] / 725360.282051282
    old_retention_density = 27001 / 289240.8461538461
    assert abs(update_density / old_update_density - 1) < 2e-4
    assert abs(retention_density / old_retention_density - 1) < 2e-4
    assert candidate['checkpoint']['run_name'] == b['run_name']
    assert candidate['wandb']['run_name'] == b['run_name']
    print(json.dumps({
        'status': 'PASS', 'changed_dagger_keys': sorted(changed),
        'label_ratio': ratio, 'updates_per_round': b['updates_per_round'],
        'update_per_label': update_density, 'v621_update_per_label': old_update_density,
        'online_rows_per_round': b['dagger_online_replay_rows_per_round'],
        'retained_rows_per_label': retention_density,
        'v621_retained_rows_per_label': old_retention_density,
    }, indent=2))


if __name__ == '__main__':
    main()
