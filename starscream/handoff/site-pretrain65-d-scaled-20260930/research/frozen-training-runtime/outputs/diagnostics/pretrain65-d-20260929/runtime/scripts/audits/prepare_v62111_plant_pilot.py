"""Small integration exercise, deliberately not a learning-quality ablation."""
import json
from pathlib import Path

config = json.loads(Path('configs/exp/v6.21.1.1/plant_dagger.yaml').read_text())
name = 'starscream-v62111-plant-pipeline-smoke-v2'
if (Path('outputs/checkpoints')/name).exists():
    raise FileExistsError(name)
for section in ('dagger', 'checkpoint', 'wandb'):
    config[section]['run_name'] = name
config['dagger'].update(rounds=3, episodes_per_round=65, updates_per_round=32,
    evaluation_episodes=8, reporting_evaluation_curriculum=None,
    midtrain_pace_probe=None, reporting_evaluation_interval=0,
    dagger_permanent_expert_rounds=1, dagger_successful_coverage_rounds=1, dagger_dart_expert_rounds=1,
    dagger_minimum_successful_episodes_per_track=0,
    dagger_permanent_expert_minimum_episodes_per_track=0)
config['wandb'].update(enabled=False, event_path=f'/workspace/outputs/logs/{name}.events.jsonl',
    local_event_path=None, local_full_event_path=f'/workspace/outputs/logs/{name}.full.events.jsonl')
path = Path('outputs/dagger-throughput')/f'{name}.json'
path.write_text(json.dumps(config, indent=2)+'\n')
print(path)
