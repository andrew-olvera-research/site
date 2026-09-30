"""Freeze the plant-teacher EKF/readout student recipe and fingerprints."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
config = json.loads((root/'configs/exp/v6.21.1.distill/base_ekf_interface_3m.json').read_text())
teacher = 'outputs/checkpoints/starscream-v6.21.1.1-plant-selection25-dagger/best-step-150819355-selection_suite_success-0.57875.pt'
source = 'configs/exp/v6.21.1.1/plant_dagger.yaml'
config['teacher_checkpoint'] = teacher
config['teacher_sha256'] = hashlib.sha256((root/teacher).read_bytes()).hexdigest()
config['training_source'] = source
config['training_source_sha256'] = hashlib.sha256((root/source).read_bytes()).hexdigest()
config['output'] = 'outputs/vision-distillation/v62111-plant-ekf-readout-async-48r'
config.update(rounds=48, permanent_rounds=4, recent_replay_rounds=5,
              rows_per_track_per_round=1024, replay_strata=[.40,.35,.25],
              permanent_fraction=.25, readout_weight=.05,
              collector_timeout_seconds=3600, early_stop_min_round=32,
              early_stop_patience_evals=8)
config['wandb']['run_name'] = 'v62111-plant-ekf-readout-async-48r'
config['wandb']['local_event_path'] = config['output']+'/wandb-events.jsonl'
config['wandb']['tags'] = ['vision-distillation','v6.21.1.1','plant-teacher',
                           'ekf','action-readout','async-dagger']
config['wandb']['train_metric_allowlist'] += [
    'readout','collection_rows','collection_seen','replay_rows',
    'actor_staleness_rounds','evaluation_seconds']
target = root/'configs/exp/v6.21.1.1/plant_ekf_readout_student_async.json'
target.write_text(json.dumps(config,indent=2)+'\n')
print(target)
