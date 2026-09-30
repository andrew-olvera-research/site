"""Run the frozen 100-episode suite without training or W&B publication."""
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from collections import Counter
import torch
from scripts import train_privileged_racing as t
from starscream.privileged_racing import load_policy_checkpoint
from starscream.evaluation_suite import configure_selection_suite, selection_metrics, selection_episode_seed
from starscream.behavior_feedback import sampler_feedback


def main():
    torch.set_num_threads(2)
    settings = json.loads(Path('configs/exp/v6.21.1.1/plant_dagger.yaml').read_text())['dagger']
    configure_selection_suite(settings)
    checkpoint = Path('outputs/checkpoints/starscream-v62111-plant-pipeline-smoke-v2/latest.pt')
    policy,normalizer,state,_ = load_policy_checkpoint(checkpoint,'cuda')
    policy.eval()
    stage = t.parse_stage(settings['evaluation_curriculum'])
    started = time.perf_counter()
    collector = t.ProcessRaceCollector(policy,normalizer,settings,stage,'cuda',
        workers=settings['evaluation_workers'],sampling_prefix='evaluation')
    jobs = []
    episode_plants = {}
    seen_plants = set()
    checked_batches = 0
    original_reset = collector._reset_slot
    def reset(slot,index,seed_base,track):
        jobs.append(dict(track=track,index=index,
            seed=selection_episode_seed(settings,track,index,seed_base,len(stage.tracks))))
        result = original_reset(slot,index,seed_base,track)
        episode_plants[id(slot)] = slot.history.array()[-1,103:152].copy()
        seen_plants.add(episode_plants[id(slot)].tobytes())
        assert slot.target_speed == settings['evaluation_track_speed_commands'][track]
        return result
    collector._reset_slot = reset
    def check_batch(histories,speeds):
        nonlocal checked_batches
        active = [slot for slot in collector.slots if not slot.done]
        assert histories.shape == (len(active),3,167)
        assert np.isfinite(histories).all()
        np.testing.assert_array_equal(speeds,np.asarray([slot.target_speed for slot in active],np.float32))
        raw = np.stack([slot.history.array() for slot in active])
        np.testing.assert_array_equal(histories,normalizer.numpy(raw))
        for slot,history in zip(active,raw):
            expected = episode_plants[id(slot)]
            np.testing.assert_array_equal(history[:,103:152],np.broadcast_to(expected,(3,49)))
        checked_batches += 1
    if collector.host_evaluation_graphs is not None:
        predict = collector.host_evaluation_graphs.predict_numpy
        def checked_predict(histories,speeds):
            check_batch(histories,speeds)
            return predict(histories,speeds)
        collector.host_evaluation_graphs.predict_numpy = checked_predict
    else:
        predict = collector.inference_policy
        def checked_predict(histories,speeds):
            check_batch(histories.cpu().numpy(),speeds.cpu().numpy())
            return predict(histories,speeds)
        collector.inference_policy = checked_predict
    try:
        rows = collector.evaluate_rows(episodes=settings['evaluation_episodes'],seed_base=settings['evaluation_seed'])
    finally:
        collector.close()
    seconds = time.perf_counter()-started
    assert len(rows) == len(jobs) == 100
    assert checked_batches > 0 and len(seen_plants) > 1
    assert set(Counter(r['track'] for r in rows).values()) == {4}
    jobs_by_index = {job['index']:job for job in jobs}
    for row in rows:
        assert row['episode_seed'] == jobs_by_index[row['episode_index']]['seed']
        assert row['start_gate_index'] == 0 and row['rollout_laps'] == 1
        assert row['target_speed_mps'] == 16.5
    metrics = selection_metrics(t.multitrack_metrics(rows,stage.target_gates),settings,stage)
    competence,frontiers = sampler_feedback(metrics,settings,settings['track_manifest'],6)
    assert len(settings['dagger_dynamic_sampling']['behavior_competence']) == 15
    mapping = json.loads(Path(settings['dagger_dynamic_sampling']['behavior_map']).read_text())
    sampler,weights,gates = t.update_dynamic_dagger_sampler(
        base_family_weights=settings['track_sampling_family_weights'],
        source_by_family={r['family']:r['source'] for r in mapping['training']},
        observed_competence=competence,observed_frontiers=frontiers,
        observed_losses={},previous_state=None,config=settings['dagger_dynamic_sampling'])
    report = dict(checkpoint=str(checkpoint),model_round=state['round'],
        suite_sha256=settings['evaluation_suite_sha256'],seconds=seconds,
        environment_steps=sum(r['steps'] for r in rows),metrics=metrics,jobs=jobs,
        conditioning=dict(checked_batches=checked_batches,distinct_episode_plants=len(seen_plants),
            commands=settings['evaluation_track_speed_commands'],
            plant_constants_stable_in_history=True,batched_commands_match_episode_labels=True),
        rows=rows,sampler=sampler)
    output = Path('outputs/dagger-throughput/selection25-native-audit.json')
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(output=str(output),seconds=seconds,episodes=len(rows),
        courses=len(Counter(r['track'] for r in rows)),
        selection_suite_success=metrics['selection_suite_success'],
        environment_steps=report['environment_steps'],
        training_shards=len(weights),behavior_families=len(sampler['competence']))))


if __name__ == '__main__':
    main()
