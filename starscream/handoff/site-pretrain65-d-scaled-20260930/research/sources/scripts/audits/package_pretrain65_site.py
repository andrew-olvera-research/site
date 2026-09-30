"""Assemble descriptive research evidence and finite frontend statistics."""
import csv
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/site-pretrain65-d-scaled-20260930'
EVAL=ROOT/'outputs/evals/pretrain65-d-scaled-final-20260929'
BEHAVIOR=ROOT/'outputs/diagnostics/pretrain65-d-behavior-20260929'

def read(path): return json.loads(path.read_text())
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def copy(relative, category='research/sources'):
    source=ROOT/relative
    if not source.is_file(): return
    target=OUT/category/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
def dist(values):
    values=np.asarray(values,float)
    if not len(values): return dict(count=0,mean=None,median=None,p10=None,p90=None,p95=None,p99=None,minimum=None,maximum=None)
    return dict(count=len(values),mean=float(values.mean()),median=float(np.median(values)),
        p10=float(np.quantile(values,.1)),p90=float(np.quantile(values,.9)),p95=float(np.quantile(values,.95)),
        p99=float(np.quantile(values,.99)),minimum=float(values.min()),maximum=float(values.max()))
def aggregate(rows):
    n=len(rows)
    successes=[r for r in rows if r['success']]
    return dict(episodes=n,completed=len(successes),eventual_sr=len(successes)/n,
        timely_sr=sum(r['timely_success'] for r in rows)/n,
        clean_sr=sum(r['clean_success'] for r in rows)/n,
        clean_timely_sr=sum(bool(r['clean_success'] and r['timely_success']) for r in rows)/n,
        crash_sr=sum(r['crashed'] for r in rows)/n,
        success_lap_seconds=dist([r['steps']/130 for r in successes]),
        success_reference_ratio=dist([r['steps']/130/r['reference_seconds'] for r in successes]),
        clean_timely_lap_seconds=dist([r['steps']/130 for r in successes if r['clean_success'] and r['timely_success']]),
        recovered_lap_seconds=dist([r['steps']/130 for r in successes if not r['clean_success']]))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    report=read(EVAL/'real100-v2-e32.json')
    protocol=read(ROOT/'configs/eval/real100_v2_timed_protocol_v1.json')
    records={r['slot']:r for r in protocol['records']}
    rows=report['episodes']
    assert len(rows)==3200 and sum(r['success'] for r in rows)==2949
    raw60=read(EVAL/'real60-e32.json')
    summary=read(EVAL/'summary.json')
    courses=[]
    for slot,record in records.items():
        selected=[r for r in rows if r['slot']==slot]
        courses.append(dict(slot=slot,name=record['name'],family=record['family'],track=record['path'],
            reference_seconds=record['reference_seconds'],reference_kind=record['reference_kind'],
            deadline_steps=record['deadline_steps'],**aggregate(selected)))
    families={family:aggregate([r for r in rows if r['family']==family]) for family in sorted({r['family'] for r in rows})}
    old100=read(ROOT/'outputs/evals/v6211-update-fix-final/update-fix-real100-hard-v2-e8.json')
    old60=read(ROOT/'outputs/evals/v6211-update-fix-final/update-fix-real60-e8.json')
    comparisons=[]
    for label,old,new,count in [('real60',old60,summary['real60']['eventual'],1920),('real100-v2',old100,report['aggregate']['total_success'],3200)]:
        previous=float(old['metrics']['full_course_success'])
        comparisons.append(dict(suite=label,old_eventual_sr=previous,new_eventual_sr=new,
            delta_percentage_points=100*(new-previous),old_starts_per_course=8,new_starts_per_course=32,
            new_total_starts=count,old_timely_sr=None,old_clean_timely_sr=None,
            comparability='Historical recipe-level comparison. Different start count and inference/evaluation contracts; no matched causal attribution.'))
    outcomes={
        'clean_timely':sum(bool(r['clean_success'] and r['timely_success']) for r in rows),
        'missed_timely':sum(bool(r['success'] and not r['clean_success'] and r['timely_success']) for r in rows),
        'clean_late':sum(bool(r['clean_success'] and not r['timely_success']) for r in rows),
        'missed_late':sum(bool(r['success'] and not r['clean_success'] and not r['timely_success']) for r in rows),
        'noncompletion':sum(not r['success'] for r in rows)}
    frontend=dict(schema='starscream-pretrain65-site-data-v1',generated_date='2026-09-30',
        units=dict(time='seconds',distance='metres',orientation='unit quaternion (w,x,y,z)',rates='fractions in [0,1]',control_hz=130),
        checkpoint=report['checkpoint'],checkpoint_round=report['checkpoint_round'],checkpoint_steps=report['checkpoint_steps'],
        real100_v2=dict(aggregate=aggregate(rows),outcome_counts=outcomes,courses=courses,families=families,
            deadline_formula=protocol['deadline_formula'],hard_cap_steps=6000,miss_definition=report['miss_definition'],
            aggregation='Equal course weight, 32 starts on each of 100 courses; not family-weighted selection25.'),
        real60=dict(courses=60,starts_per_course=32,total_starts=1920,eventual_sr=summary['real60']['eventual'],
            timely_sr=None,clean_timely_sr=None,clean_sr=None,
            unavailable_reason='No complete frozen real60 timed/clean evaluation protocol in this result.',
            successful_per_course_statistics=summary['real60']['courses'],
            note='Per-course successful-lap medians are not a pooled episode distribution.'),
        historical_comparison=comparisons,
        student_transfer=dict(status='upcoming corrected-checkpoint distillation; no result included',
            teacher_reference='real100_v2.aggregate',student_eventual_sr=None,student_timely_sr=None,
            student_clean_timely_sr=None,paired_retention_ratio=None,
            note='Do not use the previous bug-affected student distillation as current transfer evidence.'),
        quality_notes=['Timely permits misses; clean/timely requires success, no ordered-reference miss, and deadline compliance.',
            'Successful-lap quantiles exclude noncompletions; retain completion denominators.',
            'Reference times are empirical witnesses, not certified optima.',
            'Public-reference adaptations are simulated reconstructions; these scores are not physical-racing leaderboard scores.',
            'Selection25 was reused to rank checkpoints. real60 and real100-v2 have been inspected during research.',
            'Gallery consists of curated complete benchmark episodes, including explicitly labeled failures and recoveries.'])
    write(OUT/'evals/frontend-data.json',frontend)
    write(OUT/'evals/real100-v2-course-statistics.json',courses)
    write(OUT/'evals/real100-v2-family-statistics.json',families)
    write(OUT/'evals/historical-comparison.json',comparisons)
    write(OUT/'evals/lap-distributions.json',dict(all_success=aggregate(rows)['success_lap_seconds'],
        successful_reference_ratio=aggregate(rows)['success_reference_ratio'],
        successful_lap_histogram=dict(bin_edges_s=list(range(0,49,2)),
            counts=np.histogram([r['steps']/130 for r in rows if r['success']],bins=list(range(0,49,2)))[0].tolist(),
            denominator=2949,noncompletions_excluded=251),
        successful_reference_ratio_histogram=dict(bin_edges=[0,.5,1,1.25,1.5,2,3,4,5,6],
            counts=np.histogram([r['steps']/130/r['reference_seconds'] for r in rows if r['success']],
                bins=[0,.5,1,1.25,1.5,2,3,4,5,6])[0].tolist(),denominator=2949),
        by_outcome={kind:dist([r['steps']/130 for r in rows if r['success'] and
            (('clean' if r['clean_success'] else 'missed')+'_'+('timely' if r['timely_success'] else 'late'))==kind])
            for kind in ['clean_timely','missed_timely','clean_late','missed_late']}))
    for source in EVAL.iterdir():
        if source.suffix in {'.json','.txt'}: copy(source.relative_to(ROOT),'evals/raw')
    for source in BEHAVIOR.iterdir():
        if source.suffix in {'.json','.csv','.png'}: copy(source.relative_to(ROOT),'evals/behavior')
    for source in [ROOT/'outputs/evals/v6211-update-fix-final/update-fix-real100-hard-v2-e8.json',ROOT/'outputs/evals/v6211-update-fix-final/update-fix-real60-e8.json']:
        copy(source.relative_to(ROOT),'evals/historical')
    # Human inspectable scalar rows, with units and actual total denominators.
    with (OUT/'evals/real100-v2-courses.csv').open('w',newline='') as f:
        fields=['slot','name','family','episodes','completed','eventual_sr','timely_sr','clean_sr','clean_timely_sr','crash_sr','median_success_s','p90_success_s','p95_success_s','reference_s']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in courses:
            flat={k:r[k] for k in fields if k in r}
            flat.update(median_success_s=r['success_lap_seconds']['median'],p90_success_s=r['success_lap_seconds']['p90'],
                p95_success_s=r['success_lap_seconds']['p95'],reference_s=r['reference_seconds'])
            writer.writerow(flat)
    checkpoint=Path(report['checkpoint'])
    if checkpoint.is_absolute(): checkpoint=checkpoint.relative_to('/workspace')
    model_dir=OUT/'model';model_dir.mkdir(exist_ok=True)
    # Drive's connector caps each file at 100 MiB. Ship identical inference state,
    # not optimizer/replay/RNG copies that are irrelevant to the site and distillation.
    for path in model_dir.rglob('*.pt'):
        path.unlink()
    payload=torch.load(ROOT/checkpoint,map_location='cpu',weights_only=False)
    inference_keys=['contract','stage','track','model','model_config','normalizer','feature_dim',
        'action_horizon','control_hz','plant_settings_schema','dynamics_target_mean','dynamics_target_std',
        'round','environment_steps','initial_checkpoint','action_contract','training_config','step','metrics']
    inference={key:payload[key] for key in inference_keys if key in payload}
    inference_path=model_dir/'teacher-inference-only.pt'
    torch.save(inference,inference_path)
    loaded=torch.load(inference_path,map_location='cpu',weights_only=False)
    def identical(a,b):
        if isinstance(a,torch.Tensor): return torch.equal(a,b)
        if isinstance(a,np.ndarray): return np.array_equal(a,b)
        if isinstance(a,dict): return a.keys()==b.keys() and all(identical(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)): return len(a)==len(b) and all(identical(x,y) for x,y in zip(a,b))
        return a==b
    assert identical(payload['model'],loaded['model']) and identical(payload['normalizer'],loaded['normalizer'])
    write(model_dir/'inference-checkpoint-provenance.json',dict(source_checkpoint=str(checkpoint),
        source_sha256=hashlib.sha256((ROOT/checkpoint).read_bytes()).hexdigest(),
        inference_checkpoint='teacher-inference-only.pt',
        inference_sha256=hashlib.sha256(inference_path.read_bytes()).hexdigest(),
        retained_keys=list(inference),excluded_training_resume_keys=sorted(set(payload)-set(inference)),
        all_model_tensors_and_normalizer_values_exactly_equal=True,
        use='Inference and student distillation; not an exact optimizer/replay/RNG training resume artifact.'))
    # Rebuild only this script's evidence folders so older scope cannot leak into the archive.
    for relative in ['research/sources','research/rl-context']:
        target=(OUT/relative).resolve()
        assert target.is_relative_to(OUT.resolve())
        if target.exists(): shutil.rmtree(target)
    performance_snapshot=OUT/'research/gpu-performance-history.json'
    if performance_snapshot.exists(): performance_snapshot.unlink()
    for path in ['outputs/diagnostics/pretrain65-d-20260929/production-config.json',
        'outputs/diagnostics/pretrain65-d-20260929/validation.json','outputs/diagnostics/pretrain65-d-20260929/train.status.json',
        'outputs/diagnostics/pretrain65-d-20260929/learning-analysis-snapshot.json',
        'outputs/checkpoints/starscream-pretrain65-d-scaled-20260929/top-k.json']:
        copy(path)
    for path in (ROOT/'docs').glob('*.md'):
        if path.name in {'pretrain65_d_scaled_20260929.md','pretrain65_d_learning_r13_20260929.md',
            'collection_next_20260929.md','collection_next_results_20260929.md','collection_ablation_20260929.md',
            'multifix_implementation_20260928.md','multifix_design_20260928.md','multifix_results_20260928.md',
            'v62111_plant_privileged.md','v622_benchmark_design.md','v622_base_pretrain50.md',
            'v6211_pretraining_behavior_site_brief.md'}:
            copy(path.relative_to(ROOT))
    for path in (ROOT/'configs/exp/v6.21.1.1').glob('*'):
        if path.is_file() and (path.name=='pretrain65_d_scaled.json' or
            path.name.startswith(('collection_next','mini_slalom_aug2'))): copy(path.relative_to(ROOT))
    # Preserve the training-time implementation separately from current exporter source.
    frozen=ROOT/'outputs/diagnostics/pretrain65-d-20260929/runtime'
    for path in frozen.rglob('*.py'):
        if '__pycache__' not in path.parts:
            copy(path.relative_to(ROOT),'research/frozen-training-runtime')
    for path in ['configs/eval/v6211_vision_selection25.json','configs/eval/v6_22_real60.yaml',
        'configs/eval/v6_22_real60.manifest.json','configs/eval/v6_22_real100_hard_v2.yaml',
        'configs/eval/v6_22_real100_hard_v2.manifest.json','configs/eval/real100_v2_timed_protocol_v1.json']:
        copy(path)
    # Track and reference evidence needed to validate the frozen protocol and gallery.
    track_paths={r['path'] for r in protocol['records']}
    for r in protocol['records']: copy(r['evidence'])
    import yaml
    for suite in ['configs/eval/v6_22_real60.yaml']:
        for row in yaml.safe_load((ROOT/suite).read_text())['active']:
            track_paths.add(row['track'])
    for path in track_paths: copy(path)
    config=read(ROOT/'outputs/diagnostics/pretrain65-d-20260929/production-config.json')
    def referenced_files(value):
        if isinstance(value,dict):
            for key,item in value.items():
                yield from referenced_files(key)
                yield from referenced_files(item)
        elif isinstance(value,list):
            for item in value: yield from referenced_files(item)
        elif isinstance(value,str):
            candidate=value.removeprefix('/workspace/')
            path=ROOT/candidate
            if path.is_file() and path.suffix in {'.json','.yaml','.yml'}:
                yield path
            elif path.is_file() and path.suffix=='.pt' and 'normalization' in path.name:
                yield path
    dependencies=set(referenced_files(config))
    for path in list(dependencies):
        if path.suffix=='.json':
            try: dependencies.update(referenced_files(read(path)))
            except (ValueError,UnicodeError): pass
    for path in dependencies:
        if path.resolve().is_relative_to(ROOT.resolve()): copy(path.relative_to(ROOT))
    for path in ['scripts/audits/export_pretrain65_site.py','scripts/audits/package_pretrain65_site.py',
        'scripts/render_research_gifs.py','scripts/export_site_trajectories.py',
        'scripts/audits/eval_privileged_real100_v2_dual.py','scripts/audits/run_pretrain65_d_full_evals.sh',
        'scripts/audits/summarize_pretrain65_d_full_evals.py','scripts/audits/analyze_pretrain65_d_behavior.py',
        'scripts/audits/analyze_pretrain65_d_live.py','scripts/audits/analyze_collection_ablation.py',
        'starscream/racing_evaluation.py','starscream/dagger_quality.py','starscream/plant_privileged.py',
        'starscream/privileged_racing.py','starscream/dagger_collection.py']:
        copy(path)
    body=REPORT
    report_dir=OUT/'research';report_dir.mkdir(exist_ok=True)
    (report_dir/'research-program-report.txt').write_text(body)
    paragraphs=body.strip().split('\n\n')
    page=['<!doctype html><html><head><meta charset="utf-8"><title>Starscream pretraining research program</title>',
        '<style>body{font:17px/1.65 system-ui;max-width:900px;margin:45px auto;padding:25px;color:#16202c}h1,h2{line-height:1.25}pre{white-space:pre-wrap;font:inherit;background:#f5f7fa;padding:16px}img{max-width:100%}</style></head><body>']
    for block in paragraphs:
        if block.startswith('STARSCREAM'):
            page.append('<h1>'+html.escape(block)+'</h1>')
        elif re.match(r'^\d+\. [^\n]+$',block): page.append('<h2>'+html.escape(block)+'</h2>')
        else: page.append('<p>'+html.escape(block).replace('\n','<br>')+'</p>')
    for image in ['behavior-overview.png','speed-response.png','long-recovery-trace.png']:
        page.append(f'<figure><img src="../evals/behavior/outputs/diagnostics/pretrain65-d-behavior-20260929/{image}"><figcaption>{html.escape(image)}</figcaption></figure>')
    page.append('</body></html>')
    (report_dir/'research-program-report.html').write_text('\n'.join(page))
    (OUT/'README.txt').write_text('Starscream scaled-D research handoff, 30 September 2026\n\nStart with research/research-program-report.html. Gallery: generated/index.html. Frontend data: evals/frontend-data.json. All SR values are fractions; time is seconds. Missing real60 clean/timely scores are null. This is the round-71 pretrained teacher.\n\nThe model/ folder contains teacher-inference-only.pt: model tensors, normalization, architecture and action/plant contracts exactly retained from the pinned checkpoint. Optimizer, replay, RNG and redundant safety snapshots are excluded to fit the Drive connector file limit. This supports inference/distillation but not exact training resume. Source and derived hashes are in model/inference-checkpoint-provenance.json. The original full checkpoint remains local. research/sources preserves source paths for configs, courses, methods and analysis. evals/raw contains the frozen result files, and evals/behavior contains episode CSVs, plots and tail analyses. captures contains exact full-state benchmark replays. Historical docs describe their dates and may precede later decisions; the dated report reconciles them.\n\nReproduce capture/render in the existing /workspace container: python scripts/audits/export_pretrain65_site.py. After rendering, build descriptive artifacts: python scripts/audits/package_pretrain65_site.py. Capture requires CUDA; render-only reuses verified full-state NPZ files. The folder preserves the model and relevant source evidence, not full training replay storage or a complete installable repository.\n')
    print(json.dumps(dict(package=str(OUT),courses=len(courses),families=len(families),outcomes=outcomes,report_words=len(body.split()))))

REPORT='''STARSCREAM — completion of the scaled-D pretraining phase
Research-program handoff · 30 September 2026 · privileged teacher/base model

1. What this handoff establishes

The strongest current pretrained base is the round-71 checkpoint from starscream-pretrain65-d-scaled-20260929, at 29,465,874 training environment steps. It reaches 92.15625% eventual completion on real100-v2, 79.65625% timely completion, 71.00000% clean completion, and 68.81250% clean-and-timely completion across 100 courses and 32 randomized starts per course. The separate real60 panel reaches 91.04167% eventual completion across 60 courses and 32 starts each. These are substantial improvements over the earlier base, achieved much earlier in training. They support a useful, broadly transferable teacher for the upcoming student distillation.

The central result is a recipe-level change in fitting efficiency and behavioral transfer. Corrected input/loss/replay contracts, coherent learner-controlled collection, refreshed normalization, and diverse teacher-qualified geometry collectively moved the program beyond the apparent earlier fitting wall. They did not remove every undesired recovery pattern. Clean execution and pace remain measurably below eventual completion, and public-reference outliers remain difficult. These measured teacher limits define what the next student must preserve and what cannot be assumed solved.

This is simulation evidence from a policy with privileged true route and plant state. The word teacher here means the neural base model that teaches a downstream student. The MPCC expert is a separate controller that supplies pretraining labels and feasibility evidence. Neither these results nor the gallery constitutes physical-flight or vision-only validation. Earlier student distillation affected by the known bug is not used as current transfer evidence.

2. Identity, selection, and experiment status

Pinned weights: outputs/checkpoints/starscream-pretrain65-d-scaled-20260929/best-step-029465874-selection_suite_timely_success-0.7375.pt.
SHA-256: cee73ce74fdfcfd64adb7b10d374ff143ef07980a61feff53112641d7a56eb3a.
Frozen production config: outputs/diagnostics/pretrain65-d-20260929/production-config.json.
Config SHA-256: d42ff193f4a7f751d89bfc372b65bfb0c35b4479115f49be98b2c6cd876a521c.

The checkpoint was ranked on family-weighted selection25 timely success, with score 0.7375. That selection criterion permits recovered flights; it is not clean/timely success. The 25 geometries were reused every round and are development validation. The saved best-three ledger identifies this as the leading retained checkpoint. latest.pt is a later checkpoint, not the selected base.

The planned scratch budget was 218 rounds, but the launcher ended with trainer exit -2 after latest reached round 93. The evaluated round-71 weights are valid and retained. Do not describe the run as a completed 218-round budget, establish asymptotic convergence from the interruption, or say that pretraining is mathematically exhausted. This handoff closes the current pretraining research phase around its best measured base.

3. Settings token and the observable plant

The starscream_route_plant_v1 observation has 167 values per history row: the original 103-value task/control/route input plus 64 plant and causal-context values. The suffix describes the applied rigid body and motors (15 values), aerodynamics (30), timing and actuator limits (4), and current wind/frame context (15). Forty-nine values are reset constants and fifteen refresh at command time. It exposes mass, inertia, motor behavior, drag, wind/gust parameters and phase, delay, actuator bounds, and frame context without adding course IDs, seeds, future state, or future expert actions.

The applied descriptor is scaled in fixed physical units, preserving absolute magnitude and physical zero; it is not normalized by each sample's RMS. The scaled suffix uses an identity checkpoint normalizer. Fresh expert/DART statistics normalize the corrected base feature/action contract. Earlier plant-interface documents describe the introduction of the schema; the scaled-D run's own frozen config and normalization define the actual experiment.

One settings token from the latest history row is prepended to the sequence: settings, state/control pairs for three history rows, six ordered future-route tokens, and an action readout token. The sequence grows from 13 to 14 tokens. A settings projector supplies a direct attention path and a zero-initialized FiLM scale/shift projection to ordinary tokens. The backbone retains width 320, four transformer blocks, eight attention heads, feedforward width 640, and three-frame history. The corresponding plant-model parameter count is 5,839,067, about 6.4% above the original 5,488,987 model.

This makes the controller conditional on the actual simulated plant rather than forcing one action mapping to average across hidden randomized dynamics. It does not establish arbitrary long-memory observability, an internal estimator, or causal attribution of the final success gain to the settings token alone. The transformer receives a finite observation history; the evaluator does not carry persistent long-horizon hidden state between calls.

4. Correctness repairs behind the improved fitting contract

The previous-action feature encoding was corrected to follow the configured action contract. Physical thrust authority remains 40 m/s² and labels remain unchanged. Offline action remapping had also overwritten previous-action features even when online inference retained the legacy encoding; replacement now follows the requested feature map consistently. Checkpoints/statistics with the incompatible legacy input contract are rejected for this corrected mapping.

Masked dynamics and chunk auxiliaries had displayed a valid-row mean while diluting gradients by the valid-row fraction. The corrected reduction preserves the valid-row mean gradient; all-invalid batches have zero loss and gradient. The correction applies to the relevant shared reduction paths rather than merely changing displayed telemetry.

Hierarchical replay quota remainder draws had repeatedly favored the first families/tracks/gates, allowing later groups to be starved at small quotas. Cached and uncached paths now use randomized unbiased quota rounding and retain the requested batch size. Regression evidence reproduced the mechanism; this does not imply every historical batch was starved. Shard checks were tightened to reject changed committed data, missing arrays, inconsistent rows/identity, and nonfinite values. That integrity work is preventative and does not demonstrate that historical shards were corrupt.

The experiment also changed action weighting and replay/retention. Those are fitting choices, not bugs. The report keeps those categories distinct. Their combined contribution is unmeasured; a matched factorial study would be required to assign an isolated causal share to each repair.

5. Collection methodology and full-scale pretraining recipe

Training uses the original 65-course pretrain65 corpus and its qualified per-course teacher/planner maps. Each original course is its own uniformly weighted collection/replay family; dynamic sampling is disabled. Mini-slalom augmentation and pace profiles used in the method screen do not enter this scaled training corpus. Selection geometry contributes neither replay nor normalization.

The D recipe begins with one expert/DART bootstrap round and uses 50% expert-prefix starts. Active collection then gives the learner coherent beta-zero control for a segment ending after 2–4 accepted gates. The MPCC supplies counterfactual corrective labels at learner-induced states without taking active control. Expert prefixes repeatedly seed later route states. Segment endpoints bound failed-state exposure without introducing an additional recovery cap or takeover mechanism.

The plausible mechanism is improved state relevance: coherent learner trajectories expose the errors that the policy must correct, while bounded segments avoid spending most fitting effort on very long failed recoveries. Diverse geometry and later-gate starts broaden the support; clean expert labels and corrected losses make that support easier to fit. This is a supported working explanation, not proof that segmentation alone created the gain.

The scaled schedule requests 520 episodes per round (eight per course), 5,081 optimizer updates per round, batch 1,536, constant learning rate 0.00012, and gradient clip 2. Action dimension weights are [1,1,1,0.35], physical action loss weight 0.75, and dynamics auxiliary weight 0.15. The enabled objective uses single-action and dynamics heads. Matching the older planned schedule would total 113,360 scheduled episodes and 1,107,658 updates at 218 rounds, but these are planned totals, not the actual interrupted run's counts.

Replay uses 90% recent online and 10% permanent expert draws. Retention is 203,142 online rows per round, online capacity 3,047,119, permanent capacity 423,199, and roughly a 15-round online horizon. Retention can underfill; episodes, valid labels, environment steps, optimizer updates, and lifetime sample reuse are separate measures.

Fresh normalization collected 260 expert/DART episodes over all 65 training courses, accepted 259, and retained 1,818 balanced rows per course (118,170 total), including 117,172 valid dynamics targets. Invalid labels were excluded. The production preflight recorded eighty passing regression tests, all-course integration, finite CUDA updates, and exact replay reconstruction/resume checks. This is validation of the recipe/runtime, not an additional benchmark success estimate.

6. From the apparent wall to much faster acquisition

The nine-course collection screen tested twelve scratch arms for 24 rounds and 15,024 updates each. Learner-controlled collection consistently helped eventual completion relative to per-step mixing, but clean/timely ranking was volatile. D repeat averaged 50% eventual / 13% clean-timely over its last six checkpoints; full learner averaged 60% / 9%; relaxed eight-second recovery averaged 51% / 15%. These small repeated validation panels motivate contrasts rather than establish a universally optimal collector.

On the closer full-scale plant/selection25 comparator, scaled D first reached 40% family-weighted completion at round 8 and 5.30M training steps versus round 116 and 83.76M. It first reached 50% at round 10 and 6.37M versus round 151 and 108.70M. That threshold used approximately 15.1 times fewer optimizer updates and 17.1 times fewer recorded environment steps. At D round 13, eventual success was 57.19%; the earlier run first matched that level around round 208 and 149.38M steps.

These are first-crossing descriptions from one seed per full-scale recipe. The old evaluation used four starts/course and a 10,000-step horizon; D used eight starts/course and 6,000 steps. Multiple collection, input, replay, loss and schedule changes are bundled. Normalization, smoke and evaluation costs are excluded from the training-step comparison. Present the large gain as recipe-level acquisition efficiency, not a causal multiplier assigned solely to D.

7. Why real60 and real100-v2 both matter

real60 represents championship-like course demands: 52 independently generated courses and eight shared public-reference adaptations. real100-v2 is the principal difficult panel: 92 generated courses plus the same eight reference anchors, with harder ordered geometry and replacement stress families introduced in v2. Their generated geometry is independent across panels; they are not two differently named copies of one set. Neither name means every course is a surveyed real racing venue.

The frozen manifests document teacher qualification, leakage/geometry checks, course provenance, and reference uncertainty. Public anchors include Swift, CDRA, A2RL, and UTT03/04/05/06/08 adaptations. They retain reconstruction/arena adjustments and are not official organizer-survey timing equivalents. The policy is evaluated under headless simulated dynamics, directed gate acceptance, randomized applied plants and starts, and ground-contact termination.

The old update-fix base recorded 55.625% eventual real60 completion and 43.750% real100-v2 completion at eight starts/course. The new base records 91.04167% and 92.15625% at 32 starts/course: descriptive increases of 35.42 and 48.41 percentage points. The historical hard-panel deficit has largely disappeared in eventual SR for this recipe. This does not imply the panels are equally difficult: real100 still exposes misses, timing deficits, severe public-reference failures, and a longer successful-lap distribution. The historical comparison changes episode count and evaluation contracts and is not a matched causal experiment.

8. Eventual, timely, and clean/timely must remain distinct

For real100-v2, eventual success means completing the full ordered course before the 6,000-step hard cap (46.15 seconds). Timely additionally requires finishing by the course's frozen deadline. The deadline is floor(130 × max(1.5 × reference_seconds, reference_seconds + 3)). References are empirical teacher witnesses, some explicitly provisional; they are not certified fastest possible times. Clean success requires no ordered-reference-aware gate miss. Clean/timely requires all three conditions. A clean label alone does not certify the shortest path, smooth flight, or absence of every hesitation.

Of 3,200 starts, 2,949 complete. The full partition is 2,202 clean/timely (68.8125%), 347 missed/timely (10.84375%), 70 clean/late (2.1875%), 330 missed/late (10.3125%), and 251 noncompletions (7.84375%). Failures comprise 248 crashes and three hard caps. Of successful flights, 74.67% are clean/timely. There are 747 completions outside the target quality class, including 677 after detected misses. A high eventual score can therefore coexist with substantial recovery behavior.

Fifty-nine courses complete all 32 starts, but only twenty-two are clean/timely on all 32; eleven have zero clean/timely. Some zero-quality courses still complete nearly every start. Family/course breakdowns are essential because 3,200 starts are not 3,200 independent geometries. real60 has no complete frozen timed protocol in its saved result; its timely and clean/timely fields in the frontend export are null rather than fabricated comparable scores.

9. Lap-time distribution, survivor bias, and recovery tails

Across the 2,949 successful real100-v2 flights, mean lap time is 10.95 seconds, median 10.32, p90 15.53, p95 17.63, and maximum 41.92. Successful reference-normalized time has median 1.176×, p90 1.625×, p95 1.978×, and maximum 5.593×. Course lengths differ, so absolute seconds and reference ratios should be shown together. References express empirical pace context rather than optimality.

Among the 99 courses with at least one completion, the mean per-course successful median is 10.65 seconds, median of course medians 10.22, and p90 of course medians 15.15. These describe course-balanced timing rather than pooled episode timing. The zero-success course stays in failure/SR summaries and has no finite successful median. real60's mean of successful per-course medians is 8.96 seconds and median of those medians is 7.60; these cannot be substituted for a pooled real60 episode median.

The completed flights include 408 with a resolved recovery longer than three seconds, 228 longer than five, 87 longer than eight, 49 longer than ten, 21 longer than fifteen, and nine longer than twenty. Those 87 represent 2.72% of all starts or 2.95% of completions. Separately, 150 completions have a no-pass interval longer than eight seconds; ordinary travel is included in no-pass time, so it is not identical to recovery time. Repeated misses and multiple recovered gates remain visible in the event CSVs.

The long-recovery witness real100-hard-055, seed 2035006365, takes 38.65 seconds and travels 308.4 metres, with 32.21 seconds from missing gate 5 to accepting it. The regenerated full episode retains the repeated excursions; no recovery is cut out. This is direct evidence that the undesired behavior was reduced sufficiently to enable strong fitting and transfer, but was not eliminated. Pair every timing chart with completion denominators: a faster survivor median can arise simply because difficult flights fail.

10. A2RL: an important boundary of the pretrained controller

On a2rl_s2_2026_source_consistent_v2 in real100-v2, the base completes 4/32 starts (12.5%); all four are timely after misses and none is clean/timely. Twenty-eight crash. Its successful-lap median is 19.37 seconds and p90 23.46, based on only four survivors. This is a severe outlier, not evidence of clean A2RL competence or saturation of every reference course. The generated folder includes a complete recovery success and a failure from the same frozen cohort.

The failures occur before several required gates rather than only one finish-gate event. Prior analysis located failures at next required gates 5,6,7,8,9,11,12. These are failure locations, not necessarily the instant the original error began. The teacher's dedicated qualification profile uses a 6 m/s command and supplies a 17.692-second nominal reference; that does not establish neural-policy competence at its benchmark command of 16.5 m/s. A separate actor-command-6 probe achieved only 1/32 completions and zero clean/timely, so simply reducing the command did not solve A2RL.

Ordered-chain support weakens as longer geometry combinations are compared with train65. The exploratory normalized nearest-support distance rises from 0.217 for individual transitions to 0.357 for three-transition chains and 0.403 for four-transition chains. This describes disclosed geometry features, not a causal architecture diagnosis. Broad transfer with this rough edge supports downstream exploration while leaving a clear longer-chain/first-pass execution challenge.

11. Command response and meaningful aggressiveness

The existing paired speed intervention spans five commands on nineteen unique diagnostic courses, with a random sixteen-course panel scored separately. Baseline 16.5 m/s gives 83.59% eventual and 60.16% clean/timely on that random panel; command 21 gives 87.50% eventual and 61.72% clean/timely. On starts completing at both commands, median lap ratio is 0.940, about 6% faster. The clean/timely gain is only two of 128 starts and its course-bootstrap interval includes zero.

Command 26 collapses eventual success to 23.44% and clean/timely to 12.50%. The base can respond usefully to pace requests, but higher commands are not a universal route to competent racing. Commands 21 and 26 extrapolate beyond the course-level training command range. Training commands are confounded with course/profile; there was no balanced within-course speed curriculum. The gallery and lap distributions show real pace capability alongside the limits rather than equating path speed with progress.

12. Student transfer: the measured teacher target and next handoff

The new teacher provides broad held-out geometry completion, useful corrections away from expert trajectories, and meaningful pace behavior to transfer into the student. Those are measured teacher properties. They are not a measured student retention ratio. A fresh student distillation from this corrected checkpoint has not yet produced a result in this handoff; the earlier bug-affected student is deliberately not treated as evidence for current student capability.

The transfer evaluation should pair teacher and student on identical course geometry, randomized starts/plants, speed commands, deadlines and canonical inference settings. Report absolute student eventual, timely, clean and clean/timely SR beside the teacher values, then report retention ratios for the same metric and panel. A retention ratio alone can conceal a weak absolute result. Preserve successful-lap distributions, reference ratios, failure counts and hard-reference outliers alongside retention. Use selection25 for development selection, and disclose the prior inspection of real60/real100-v2.

The settings-token diagram describes privileged teacher input. A deployable student's sensing/estimation inputs must be separately labeled; it must not be shown as receiving perfect plant or route state unless its configured interface actually provides that information. The useful current conclusion is that the corrected teacher has a much stronger and more efficiently learned control substrate to distill. Numerical claims about the upcoming student must wait for its paired evaluation.

13. Gallery provenance, package contents, and site guidance

The twenty-two regenerated assets are chosen from the frozen real100-v2 e32 cohort. The original eighteen gallery courses retain original benchmark geometry. Each prefers its fastest clean completion when available; additional assets show A2RL recovery and failure, a short timely recovery, and the long late rescue. Every full-state replay exactly matched the original step count, success, crash, gate count, misses, and complete pass/miss event sequence. Stable original inference lanes are retained in zero-padded batch 16, under canonical FP32 math and no TF32.

This cohort-based curation selects attractive and explanatory examples; it is not an unbiased gallery SR. The GIF/MP4 motion is rendered at simulation time, with state NPZ files and metre/second/unit-quaternion frontend trajectories retained. Poster, media, metadata and manifests are included. Recovery and failure labels use the same ordered-reference rule as the benchmark, replacing ambiguity from the older showcase's simpler raw-plane filter.

The archive includes generated media, interactive trajectory data, complete raw evaluations, episode/course CSVs, finite frontend JSON, outcome/time distributions, behavior plots, historical comparison data, the selected teacher's inference checkpoint, frozen production settings, course/protocol evidence, and relevant pretraining method reports/configs/source scripts. Every model tensor and normalization value in the derived inference checkpoint exactly matches the pinned source. Optimizer, replay, RNG and redundant safety snapshots are omitted to satisfy the Drive connector's 100 MiB file limit. Both hashes and the retained/excluded keys are documented; the derived artifact supports inference/distillation rather than an exact training resume. It also excludes full HDF5 training replay, credentials, and unrelated project storage. This is a descriptive and inspectable research handoff; it is not a complete clean-room installation of all runtime dependencies.

For the site, lead with the large recipe-level efficiency improvement and strong geometry transfer. Show real100-v2 as the principal hard evaluation and real60 as the championship-like companion. Present eventual, timely and clean/timely together. Use the comparable eventual rates to explain the disappearance of the earlier hard-panel deficit, then use quality and tails to describe the measured teacher target for student transfer. The strongest accurate description is a highly capable pretrained base with remaining first-pass, pacing and outlier-course weaknesses. Broad eventual completion is approaching a ceiling on many courses; clean/timely performance and all references have not saturated.

Sources and reading order: evals/frontend-data.json; raw real100-v2-e32.json and real60-e32.json; evals/behavior full-analysis.json, episode/course CSVs and speed-analysis.json; research/sources/docs/pretrain65_d_scaled_20260929.md, pretrain65_d_learning_r13_20260929.md, collection_next_results_20260929.md, multifix_implementation_20260928.md, v62111_plant_privileged.md, v622_benchmark_design.md; frozen configs/manifests/protocol. Original reports retain dates and experimental boundaries. Numerical frontend values should come from the machine-readable export rather than rounding in this narrative.
'''

if __name__=='__main__': main()
