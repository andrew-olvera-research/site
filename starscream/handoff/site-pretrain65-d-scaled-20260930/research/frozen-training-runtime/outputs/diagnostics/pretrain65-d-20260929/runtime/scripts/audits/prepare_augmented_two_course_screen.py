"""Create a small, disjoint augmented two-parent slalom screen."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from starscream.env.racing_distribution.named_course_augmentation import generate_named_course_augmentations

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'outputs/diagnostics/v62111-train65-teacher-v4/frozen/approved/manifest.json'
DEST=ROOT/'outputs/course-pools/mini-slalom-aug2'
NAMES=('v6201_train_10_slalom_0','v6201_train_4_slalom_1')

def main():
    source=json.loads(SOURCE.read_text())
    byname={r['name']:r for r in source['records']}
    parents=[]
    for name in NAMES:
        r=byname[name]
        assert r['qualified'] and r['split']=='train'
        parents.append(dict(key=r['family'],path=r['path'],qualified_speed_mps=r['qualified_speed_mps']))
    path=generate_named_course_augmentations(DEST,parents=parents,
            train_per_parent=4,validation_per_parent=2,seed=2026092911)
    report=json.loads(path.read_text())
    assert len(report['records'])==12
    assert all(r['static_audit']['valid'] for r in report['records'])
    print(json.dumps(dict(manifest=str(path),records=len(report['records']),
                          splits={s:sum(r['split']==s for r in report['records']) for s in ('train','family_validation')})))

if __name__=='__main__':main()
