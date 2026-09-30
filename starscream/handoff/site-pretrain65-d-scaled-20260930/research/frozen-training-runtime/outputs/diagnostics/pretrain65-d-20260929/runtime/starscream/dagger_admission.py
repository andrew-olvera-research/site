"""Frozen round-based course admission; no competence feedback or new labels."""
from pathlib import Path


def admission_schedule(settings, tracks):
    raw = settings.get("dagger_course_admission", [])
    if not raw:
        return ()
    if settings.get("dagger_learning_progress_task_bank", {}).get("enabled", False):
        raise ValueError("fixed admission cannot coexist with a task bank")
    if settings.get("dagger_dynamic_sampling", {}).get("enabled", False):
        raise ValueError("fixed admission experiment requires fixed sampling")
    allowed = {str(Path(p).resolve()): p for p in tracks}
    previous = set()
    schedule = []
    for phase in raw:
        start = int(phase["start_round"])
        selected = phase["tracks"]
        selected = list(tracks) if selected == "all" else selected
        if not isinstance(selected, (list, tuple)) or not selected:
            raise ValueError("admission phase requires nonempty tracks")
        keys = [str(Path(p).resolve()) for p in selected]
        if len(set(keys)) != len(keys) or not set(keys) <= allowed.keys():
            raise ValueError("duplicate or out-of-training admission tracks")
        if not previous <= set(keys):
            raise ValueError("course admission must be monotonic")
        if (not schedule and start != 1) or (schedule and start <= schedule[-1][0]):
            raise ValueError("admission rounds must start at one and increase")
        if start > int(settings["rounds"]):
            raise ValueError("admission is beyond the training budget")
        schedule.append((start, tuple(allowed[k] for k in keys)))
        previous = set(keys)
    if previous != allowed.keys():
        raise ValueError("final admission must cover the entire training set")
    return tuple(schedule)


def admitted_tracks(schedule, round_index, default):
    if round_index < 1:
        raise ValueError("DAgger rounds are one-based")
    selected = default
    for start, tracks in schedule:
        if round_index < start:
            break
        selected = tracks
    return tuple(selected)
