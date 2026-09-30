"""Local ordered-bank control; feasibility must be established independently."""

def ordered_geometry_choice(source_success,frontier_success,candidates,*,
                            source_floor=.65,frontier_floor=.25,mastery=.5,
                            minimum_gate_fraction=.5,minimum_episodes=16):
    """Candidates are ordered nearest-to-farthest: (name,success,fraction,n).

    Choose the farthest ready candidate; zero successes are never admitted.
    This estimates competence, not target learning utility.
    """
    if source_success<source_floor:return 'rehearse',None
    if frontier_success<frontier_floor:return 'retreat',None
    if frontier_success<mastery:return 'hold',None
    ready=[name for name,success,fraction,n in candidates
           if n>=minimum_episodes and success>0 and success>=frontier_floor
           and fraction>=minimum_gate_fraction]
    return ('expand',ready[-1]) if ready else ('hold',None)
