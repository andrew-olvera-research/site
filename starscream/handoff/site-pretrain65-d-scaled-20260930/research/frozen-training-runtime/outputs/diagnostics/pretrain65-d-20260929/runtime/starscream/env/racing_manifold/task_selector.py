"""Checkpointable policy-dependent task selector for an empirical curriculum.

Online responses are observational associations under a changing policy, not
causal estimates of isolated task transfer. Hard competence gates precede all
learned scores. This module does not change PPO rewards or likelihood math.
"""
from __future__ import annotations
import numpy as np


def completion_gate_count(track, stage_target, *, enabled=False):
    goal=min(int(stage_target),len(track.gates))
    override=(track.metadata or {}).get("curriculum_completion_gates")
    if enabled and override is not None:
        if int(override)!=override or not 1<=int(override)<=goal:
            raise ValueError("invalid curriculum completion override")
        goal=int(override)
    return goal


class PolicyDependentTaskSelector:
    def __init__(self, raw=None):
        self.raw = dict(raw or {})
        self.observations = {}
        self.responses = []
        self.previous = None
        self.last_switch = 0
        self.cursor = 0
        self.last_scores = []

    @staticmethod
    def lower_bound(success, episodes, z=1.0):
        n=max(int(episodes),1); p=float(success)
        return (p+z*z/(2*n)-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)

    def probes(self, names, frontier, limit):
        names = sorted(set(names))
        if not names:
            return ()
        chosen = [frontier] if frontier in names else []
        for i in range(len(names)):
            name=names[(self.cursor+i)%len(names)]
            if name not in chosen:
                chosen.append(name)
            if len(chosen)>=limit:
                break
        return tuple(chosen)

    def observe(self, metrics, gate_counts, cycle):
        floor=float(self.raw.get("success_floor",.2))
        minimum=int(self.raw.get("minimum_episodes",24))
        for name,count in gate_counts.items():
            prefix=f"track/{name}/"
            if prefix+"full_course_success" not in metrics:
                continue
            p=float(metrics[prefix+"full_course_success"])
            n=int(metrics.get(prefix+"episodes",0))
            if not (0<=p<=1 and n>=minimum):
                continue
            depth=float(np.clip(metrics.get(prefix+"mean_gates",0)/count,0,1))
            previous=self.observations.get(name,{})
            passed=p>=floor and self.lower_bound(p,n)>=float(self.raw.get("minimum_success_lcb",.10)) and depth>=float(self.raw.get("minimum_gate_fraction",.55))
            self.observations[name]=dict(success=p,depth=depth,n=n,cycle=cycle,
                passes=previous.get("passes",0)+1 if passed else 0,
                progress=.5*previous.get("progress",0)+.5*(depth-previous.get("depth",depth)))
        self.cursor += max(1,int(self.raw.get("probe_rotation_stride",3)))

    @staticmethod
    def features(record, observation):
        lift=float(record.get("vertical_lift_m",record.get("metadata",{}).get("vertical_lift_m",2.5)))
        width=float(record.get("width_scale",record.get("metadata",{}).get("width_scale",2.5)))
        target_side=bool(record.get("target_side",False))
        proximity=float(np.clip(1-.5*(lift/2.5+(width-1)/1.5),0,1)) if target_side else 0.
        goal=float(record.get("curriculum_completion_gates",5))/5
        return np.array([1.,lift/2.5,(width-1)/1.5,float(target_side),
                         observation.get("success",0),observation.get("depth",0),proximity,goal])

    def record_response(self, reporting, *, target, source, frontier, record, steps):
        key=f"track/{target}/"
        if not reporting or key+"mean_gates" not in reporting:
            return
        # Report suite is immutable; p3/depth supply signal before rare completions.
        score=.5*float(reporting.get(key+"p3",0))+.3*float(reporting[key+"mean_gates"])/5+.2*float(reporting.get(key+"full_course_success",0))
        retention=float(reporting.get(f"track/{source}/full_course_success",0))
        if self.previous and steps>self.previous["steps"]:
            delta=(score-self.previous["score"])+.5*min(0.,retention-self.previous["retention"])
            self.responses.append(dict(x=self.previous["features"],
                y=float(np.clip(delta*100000/(steps-self.previous["steps"]),-1,1)),
                frontier=self.previous["frontier"],start_steps=self.previous["steps"],end_steps=steps))
            self.responses=self.responses[-128:]
        self.previous=dict(score=score,retention=retention,steps=steps,frontier=frontier,
            features=self.features(record,self.observations.get(frontier,{})).tolist())

    def select(self, records, *, active, frontier, cycle):
        self.last_scores=[]
        if cycle-self.last_switch<int(self.raw.get("minimum_tenure_cycles",40)):
            return None
        eligible=[]
        for name,record in records.items():
            o=self.observations.get(name,{})
            if o.get("passes",0)<int(self.raw.get("required_passes",2)) or cycle-o.get("cycle",-10000)>int(self.raw.get("maximum_probe_age_cycles",80)):
                continue
            x=self.features(record,o)
            # Cold-start prior: reachable tasks nearer the actual target, with
            # useful remaining learning headroom. No cyclic geometric ray.
            value=.25*x[6]+.2*x[3]+.25*x[7]+.1*4*o["success"]*(1-o["success"])+.1*o["progress"]
            mean=uncertainty=0.
            if len(self.responses)>=int(self.raw.get("minimum_response_samples",8)):
                a=np.array([r["x"] for r in self.responses]); y=np.array([r["y"] for r in self.responses])
                covariance=np.linalg.inv(a.T@a+np.eye(a.shape[1])*4.)
                mean=float(x@covariance@a.T@y)
                uncertainty=float(np.sqrt(x@covariance@x))
                value+=float(np.clip(mean-.02*uncertainty,-.1,.1))
            if name in active and name!=frontier:
                value-=.02
            self.last_scores.append(dict(name=name,score=value,response_mean=mean,response_uncertainty=uncertainty))
            eligible.append((value,name))
        if not eligible:
            return None
        selected=max(eligible)[1]
        return None if selected==frontier else selected

    def state_dict(self):
        return dict(observations=self.observations,responses=self.responses,previous=self.previous,
                    last_switch=self.last_switch,cursor=self.cursor,last_scores=self.last_scores)

    def load_state_dict(self,state):
        for key in self.state_dict():
            if key in state:
                setattr(self,key,state[key])
