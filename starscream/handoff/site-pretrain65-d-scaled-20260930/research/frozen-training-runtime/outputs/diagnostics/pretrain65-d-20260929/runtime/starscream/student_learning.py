"""Small, explicit student ablations; legacy defaults preserve existing recipes."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def make_readout_predictor(student_width,teacher_width,config):
    kind=config.get('readout_predictor','linear')
    if kind=='linear':return nn.Linear(student_width,teacher_width)
    if kind=='mlp':
        return nn.Sequential(nn.Linear(student_width,int(config.get('readout_predictor_width',512))),
            nn.SiLU(),nn.Linear(int(config.get('readout_predictor_width',512)),teacher_width))
    raise ValueError('readout predictor must be linear or mlp')


def calibrate_readout(readouts,track_ids,minimum_std=.1):
    """Equal-course target moments from the permanent training bank only."""
    if minimum_std<=0:raise ValueError('positive normalization floor required')
    means=[];seconds=[]
    for course in np.unique(track_ids):
        x=np.asarray(readouts[track_ids==course],np.float64)
        if not np.isfinite(x).all():raise ValueError('nonfinite teacher readout')
        means.append(x.mean(0));seconds.append(np.square(x).mean(0))
    mean=np.mean(means,0)
    std=np.sqrt(np.maximum(np.mean(seconds,0)-mean**2,minimum_std**2))
    return dict(mean=mean.tolist(),std=std.tolist(),source='permanent-training-round-0000',
                courses=len(means),rows=len(track_ids),minimum_std=minimum_std)


def readout_losses(prediction,target,normalization=None):
    prediction=prediction.float();target=target.detach().float()
    cosine=(1-F.cosine_similarity(prediction,target,dim=-1)).mean()
    regression=cosine.new_zeros(())
    if normalization is not None:
        std=torch.as_tensor(normalization['std'],device=prediction.device,dtype=torch.float32)
        if std.shape!=(target.shape[-1],) or not torch.isfinite(std).all() or (std<=0).any():
            raise ValueError('invalid readout normalization')
        regression=F.smooth_l1_loss((prediction-target)/std,torch.zeros_like(target),beta=.1)
    return cosine,regression


def learning_rate_for_round(config,index):
    base=float(config['learning_rate']);schedule=config.get('learning_rate_schedule')
    if not schedule:return base
    warmup=int(schedule.get('warmup_rounds',2));floor=float(schedule.get('final_fraction',.25))
    if not 0<floor<=1 or not 0<=warmup<config['rounds']:raise ValueError('invalid learning-rate schedule')
    if index<warmup:return base*(index+1)/max(1,warmup)
    progress=(index-warmup)/max(1,config['rounds']-1-warmup)
    return base*(floor+(1-floor)*.5*(1+math.cos(math.pi*min(1,progress))))


def collection_beta(config,index):
    if config.get('permanent_teacher_only',False):
        if index<int(config['permanent_rounds']):return 1.
        index-=int(config['permanent_rounds'])
    return max(float(config.get('beta_min',.05)),1-index/config['beta_decay_rounds'])


def selection_key(evaluation):
    key=tuple(float(v) for v in evaluation.get('selection_key',[evaluation['selection_score']]))
    if not all(math.isfinite(v) for v in key):raise ValueError('nonfinite checkpoint score')
    return key
