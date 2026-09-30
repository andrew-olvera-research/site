"""Small encoder-only corruption for denoising; clean reconstruction targets.

Do not perturb poses as new feasible courses, reorder gates, or erase gate
semantics. Physical targets/context stay unchanged, including floor clearance.
"""
import torch


def corrupt_encoder(gates,position_std_m=.02,log_aperture_std=.005,probability=.5):
    if position_std_m<0 or log_aperture_std<0 or not 0<=probability<=1:
        raise ValueError('invalid encoder corruption')
    if not probability or not (position_std_m or log_aperture_std):return gates
    result=gates.clone()
    use=(torch.rand(len(gates),1,1,device=gates.device)<probability).to(gates)
    result[...,:3]+=torch.randn_like(gates[...,:3])*position_std_m*use
    result[...,9:11]+=torch.randn_like(gates[...,9:11])*log_aperture_std*use
    return result
