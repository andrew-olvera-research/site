"""Standalone route6 vision student; no privileged teacher in the deployable model."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.spatial.transform import Rotation

from starscream.sequence_architecture import AdaRMSNormSwiGLUBlock, SwiGLUProjection, SwiGLUActionMLP


class KinematicEKF:
    """9D error-state EKF on the SAME noisy VIO stream as the raw arm.

    State is position, velocity, SO(3); rates are measured body rates. This is
    a simulated VIO-fusion baseline, not an IMU/VIO front end or hardware EKF.
    Gate-frame changes reset from the new-frame measurement (no truth input).
    """
    def __init__(self, dt: float):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("positive finite dt required")
        self.dt = dt
        self.gate = None

    @staticmethod
    def rotation(columns):
        a, b = np.asarray(columns, float).reshape(2, 3)
        a = a / max(np.linalg.norm(a), 1e-8)
        b = b - a * np.dot(a, b)
        b = b / max(np.linalg.norm(b), 1e-8)
        return np.stack([a, b, np.cross(a, b)], axis=1)

    def update(self, raw, rates, gate):
        raw = np.asarray(raw, float)
        if raw.shape != (32,) or not np.isfinite(raw).all():
            raise ValueError("finite raw32 required")
        measurement = np.r_[raw[:3] * 20, raw[3:6] * 30]
        rotation = self.rotation(raw[6:12])
        noise = np.maximum(raw[24:27], [0.15, 0.25, 0.025])
        covariance = np.diag(np.repeat(noise ** 2, 3))
        if self.gate != gate:
            self.x, self.r, self.p = measurement.copy(), rotation, covariance.copy()
            self.gate = gate
        else:
            dt = self.dt
            self.x[:3] += self.x[3:] * dt
            increment = Rotation.from_rotvec(np.asarray(rates) * dt).as_matrix()
            self.r = self.r @ increment
            transition = np.eye(9)
            transition[:3, 3:6] = np.eye(3) * dt
            transition[6:, 6:] = increment.T
            self.p = transition @ self.p @ transition.T + np.diag(
                np.repeat([0.02**2, 3.0**2, 0.1**2], 3)) * dt
            # Held stale measurements must never be repeatedly assimilated.
            if raw[28] > 0.5 and raw[30] == 0:
                innovation = np.r_[measurement - self.x,
                    Rotation.from_matrix(self.r.T @ rotation).as_rotvec()]
                gain = np.linalg.solve((self.p + covariance).T, self.p.T).T
                correction = gain @ innovation
                self.x += correction[:6]
                self.r = self.r @ Rotation.from_rotvec(correction[6:]).as_matrix()
                residual = np.eye(9) - gain
                self.p = residual @ self.p @ residual.T + gain @ covariance @ gain.T
                self.p = (self.p + self.p.T) * 0.5
        result = raw.copy()
        result[:6] = self.x / np.r_[np.full(3, 20), np.full(3, 30)]
        result[6:12] = self.r[:, :2].T.reshape(-1)
        result[24:27] = np.sqrt(np.diag(self.p).reshape(3, 3).mean(1))
        return result.astype(np.float32)


def augment_masks(masks: torch.Tensor) -> torch.Tensor:
    """Sequence-consistent small shifts, erosion/dilation, and missing patches.

    Applied to student inputs only, during updates; no flips or camera rotations
    that would contradict unchanged state/route measurements.
    """
    b, t, c, h, w = masks.shape
    affine = masks.new_zeros(b, 2, 3)
    affine[:, 0, 0] = affine[:, 1, 1] = 1
    shift = torch.randint(-2, 3, (b, 2), device=masks.device)
    affine[:, :, 2] = shift * masks.new_tensor([2/w, 2/h])
    grid = F.affine_grid(affine.repeat_interleave(t, 0), (b*t,c,h,w), align_corners=False)
    value = F.grid_sample(masks.flatten(0,1), grid, mode='nearest', align_corners=False)
    morphology = torch.randint(0,3,(b,1,1,1),device=masks.device).repeat_interleave(t,0)
    value = torch.where(morphology==1, F.max_pool2d(value,3,1,1),
                        torch.where(morphology==2, -F.max_pool2d(-value,3,1,1), value))
    keep = (torch.rand(b,1,8,10,device=masks.device)>.03).float()
    return value.reshape(b,t,c,h,w) * F.interpolate(keep,size=(h,w),mode='nearest')[:,None]


class VisionStudent(nn.Module):
    """Teacher-style causal RMSNorm/SwiGLU trunk with observation patch tokens.

    Per step: patches (2D position), state, previous-command/timing. Temporal
    position is shared by ALL tokens from that observation, independent of patch
    count. Latest route6 (with approach features), speed, action query follow.
    """
    def __init__(self, width=256, depth=3, heads=8, feedforward=504, history=3,
                 explicit_estimation=False, conditioning=True,
                 memory_steps=0, memory_stride=2, memory_width=64):
        super().__init__()
        self.history, self.width = history, width
        self.explicit_estimation, self.conditioning = explicit_estimation, conditioning
        self.memory_steps,self.memory_stride=int(memory_steps),int(memory_stride)
        if self.memory_steps<0 or self.memory_stride<1:
            raise ValueError('invalid causal memory length/stride')
        self.memory_encoder=nn.GRU(21,memory_width,batch_first=True) if memory_steps else None
        self.memory_projector=nn.Linear(memory_width,width) if memory_steps else None
        self.vision_encoder = nn.Sequential(nn.Conv2d(1, 32, 5, 2, 2), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.SiLU(), nn.AdaptiveAvgPool2d((4, 5)))
        self.vision_projector = nn.Linear(64, width)
        self.camera_timing = nn.Linear(2, width, bias=False)
        self.state = SwiGLUProjection(39, width, input_rms_norm=False)
        self.control = SwiGLUProjection(6, width, input_rms_norm=False)
        self.route = SwiGLUProjection(22, width, input_rms_norm=False)
        self.speed = SwiGLUProjection(1, width, input_rms_norm=False)
        self.time_position = nn.Parameter(torch.randn(1, history, 1, width) * .02)
        self.row_position = nn.Parameter(torch.randn(1, 1, 4, 1, width) * .02)
        self.column_position = nn.Parameter(torch.randn(1, 1, 1, 5, width) * .02)
        self.types = nn.Parameter(torch.randn(1, 1, 3, width) * .02)
        self.route_position = nn.Parameter(torch.randn(1, 6, width) * .02)
        self.query = nn.Parameter(torch.randn(1, 1, width) * .02)
        self.blocks = nn.ModuleList([AdaRMSNormSwiGLUBlock(width, heads, feedforward,
            adaln_zero=False) for _ in range(depth)])
        self.norm = nn.RMSNorm(width)
        self.action = SwiGLUActionMLP(width, width, 3, 4)
        # Count all auxiliary heads in the 3M budget, even though deployment
        # only requires action. No hidden teacher or latent alignment projector.
        self.dynamics = nn.Sequential(nn.Linear(width + 4, width), nn.SiLU(), nn.Linear(width, 19))
        # State-conditioned FiLM is identity at initialization. The state
        # stream itself is never normalized/modulated by these branches.
        self.vision_film = nn.Linear(width, 2*width) if conditioning else None
        self.route_film = nn.Linear(width, 2*width) if conditioning else None
        for film in (self.vision_film, self.route_film):
            if film is not None:
                nn.init.zeros_(film.weight)
                nn.init.zeros_(film.bias)
        self.estimate = nn.Linear(width, 19) if explicit_estimation else None
        if self.estimate is not None:
            nn.init.zeros_(self.estimate.weight)
            nn.init.zeros_(self.estimate.bias)

    def parameter_counts(self):
        vision = sum(p.numel() for module in (self.vision_encoder, self.vision_projector) for p in module.parameters())
        total = sum(p.numel() for p in self.parameters())
        return dict(core=total - vision, vision=vision, total=total)

    def tokens(self, features, masks, speed, memory=None):
        b, t, d = features.shape
        if d != 125 or t != self.history or masks.shape != (b, t, 1, 128, 160):
            raise ValueError("expected raw125 route6/camera-timing history and [B,T,1,128,160] masks")
        patch = self.vision_encoder(masks.flatten(0, 1)).permute(0, 2, 3, 1)
        patch = self.vision_projector(patch).reshape(b, t, 4, 5, self.width)
        patch = (patch + self.row_position + self.column_position).flatten(2, 3)
        patch = patch + self.types[:, :, :1] + self.camera_timing(features[...,123:125])[:, :, None]
        state_embedding = self.state(features[..., :39])
        if self.vision_film is not None:
            scale, shift = self.vision_film(state_embedding).chunk(2, -1)
            patch = patch * (1+scale[:, :, None]) + shift[:, :, None]
        state = state_embedding[:, :, None] + self.types[:, :, 1:2]
        control = self.control(features[..., 117:123])[:, :, None] + self.types[:, :, 2:3]
        history = (torch.cat([patch, state, control], 2) + self.time_position).flatten(1, 2)
        route = features[:, -1, 39:117].reshape(b, 6, 13)
        # Approach geometry uses noisy/raw or EKF state, NEVER privileged state.
        p, v = features[:, -1, :3] * 20, features[:, -1, 3:6] * 30
        centers = route[..., :3] * 20
        n, u = route[..., 3:6], route[..., 6:9]
        rotation = torch.stack([n, torch.cross(u, n, dim=-1), u], -1)
        local_p = torch.einsum("bni,bnij->bnj", p[:, None] - centers, rotation) / 5
        local_v = torch.einsum("bi,bnij->bnj", v, rotation) / 10
        following = torch.cat([centers[:, 1:], centers[:, -1:]], 1) - centers
        local_exit = torch.einsum("bni,bnij->bnj", following, rotation) / 5
        route = self.route(torch.cat([route, local_p, local_v, local_exit], -1)) + self.route_position
        if self.route_film is not None:
            scale, shift = self.route_film(state_embedding[:, -1]).chunk(2, -1)
            route = route * (1+scale[:, None]) + shift[:, None]
        pieces=[history]
        if self.memory_encoder is not None:
            if memory is None or memory.shape!=(b,self.memory_steps,21):
                raise ValueError('configured student requires causal body-frame memory')
            _,belief=self.memory_encoder(memory)
            pieces.append(self.memory_projector(belief[-1])[:,None])
        pieces += [route,self.speed(speed[:,None]/20)[:,None],self.query.expand(b,-1,-1)]
        return torch.cat(pieces,1)

    def forward(self, features, masks, speed, executed_action=None, *, return_hidden=False, memory=None):
        tokens = self.tokens(features, masks, speed, memory)
        for block in self.blocks:
            tokens = block(tokens, condition=tokens[:, -1], is_causal=True)
        hidden = self.norm(tokens[:, -1])
        action = self.action(hidden).tanh()
        dynamics_action = action if executed_action is None else executed_action
        result = dict(action=action, dynamics=self.dynamics(torch.cat([hidden, dynamics_action], -1)))
        if return_hidden:
            result['action_readout'] = hidden
        if self.estimate is not None:
            baseline = torch.cat([features[:, -1, :12], features[:, -1, 32:39]], -1)
            bounds = hidden.new_tensor([.075]*3 + [.10]*3 + [.25]*6 + [1.]*3 + [1.5]*4)
            result['estimate'] = baseline + self.estimate(hidden).tanh()*bounds
        return result
