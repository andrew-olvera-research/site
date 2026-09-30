"""Small learned local tools. All chart normalizers fit TRAIN parents only."""
import torch
from torch import nn
from torch.nn import functional as F


class Normalizer(nn.Module):
    def __init__(self,mean,scale):
        super().__init__();self.register_buffer('mean',mean);self.register_buffer('scale',scale.clamp_min(.05))
    def forward(self,x):return (x-self.mean)/self.scale


class RBFPrecision(nn.Module):
    """Positive RBF precision, Eq.11 style (Latent Space Oddity).

    Frozen training centers/bandwidths, positive learned weights, small precision
    floor. Far from ALL centers, variance tends to 1/floor, unlike an arbitrary
    extrapolating variance MLP. This is residual uncertainty, not flyability.
    """
    def __init__(self,centers,bandwidth,output_dim,floor=1e-4):
        super().__init__();self.register_buffer('centers',centers)
        self.register_buffer('bandwidth',bandwidth.clamp_min(.1))
        self.weight=nn.Parameter(torch.full((len(centers),output_dim),-2.))
        self.floor=floor
    def forward(self,x):
        d=(x[:,None]-self.centers[None]).square().sum(-1)
        basis=torch.exp(-d/(2*self.bandwidth.square()))
        return self.floor+basis@F.softplus(self.weight)
    def sigma(self,x):return self(x).rsqrt()


class SwiBlock(nn.Module):
    def __init__(self,width):
        super().__init__();self.norm=nn.RMSNorm(width);self.up=nn.Linear(width,4*width);self.down=nn.Linear(2*width,width)
    def forward(self,x):
        a,b=self.up(self.norm(x)).chunk(2,-1)
        return x+self.down(F.silu(a)*b)/2


class LocalTraversal(nn.Module):
    """Predict normalized physical decoder response to a local latent move.

    Zero move -> exactly zero output by construction. Count is explicit, not
    treated as a continuous direction across a gate insertion/deletion.
    """
    def __init__(self,input_dim,output_dim,width=256):
        super().__init__();self.net=nn.Sequential(nn.Linear(input_dim,width),SwiBlock(width),SwiBlock(width),nn.RMSNorm(width),nn.Linear(width,output_dim))
    def forward(self,x,radius):return self.net(x)*radius[:,None]


def gaussian_precision_nll(precision,residual,mask):
    values=.5*(precision*residual.square()-precision.log())
    return (values*mask).sum()/mask.sum().clamp_min(1)


def pullback_from_jacobians(mean_jacobian,sigma_jacobian=None,ridge=1e-6):
    """Expected diagonal-Gaussian pullback; squared physical feature units.

    Fixed count/context. Constant ridge is numerical regularization, reported
    separately; it does not prove immersion or intrinsic task distance.
    """
    g=mean_jacobian.T@mean_jacobian
    if sigma_jacobian is not None:g=g+sigma_jacobian.T@sigma_jacobian
    return g+ridge*torch.eye(g.shape[0],device=g.device,dtype=g.dtype)
