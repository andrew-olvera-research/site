"""Experimental forward-only, fixed-reduction BF16 GEMM for precision diagnosis.

Not a training implementation: no backward is registered. Production never
imports this module. Fixed tile/reduction choices isolate cuBLAS batch drift.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _linear(X,W,B,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
            BM:tl.constexpr=32,BN:tl.constexpr=64,BK:tl.constexpr=32):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for start in range(tl.cdiv(K,BK)):
        k=start*BK+kk
        x=tl.load(X+rows[:,None]*K+k[None,:],(rows[:,None]<M)&(k[None,:]<K),0).to(tl.bfloat16)
        w=tl.load(W+cols[None,:]*K+k[:,None],(cols[None,:]<N)&(k[:,None]<K),0).to(tl.bfloat16)
        acc=tl.dot(x,w,acc)
    bias=tl.load(B+cols,cols<N,0).to(tl.bfloat16).to(tl.float32)
    tl.store(Y+rows[:,None]*N+cols[None,:],(acc+bias[None,:]).to(tl.bfloat16),
             (rows[:,None]<M)&(cols[None,:]<N))


def forward(value,weight,bias):
    value=value.contiguous()
    n,k=weight.shape
    out=torch.empty((*value.shape[:-1],n),device=value.device,dtype=torch.bfloat16)
    m=value.numel()//k
    _linear[(triton.cdiv(m,32),triton.cdiv(n,64))](value,weight,bias,out,m,n,k)
    return out


def install(backbone):
    for module in backbone.modules():
        if isinstance(module,torch.nn.Linear):
            assert module.bias is not None
            def call(value,module=module):
                return forward(value,module.weight,module.bias)
            module.forward=call
