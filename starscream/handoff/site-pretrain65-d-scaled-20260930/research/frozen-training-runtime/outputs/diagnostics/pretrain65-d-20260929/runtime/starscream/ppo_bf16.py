"""Experimental BF16 backbone with batch-independent GEMM reduction order.

Opt-in only. Actor head, Gaussian likelihoods, parameters and optimizer state
remain FP32. Fixed forward tiles avoid cuBLAS selecting a different reduction
for collection and PPO minibatches. Backward uses ordinary BF16 GEMMs.
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
        x=tl.load(X+rows[:,None]*K+k[None,:],(rows[:,None]<M)&(k[None,:]<K),0)
        w=tl.load(W+cols[None,:]*K+k[:,None],(cols[None,:]<N)&(k[:,None]<K),0)
        acc=tl.dot(x,w,acc)
    bias=tl.load(B+cols,cols<N,0).to(tl.float32)
    tl.store(Y+rows[:,None]*N+cols[None,:],(acc+bias[None,:]).to(tl.bfloat16),
             (rows[:,None]<M)&(cols[None,:]<N))


@torch.library.triton_op('starscream::stable_bf16_linear',mutates_args={})
def stable_linear(value:torch.Tensor,weight:torch.Tensor,bias:torch.Tensor)->torch.Tensor:
    m,k=value.shape
    n=weight.shape[0]
    out=torch.empty((m,n),device=value.device,dtype=torch.bfloat16)
    torch.library.wrap_triton(_linear)[(triton.cdiv(m,32),triton.cdiv(n,64))](value,weight,bias,out,m,n,k)
    return out


def _setup(ctx,inputs,output):
    ctx.save_for_backward(inputs[0],inputs[1])


def _backward(ctx,gradient):
    value,weight=ctx.saved_tensors
    with torch.autocast('cuda',enabled=False):
        gradient=gradient.contiguous()
        dx=gradient@weight if ctx.needs_input_grad[0] else None
        dw=gradient.T@value if ctx.needs_input_grad[1] else None
        db=gradient.sum(0) if ctx.needs_input_grad[2] else None
    return dx,dw,db


stable_linear.register_autograd(_backward,setup_context=_setup)


def linear(value,weight,bias):
    original_shape=value.shape
    x=value.reshape(-1,original_shape[-1]).to(torch.bfloat16).contiguous()
    w=weight.to(torch.bfloat16).contiguous()
    b=bias.to(torch.bfloat16).contiguous()
    return stable_linear(x,w,b).reshape(*original_shape[:-1],weight.shape[0])


def configure_stable_bf16(policy):
    if getattr(policy,'_ppo_stable_bf16',False):
        return
    if not getattr(policy,'unified_route_transformer',False):
        raise ValueError('stable BF16 is validated only for unified route actors')
    if next(policy.parameters()).device.type!='cuda':
        raise ValueError('stable BF16 requires CUDA')
    if getattr(policy,'_ppo_actor_compiled',False) or getattr(policy,'_starscream_backbone_compiled',False):
        raise ValueError('stable BF16 must be configured before compilation')
    for module in policy.step_embedding.modules():
        if isinstance(module,torch.nn.Linear):
            if module.bias is None:
                raise ValueError('stable BF16 backbone requires biased Linear layers')
            def call(value,module=module):
                return linear(value,module.weight,module.bias)
            module.forward=call
    original=policy.step_embedding.forward
    def backbone(*args,**kwargs):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            return original(*args,**kwargs).float()
    policy.step_embedding.forward=backbone
    policy._ppo_stable_bf16=True
