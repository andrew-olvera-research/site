"""Runtime-only compilation: checkpoint keys and architecture stay unchanged."""
import torch


def configure_policy_acceleration(policy, settings):
    if getattr(policy, 'unified_route_transformer', False):
        policy.step_embedding.current_route_projection_only = bool(
            settings.get('current_route_projection_only', False))
        route_training = bool(settings.get('ppo_current_route_projection_training', False))
        if route_training and not getattr(policy, 'exact_likelihood_mode', False):
            raise ValueError('training route projection shortcut is validated for exact PPO only')
        policy.step_embedding.current_route_projection_training = route_training
        if route_training:
            policy.step_embedding.current_route_projection_only = True
    compile_updates = bool(settings.get('compile_policy_backbone', False))
    if settings.get('compile_ppo_actor', False) or settings.get('compile_dagger_loss', False):
        compile_updates = False  # Compile the complete distribution below, without nested traces.
    compile_inference = bool(settings.get('compile_policy_inference', False))
    if not (compile_updates or compile_inference):
        return False
    if getattr(policy, '_starscream_backbone_compiled', False):
        return True
    if not getattr(policy, 'unified_route_transformer', False):
        raise ValueError('backbone compilation currently validated only for unified route actor')
    if next(policy.parameters()).device.type != 'cuda':
        raise ValueError('compiled policy training requires CUDA')
    # Bound CPU RAM/process fanout during compilation on the shared 16GiB host.
    import torch._inductor.config as config
    threads = int(settings.get('compile_threads', 1))
    if threads < 1:
        raise ValueError('compile_threads must be positive')
    config.compile_threads = threads
    # CUDAGraph output lifetime needs different contracts for rollout storage and
    # repeated encode calls. Disable it: fuse kernels without aliasing replay.
    if compile_updates:
        policy.step_embedding.compile(fullgraph=True, dynamic=False,
            options={'triton.cudagraphs':False})
    # Async rollout batches vary from 1..N, with different grad/autocast modes.
    # Compiling every inference shape exhausts Dynamo's per-code cache before
    # the first BF16 update. Retain eager inference and compile only updates.
    module = policy.step_embedding
    compiled_call = module._compiled_call_impl if compile_updates else module._call_impl
    # Optional, separately traced dynamic microbatch graph. Batch-one can have
    # a second specialization; 2..N share a graph. No padding, quantization,
    # autocast change, stochastic-action change or CUDA graph output aliasing.
    inference_call = (
        torch.compile(module._call_impl, fullgraph=True, dynamic=True,
                      options={'triton.cudagraphs': False})
        if compile_inference else module._call_impl
    )

    def update_only(*args, **kwargs):
        if torch.is_grad_enabled():
            return compiled_call(*args, **kwargs)
        return inference_call(*args, **kwargs)

    module._compiled_call_impl = update_only
    policy._starscream_backbone_compiled = True
    return True
