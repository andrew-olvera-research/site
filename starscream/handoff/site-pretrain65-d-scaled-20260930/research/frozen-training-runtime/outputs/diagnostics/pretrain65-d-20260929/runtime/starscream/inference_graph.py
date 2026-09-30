"""Bounded, inference-only CUDA graph replay for deterministic CTBR actors.

Uses the existing eager kernels, not reduced precision or a different network.
Static outputs are cloned before returning: subsequent replays cannot mutate
saved actions. Optimizer steps / load_state_dict(assign=False) update the same
parameter storage. Recreate this wrapper after changing device/dtype/storage.
"""
import numpy as np
import torch


class PPOHostInputTransfer:
    """Pack FP32 window inputs into one pinned H2D transfer, without new math.

    Returned views are scratch space, valid until the next call. The window
    collector consumes them and synchronizes its packed result before reuse.
    The copy itself is synchronous so pinned memory cannot be reused in flight.
    """
    def __init__(self, device):
        self.device = torch.device(device)
        self.entries = {}

    def transfer(self, history, speed, critic_input, offsets):
        arrays = (history, speed, critic_input, offsets)
        if any(a.dtype != np.float32 for a in arrays):
            raise ValueError('PPO fused inputs require FP32 arrays')
        if any(a.shape[0] != history.shape[0] for a in arrays):
            raise ValueError('PPO fused inputs require matching batch sizes')
        shapes = tuple(a.shape for a in arrays)
        if shapes not in self.entries:
            sizes = [a.size for a in arrays]
            host = torch.empty(sum(sizes), dtype=torch.float32,
                               pin_memory=self.device.type == 'cuda')
            device = torch.empty_like(host, device=self.device)
            host_views, device_views = [], []
            start = 0
            for shape, size in zip(shapes, sizes):
                host_views.append(host[start:start+size].numpy().reshape(shape))
                device_views.append(device[start:start+size].view(shape))
                start += size
            self.entries[shapes] = host, device, host_views, tuple(device_views)
        host, device, host_views, device_views = self.entries[shapes]
        for destination, source in zip(host_views, arrays):
            np.copyto(destination, source)
        device.copy_(host)
        return device_views


def inference_callable(policy, settings, max_batch):
    if not settings.get('cuda_graph_policy_inference', False):
        return policy
    if settings.get('compile_policy_inference', False):
        raise ValueError('choose eager CUDA graph inference OR compiled inference, not both')
    return PolicyInferenceGraphs(policy, max_batch)


class PolicyInferenceGraphs:
    def __init__(self, policy, max_batch):
        if (policy.action_head_type != 'mlp'
                or getattr(policy, 'action_mixture_mode_head', None) is not None):
            raise ValueError('inference graphs currently require a deterministic MLP action head')
        if max_batch < 1:
            raise ValueError('max_batch must be positive')
        self.policy = policy
        self.max_batch = int(max_batch)
        self.entries = {}

    @torch.no_grad()
    def __call__(self, history, speed):
        if self.policy.training:
            raise ValueError('inference graph replay requires policy.eval()')
        if history.device.type != 'cuda' or speed.device != history.device:
            raise ValueError('inference graphs require CUDA history and speed on one device')
        batch = len(history)
        if batch < 1 or batch > self.max_batch or speed.shape != (batch,):
            raise ValueError('inference graph batch exceeds its bounded cache contract')
        signature = (tuple(history.shape[1:]), history.dtype, speed.dtype, history.device)
        if self.entries and signature != self.signature:
            raise ValueError('history/device/dtype changed: recreate inference graph cache')
        self.signature = signature
        if batch not in self.entries:
            static_history, static_speed = history.clone(), speed.clone()
            current = torch.cuda.current_stream(history.device)
            stream = torch.cuda.Stream(device=history.device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.policy(static_history, static_speed)
            current.wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = self.policy(static_history, static_speed)
            current.wait_stream(stream)
            self.entries[batch] = (graph, static_history, static_speed, output)
        graph, static_history, static_speed, output = self.entries[batch]
        static_history.copy_(history)
        static_speed.copy_(speed)
        graph.replay()
        return output.clone()


class PPOInferenceGraphs:
    """Graph deterministic actor/critic work only; sample outside the graph.

    FP32 raw actions/log probabilities and the existing RNG order are untouched.
    Cache bounded by collector batch size; recreate on device/storage changes.
    """
    def __init__(self, policy, max_batch):
        if policy.action_head_type != 'mlp':
            raise ValueError('PPO graphs require a direct Gaussian actor')
        if max_batch < 1:
            raise ValueError('max_batch must be positive')
        self.policy = policy
        self.max_batch = int(max_batch)
        self.entries = {}
        self.critic = None

    @torch.no_grad()
    def __call__(self, history, speed, critic_input, critic):
        if self.policy.training or critic.training:
            raise ValueError('PPO graph requires eval-mode actor and critic')
        batch = len(history)
        if (not 1 <= batch <= self.max_batch or history.device.type != 'cuda'
                or speed.device != history.device or critic_input.device != history.device
                or speed.shape != (batch,) or len(critic_input) != batch):
            raise ValueError('invalid PPO graph batch/device')
        signature = tuple((tuple(x.shape[1:]), x.dtype, x.device) for x in (history,speed,critic_input))
        if self.entries and (self.critic is not critic or signature != self.signature):
            raise ValueError('PPO graph input/critic changed: recreate collector')
        self.critic = critic; self.signature = signature
        if batch not in self.entries:
            inputs = tuple(x.clone() for x in (history,speed,critic_input))
            def forward():
                distribution = self.policy.distribution(inputs[0], inputs[1])
                return distribution.location, distribution.log_std, critic(inputs[2])
            current = torch.cuda.current_stream(history.device)
            stream = torch.cuda.Stream(device=history.device); stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(3):forward()
            current.wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):outputs = forward()
            current.wait_stream(stream)
            self.entries[batch] = graph, inputs, outputs
        graph, inputs, outputs = self.entries[batch]
        for destination, source in zip(inputs,(history,speed,critic_input)):
            destination.copy_(source)
        graph.replay()
        return tuple(x.clone() for x in outputs)


class HostInferenceTicket:
    """Own a pinned output until its event completes and an owned copy is taken."""
    def __init__(self, owner, event, output):
        self.owner, self.event, self.output = owner, event, output
        self.value = None

    def ready(self):
        return self.value is not None or self.event.query()

    def result(self):
        if self.value is None:
            self.event.synchronize()
            self.value = self.output.copy()
            self.owner._pending_ticket = None
        return self.value


class HostPolicyInferenceGraphs(PolicyInferenceGraphs):
    """Capture the entire CPU -> deterministic actor -> CPU transaction.

    Each shape owns pinned host buffers and a single contiguous device input.
    H2D, forward and D2H are nodes in ONE graph, avoiding pageable allocations,
    separate speed transfers and intermediate GPU clones on every control tick.
    No precision, padding, kernel, sampling or observation changes. Single-owner
    transactions can block or return an event ticket; consume its owned output
    before reusing pinned buffers.
    Recreate after parameter storage/device/dtype changes, like the tensor API.
    """

    @torch.no_grad()
    def predict_numpy(self, history, speed, *, asynchronous=False):
        if getattr(self, '_pending_ticket', None) is not None:
            raise RuntimeError('Consume pending inference before reusing pinned buffers')
        if self.policy.training:
            raise ValueError('host inference graph requires policy.eval()')
        history, speed = np.asarray(history), np.asarray(speed)
        batch = len(history)
        if (history.ndim != 3 or history.dtype != np.float32
                or speed.dtype != np.float32 or speed.shape != (batch,)
                or not 1 <= batch <= self.max_batch):
            raise ValueError('host graph requires FP32 history[B,H,D] and speed[B]')
        device = next(self.policy.parameters()).device
        if device.type != 'cuda':
            raise ValueError('host inference graph requires a CUDA actor')
        signature = (tuple(history.shape[1:]), device)
        if self.entries and signature != self.signature:
            raise ValueError('history/device changed: recreate host graph cache')
        self.signature = signature
        if batch not in self.entries:
            host = torch.empty(history.size + batch, pin_memory=True)
            host_history = host[:history.size].view(history.shape).numpy()
            host_speed = host[history.size:].numpy()
            np.copyto(host_history, history)
            np.copyto(host_speed, speed)
            packed = torch.empty_like(host, device=device)
            h = packed[:history.size].view(history.shape)
            s = packed[history.size:]
            current = torch.cuda.current_stream(device)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                packed.copy_(host, non_blocking=True)
                for _ in range(3):
                    output = self.policy(h, s).float()
            stream.synchronize()
            host_output = torch.empty(output.shape, dtype=torch.float32, pin_memory=True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                packed.copy_(host, non_blocking=True)
                output = self.policy(h, s).float()
                host_output.copy_(output, non_blocking=True)
            self.entries[batch] = (graph, stream, host, packed, host_history,
                                   host_speed, host_output, host_output.numpy(), output)
        graph, stream, host, packed, h_cpu, s_cpu, host_out, out_cpu, output = self.entries[batch]
        np.copyto(h_cpu, history)
        np.copyto(s_cpu, speed)
        # Replay on the caller's stream: optimizer/rollback ordering is natural,
        # without allocating a cross-stream event at every tiny inference tick.
        graph.replay()
        if asynchronous:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(device))
            ticket = HostInferenceTicket(self, event, out_cpu)
            self._pending_ticket = ticket
            return ticket
        torch.cuda.current_stream(device).synchronize()
        return out_cpu.copy()
