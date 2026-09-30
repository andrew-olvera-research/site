"""Round-local CUDA capture for deterministic, fixed-shape DAgger updates."""
from __future__ import annotations

import torch
import copy


def restore_optimizer_snapshot(optimizer, snapshot):
    """Restore without allowing later updates to mutate same-device safe tensors."""
    optimizer.load_state_dict(copy.deepcopy(snapshot))


class CapturedDaggerUpdate:
    """Capture a complete update; warmup must not count as optimization.

    The caller owns the loss/clip/step function and creates a fresh instance
    whenever learning rate, objective, batch shape or trainable parameters change.
    Returned metrics have independent storage, safe for deferred logging.
    """

    def __init__(self, policy, optimizer, update):
        if not isinstance(optimizer, torch.optim.AdamW) or not all(
            group.get('fused', False) for group in optimizer.param_groups
        ):
            raise ValueError('DAgger update capture requires fused AdamW')
        self.policy, self.optimizer, self.update = policy, optimizer, update
        self.graph = None

    def _capture(self, batch):
        if not batch or any(value.device.type != 'cuda' for value in batch):
            raise ValueError('DAgger update capture requires CUDA tensors')
        self.inputs = tuple(value.clone() for value in batch)
        saved_model = [(value, value.clone()) for value in self.policy.state_dict().values()]
        saved_optimizer = {
            parameter: {key: value.clone() for key, value in state.items()}
            for parameter, state in self.optimizer.state.items()
        }
        flags = [group.get('capturable', False) for group in self.optimizer.param_groups]
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state()
        try:
            for group in self.optimizer.param_groups:
                group['capturable'] = True
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.update(self.inputs)
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.loss, self.pieces = self.update(self.inputs)
        finally:
            # Copy into captured storage. load_state_dict would replace optimizer
            # tensors and leave the graph updating detached, stale state.
            with torch.no_grad():
                for value, original in saved_model:
                    value.copy_(original)
                for parameter, state in self.optimizer.state.items():
                    for key, value in state.items():
                        original = saved_optimizer.get(parameter, {}).get(key)
                        if original is None:
                            value.zero_()  # Fresh Adam moments/step created by warmup.
                        else:
                            value.copy_(original)
            for group, flag in zip(self.optimizer.param_groups, flags):
                group['capturable'] = flag
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng)

    def __call__(self, batch):
        if self.graph is None:
            self._capture(batch)
        if len(batch) != len(self.inputs) or any(
            source.shape != target.shape or source.dtype != target.dtype or source.device != target.device
            for source, target in zip(batch, self.inputs)
        ):
            raise ValueError('DAgger captured update batch contract changed')
        for source, target in zip(batch, self.inputs):
            target.copy_(source)
        self.graph.replay()
        # One materialization for all scalars, rather than one clone kernel per
        # metric. Subsequent replays must not overwrite earlier logged values.
        packed = torch.stack([self.loss.detach(), *(v.detach() for v in self.pieces.values())])
        return packed[0], dict(zip(self.pieces, packed[1:].unbind()))
