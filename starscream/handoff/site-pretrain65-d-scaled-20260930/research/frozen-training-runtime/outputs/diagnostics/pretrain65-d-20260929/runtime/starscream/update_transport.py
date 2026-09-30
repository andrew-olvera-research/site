"""Bounded update transport and deferred logging; no sampling or loss changes."""
from __future__ import annotations

import numpy as np
import torch


class UpdateBatchTransfer:
    """Pack heterogeneous arrays into one pinned H2D copy on the current stream.

    A single CPU staging buffer is reused only after its copy completes. Device
    storage is newly allocated per batch, so backward never sees overwritten data.
    Alignment preserves native dtypes (including bool/int64) without conversions.
    """

    def __init__(self, device, packed=False):
        self.device = torch.device(device)
        self.packed = bool(packed) and self.device.type == "cuda"
        self.host = None
        self.done = None

    def __call__(self, arrays):
        if not self.packed:
            return tuple(torch.from_numpy(x).to(self.device) for x in arrays)
        layout, offset = [], 0
        for array in arrays:
            if array.dtype.hasobject:
                raise TypeError("update batches must contain numeric arrays")
            offset = (offset + 15) // 16 * 16
            layout.append((offset, array.nbytes, array.shape,
                           torch.from_numpy(np.empty(0, array.dtype)).dtype))
            offset += array.nbytes
        if self.done is not None:
            self.done.synchronize()
        if self.host is None or self.host.numel() < offset:
            self.host = torch.empty(offset, dtype=torch.uint8, pin_memory=True)
            self.host_numpy = self.host.numpy()
        for array, (start, size, shape, dtype) in zip(arrays, layout):
            np.copyto(self.host_numpy[start:start+size].view(array.dtype).reshape(shape), array)
        device_bytes = self.host[:offset].to(self.device, non_blocking=True)
        self.done = torch.cuda.Event()
        self.done.record(torch.cuda.current_stream(self.device))
        return tuple(device_bytes[start:start+size].view(dtype).view(shape)
                     for start, size, shape, dtype in layout)


def scalar_lists_to_host(values):
    """One transfer for detached metric lists; preserve each scalar's FP64 value.

    Logging reductions still happen in NumPy as before, not in lower precision on
    the device. Never use this to defer a safety check or an optimizer decision.
    """
    positions = [(name, i, v) for name, rows in values.items()
                 for i, v in enumerate(rows) if isinstance(v, torch.Tensor)]
    if not positions:
        return values
    device = positions[0][2].device
    if any(v.device != device for _, _, v in positions):
        raise ValueError("deferred metrics must share a device")
    packed = torch.stack([v.detach().reshape(()) for _, _, v in positions]).cpu().tolist()
    for (name, i, _), value in zip(positions, packed):
        values[name][i] = value
    return values
