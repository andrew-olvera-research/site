"""SkyDreamer-style GateNet and adapters for public checkpoint variants."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import inspect
from pathlib import Path
from typing import Callable

import numpy as np
import yaml


def _torch_modules():
    import torch.nn as nn
    return nn


try:
    import torch
    import torch.nn as nn
except ImportError:  # Collection without the optional data/model stack.
    torch = None
    nn = None


if nn is not None:
    class DoubleConv(nn.Module):
        def __init__(self, input_channels: int, output_channels: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)


    class GateNet(nn.Module):
        """Compact five-level U-Net with multi-scale gate-mask supervision."""

        def __init__(self, input_channels: int = 3, base_channels: int = 16) -> None:
            super().__init__()
            channels = [base_channels, base_channels * 2, base_channels * 4,
                        base_channels * 8, base_channels * 8]
            self.inc = DoubleConv(input_channels, channels[0])
            self.down = nn.ModuleList(
                nn.Sequential(nn.MaxPool2d(2), DoubleConv(channels[i], channels[i + 1]))
                for i in range(4)
            )
            decoder_channels = [base_channels * 4, base_channels * 2, base_channels, base_channels]
            input_channels_by_stage = [channels[4], *decoder_channels[:-1]]
            skip_channels = list(reversed(channels[:4]))
            self.up = nn.ModuleList(
                nn.Sequential(
                    nn.ConvTranspose2d(input_channel, output_channel, 2, stride=2),
                    nn.BatchNorm2d(output_channel),
                    nn.ReLU(inplace=True),
                )
                for input_channel, output_channel in zip(input_channels_by_stage, decoder_channels)
            )
            self.fuse = nn.ModuleList(
                DoubleConv(output_channel + skip_channel, output_channel)
                for output_channel, skip_channel in zip(decoder_channels, skip_channels)
            )
            self.decoder_outputs = nn.ModuleList(
                nn.Conv2d(channel, 1, 1) for channel in decoder_channels
            )
            self.outputs = nn.ModuleList(nn.Conv2d(channel, 1, 1) for channel in channels)
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def forward_multiscale(self, x):
            import torch.nn.functional as F
            skips = [self.inc(x)]
            for down in self.down:
                skips.append(down(skips[-1]))
            decoded = skips[-1]
            predictions = [self.outputs[-1](decoded)]
            for stage, (up, fuse) in enumerate(zip(self.up, self.fuse)):
                decoded = up(decoded)
                skip = skips[-2 - stage]
                if decoded.shape[-2:] != skip.shape[-2:]:
                    decoded = F.interpolate(decoded, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                decoded = fuse(torch.cat([skip, decoded], dim=1))
                predictions.append(self.decoder_outputs[stage](decoded))
            return list(reversed(predictions))  # highest resolution first

        def forward(self, x):
            return self.forward_multiscale(x)[0]


    class EfficientDecoderBlock(nn.Module):
        def __init__(self, input_channels: int, skip_channels: int, output_channels: int) -> None:
            super().__init__()
            self.reduce = nn.Conv2d(input_channels, output_channels, 1, bias=False)
            self.fuse = DoubleConv(output_channels + skip_channels, output_channels)

        def forward(self, x, skip):
            import torch.nn.functional as F
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            return self.fuse(torch.cat([self.reduce(x), skip], dim=1))


    class EfficientGateNet(nn.Module):
        """ImageNet-pretrained EfficientNet-B0 plus the five-scale GateNet decoder."""

        def __init__(self, *, pretrained: bool = True) -> None:
            super().__init__()
            from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.encoder = efficientnet_b0(weights=weights).features
            self.decoder = nn.ModuleList(
                [
                    EfficientDecoderBlock(320, 112, 160),
                    EfficientDecoderBlock(160, 40, 96),
                    EfficientDecoderBlock(96, 24, 64),
                    EfficientDecoderBlock(64, 16, 32),
                ]
            )
            self.full_head = nn.Sequential(DoubleConv(32, 32), nn.Conv2d(32, 1, 1))
            self.auxiliary_heads = nn.ModuleList(
                [nn.Conv2d(64, 1, 1), nn.Conv2d(96, 1, 1),
                 nn.Conv2d(160, 1, 1), nn.Conv2d(320, 1, 1)]
            )
            self.register_buffer(
                "image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False
            )
            self.register_buffer(
                "image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False
            )

        def forward_multiscale(self, image):
            import torch.nn.functional as F
            input_size = image.shape[-2:]
            x = (image - self.image_mean) / self.image_std
            skips = []
            for index, layer in enumerate(self.encoder):
                x = layer(x)
                if index in {1, 2, 3, 5}:
                    skips.append(x)
                if index == 7:
                    break
            bottleneck = x
            decoded = bottleneck
            decoded_features = []
            for block, skip in zip(self.decoder, reversed(skips)):
                decoded = block(decoded, skip)
                decoded_features.append(decoded)
            high = F.interpolate(decoded_features[-1], size=input_size, mode="bilinear", align_corners=False)
            outputs = [self.full_head(high)]
            outputs.extend(
                head(feature)
                for head, feature in zip(self.auxiliary_heads, reversed(decoded_features[:-1]))
            )
            outputs.append(self.auxiliary_heads[-1](bottleneck))
            return outputs

        def forward(self, image):
            return self.forward_multiscale(image)[0]


def build_gatenet(architecture: str = "efficientnet_b0", *, pretrained: bool = True, base_channels: int = 16):
    if nn is None:
        raise RuntimeError("GateNet requires PyTorch")
    if architecture == "paper":
        return GateNet(base_channels=base_channels)
    if architecture == "efficientnet_b0":
        return EfficientGateNet(pretrained=pretrained)
    raise ValueError(f"unknown GateNet architecture: {architecture}")


def import_factory(spec: str) -> Callable:
    """Resolve ``package.module:factory`` without assuming a model architecture."""

    module_name, separator, name = spec.partition(":")
    if not separator or not module_name or not name:
        raise ValueError("factory must use the form 'package.module:function'")
    factory = getattr(importlib.import_module(module_name), name)
    if not callable(factory):
        raise TypeError(f"{spec!r} does not resolve to a callable")
    return factory


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    camera_matrix: np.ndarray
    distortion: np.ndarray
    model: str = "fisheye"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CameraCalibration":
        with Path(path).open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        matrix = np.asarray(data["camera_matrix"], dtype=np.float64).reshape(3, 3)
        distortion = np.asarray(data["distortion_coefficients"], dtype=np.float64).reshape(-1)
        model = str(data.get("model", "fisheye")).lower()
        if model not in {"fisheye", "pinhole"}:
            raise ValueError("camera model must be fisheye or pinhole")
        return cls(matrix, distortion, model)


class CameraRectifier:
    """Map a calibrated camera to SkyDreamer's nominal pinhole intrinsics."""

    def __init__(self, calibration: CameraCalibration, output_size: tuple[int, int]) -> None:
        self.calibration = calibration
        self.output_size = output_size
        self._maps = None

    def _build_maps(self):
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("camera rectification requires OpenCV Python bindings") from error
        width, height = self.output_size
        nominal = np.asarray(
            [[25.0 / 64.0 * width, 0, 0.5 * width],
             [0, 25.0 / 64.0 * height, 0.5 * height],
             [0, 0, 1]],
            dtype=np.float64,
        )
        if self.calibration.model == "fisheye":
            self._maps = cv2.fisheye.initUndistortRectifyMap(
                self.calibration.camera_matrix,
                self.calibration.distortion,
                np.eye(3),
                nominal,
                (width, height),
                cv2.CV_32FC1,
            )
        else:
            self._maps = cv2.initUndistortRectifyMap(
                self.calibration.camera_matrix,
                self.calibration.distortion,
                np.eye(3),
                nominal,
                (width, height),
                cv2.CV_32FC1,
            )

    def __call__(self, image: np.ndarray, *, is_mask: bool = False) -> np.ndarray:
        import cv2
        if self._maps is None:
            self._build_maps()
        interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        return cv2.remap(image, *self._maps, interpolation=interpolation, borderMode=cv2.BORDER_CONSTANT)


@dataclass(slots=True)
class GateNetAdapter:
    """Normalize arbitrary PyTorch segmentation modules to an HxW probability map."""

    model: object
    output_size: tuple[int, int] = (160, 128)
    input_size: tuple[int, int] = (384, 384)
    device: str = "cuda"
    preprocessor: Callable[[np.ndarray], np.ndarray] | None = None

    @classmethod
    def from_factory(
        cls,
        factory_spec: str,
        checkpoint: str | Path,
        *,
        output_size: tuple[int, int] = (160, 128),
        input_size: tuple[int, int] = (384, 384),
        device: str = "cuda",
    ) -> "GateNetAdapter":
        import torch

        factory = import_factory(factory_spec)
        checkpoint = Path(checkpoint)
        signature = inspect.signature(factory)
        kwargs = {}
        if "checkpoint" in signature.parameters:
            kwargs["checkpoint"] = checkpoint
        if "device" in signature.parameters:
            kwargs["device"] = device
        model = factory(**kwargs)
        if "checkpoint" not in kwargs:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
            model.load_state_dict(state)
        model.to(device).eval()
        return cls(model=model, output_size=output_size, input_size=input_size, device=device)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        output_size: tuple[int, int] = (160, 128),
        device: str = "cuda",
        base_channels: int | None = None,
        input_size: tuple[int, int] | None = None,
    ) -> "GateNetAdapter":
        if torch is None:
            raise RuntimeError("GateNet requires PyTorch")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        metadata = payload if isinstance(payload, dict) else {}
        base_channels = int(base_channels or metadata.get("base_channels", 16))
        input_size = tuple(input_size or metadata.get("input_size", (384, 384)))
        architecture = str(metadata.get("architecture", "paper"))
        model = build_gatenet(architecture, pretrained=False, base_channels=base_channels)
        state = payload.get("state_dict", payload.get("model", payload)) if isinstance(payload, dict) else payload
        state = {key.removeprefix("module.").removeprefix("model."): value for key, value in state.items()}
        model.load_state_dict(state)
        model.to(device).eval()
        return cls(model=model, output_size=output_size, input_size=input_size, device=device)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        if self.preprocessor is not None:
            rgb = self.preprocessor(rgb)
        image = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device)
        image = image.permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
        image = F.interpolate(image, size=self.input_size, mode="bilinear", align_corners=False)
        with torch.inference_mode():
            output = self.model(image)
            if isinstance(output, dict):
                output = output.get("out", output.get("logits"))
            if isinstance(output, (tuple, list)):
                output = output[0]
            if output is None:
                raise ValueError("GateNet model did not return logits")
            if output.ndim == 3:
                output = output[:, None]
            if output.ndim != 4:
                raise ValueError("GateNet output must have shape (B,C,H,W) or (B,H,W)")
            if output.shape[1] > 1:
                probability = output.softmax(1)[:, 1:2]
            else:
                probability = output.sigmoid()
            width, height = self.output_size
            probability = F.interpolate(
                probability, size=(height, width), mode="bilinear", align_corners=False
            )
        return probability[0, 0].float().cpu().numpy()
