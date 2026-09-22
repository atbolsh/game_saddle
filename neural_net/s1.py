"""S1: a ResNet-18 classifier over four primitives.

The trunk is the stock ResNet-18 (no global pool, no 1000-way ``fc``).
Every stage map is queried with the target point and the four pooled
vectors are concatenated into the decision layer. On a 768 frame those
maps are stride 4, 8, 16, and 32. The eye is about four cells wide on
the stride-8 map and about one cell on the stride-32 map; both are
inputs. ``s`` is not an input. The target is ``(v + v_bar) / 2``.

``init_from_imagenet`` is the only ImageNet entry point. Checkpoints are
a plain ``state_dict`` via :meth:`save` / :meth:`load_weights`.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet18

from neural_net.oracle import ACTIONS

#: Stage output channels, layer1 through layer4.
_STAGE_CHANNELS = (64, 128, 256, 512)
_QUERY_DIM = 128


class S1(nn.Module):
    """Four-way controller. ``forward`` returns logits of shape ``[B, 4]``."""

    def __init__(self) -> None:
        super().__init__()
        trunk = resnet18(weights=None)
        self.conv1 = trunk.conv1
        self.bn1 = trunk.bn1
        self.relu = trunk.relu
        self.maxpool = trunk.maxpool
        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3
        self.layer4 = trunk.layer4

        self.query = nn.Sequential(
            nn.Linear(2, _QUERY_DIM),
            nn.ReLU(),
            nn.Linear(_QUERY_DIM, _QUERY_DIM),
        )
        self.scale_query = nn.ModuleList(
            nn.Linear(_QUERY_DIM, channels) for channels in _STAGE_CHANNELS
        )
        # +2 coordinate channels so two identical white cells stay distinct.
        self.scale_key = nn.ModuleList(
            nn.Conv2d(channels + 2, channels, kernel_size=1)
            for channels in _STAGE_CHANNELS
        )
        self.classifier = nn.Linear(sum(_STAGE_CHANNELS), len(ACTIONS))

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("imagenet_mean", mean, persistent=False)
        self.register_buffer("imagenet_std", std, persistent=False)

    def init_from_imagenet(self) -> int:
        """Copy ImageNet trunk weights. The query and the classifier stay
        as they are. Returns the number of tensors copied."""
        from torchvision.models import ResNet18_Weights

        ref = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        ref_sd = ref.state_dict()
        own = self.state_dict()
        copied = 0
        missed: list[str] = []
        for key, value in ref_sd.items():
            if key.startswith("fc."):
                continue
            if key not in own:
                missed.append(key)
                continue
            if own[key].shape != value.shape:
                raise RuntimeError(
                    f"init_from_imagenet: shape mismatch for {key}: "
                    f"S1 {tuple(own[key].shape)} vs ImageNet {tuple(value.shape)}"
                )
            own[key] = value.detach().clone()
            copied += 1
        if missed:
            raise RuntimeError(
                "init_from_imagenet: trunk keys missing from S1: "
                + ", ".join(missed)
            )
        if copied == 0:
            raise RuntimeError("init_from_imagenet: copied no tensors")
        self.load_state_dict(own)
        return copied

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str | Path) -> None:
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
        self.load_state_dict(state)

    def forward(self, images: torch.Tensor, target_xy: torch.Tensor) -> torch.Tensor:
        """``images`` is ``[B,3,H,W]`` uint8 or float in ``[0, 1]``.
        ``target_xy`` is ``[B, 2]`` in world coordinates (it may leave
        the unit square). Returns ``[B, 4]`` logits, order
        ``noop, CLOCK, ANTICLOCK, FORWARD``."""
        if images.dim() != 4 or images.shape[1] != 3:
            raise ValueError(
                f"S1 images must be [B,3,H,W], got {tuple(images.shape)}"
            )
        if target_xy.dim() != 2 or target_xy.shape[1] != 2:
            raise ValueError(
                f"S1 target_xy must be [B,2], got {tuple(target_xy.shape)}"
            )
        if images.shape[0] != target_xy.shape[0]:
            raise ValueError(
                f"S1 batch mismatch: images {images.shape[0]} vs "
                f"target {target_xy.shape[0]}"
            )
        x = images.float()
        if images.dtype == torch.uint8:
            x = x / 255
        x = (x - self.imagenet_mean) / self.imagenet_std

        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        m1 = self.layer1(x)
        m2 = self.layer2(m1)
        m3 = self.layer3(m2)
        m4 = self.layer4(m3)
        query = self.query(target_xy.float())
        pooled = [
            self._pool(feat, query, index)
            for index, feat in enumerate((m1, m2, m3, m4))
        ]
        return self.classifier(torch.cat(pooled, dim=-1))

    def _pool(
        self, feat: torch.Tensor, query: torch.Tensor, index: int
    ) -> torch.Tensor:
        batch, _channels, height, width = feat.shape
        # Row 0 is the top of the y-up export, which is world y = 1.
        ys = torch.linspace(1, 0, height, device=feat.device, dtype=feat.dtype)
        xs = torch.linspace(0, 1, width, device=feat.device, dtype=feat.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack((grid_x, grid_y), dim=0)
        coords = coords.unsqueeze(0).expand(batch, -1, -1, -1)
        keys = self.scale_key[index](torch.cat((feat, coords), dim=1))
        q = self.scale_query[index](query)
        scores = torch.einsum("bc,bchw->bhw", q, keys).float()
        scores = scores / math.sqrt(q.shape[-1])
        weights = scores.flatten(1).softmax(dim=-1).view(batch, 1, height, width)
        return (keys.float() * weights).sum(dim=(2, 3))
