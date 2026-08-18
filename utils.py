"""Small utility subset required by the standalone KK SAM3 node.

The conversion and prompt parsing behavior is based on ComfyUI-Easy-Sam3,
licensed under Apache-2.0.  Only helpers used by the image segmentation node
are kept here so this plugin does not import the original plugin at runtime.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
import torch
from PIL import Image


def tensor_to_pil(images: torch.Tensor) -> list[Image.Image]:
    if not isinstance(images, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(images)}")

    images = images.detach().cpu()
    if images.ndim == 3:
        images = images.unsqueeze(0)
    elif images.ndim == 2:
        images = images.unsqueeze(0).unsqueeze(-1)
    if images.ndim != 4:
        raise ValueError(f"Expected IMAGE tensor [B,H,W,C], got {tuple(images.shape)}")

    if images.numel() and images.max() <= 1.0:
        images = images * 255.0
    images = images.clamp(0, 255).byte()

    result: list[Image.Image] = []
    for image in images:
        array = image.numpy()
        channels = array.shape[-1]
        if channels == 1:
            result.append(Image.fromarray(array.squeeze(-1), mode="L"))
        elif channels == 3:
            result.append(Image.fromarray(array, mode="RGB"))
        elif channels == 4:
            result.append(Image.fromarray(array, mode="RGBA"))
        else:
            raise ValueError(f"Unsupported channel count: {channels}")
    return result


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    if not isinstance(image, Image.Image):
        raise ValueError(f"Expected PIL.Image, got {type(image)}")
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def masks_to_tensor(masks) -> Optional[torch.Tensor]:
    if masks is None:
        return None
    if isinstance(masks, np.ndarray):
        masks = torch.from_numpy(masks)
    if not isinstance(masks, torch.Tensor):
        return None

    masks = masks.detach().float().cpu()
    if masks.numel() and masks.max() > 1.0:
        masks = masks / 255.0
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    elif masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks.squeeze(1)
    elif masks.ndim == 4 and masks.shape[-1] == 1:
        masks = masks.squeeze(-1)
    if masks.ndim != 3:
        raise ValueError(f"Expected masks [N,H,W], got {tuple(masks.shape)}")
    return masks


def join_image_with_alpha(image: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    alpha = torch.nn.functional.interpolate(
        alpha.reshape(-1, 1, alpha.shape[-2], alpha.shape[-1]),
        size=image.shape[1:3],
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    count = min(image.shape[0], alpha.shape[0])
    return torch.cat((image[:count, :, :, :3], alpha[:count].unsqueeze(-1)), dim=-1)


def parse_points(value, image_shape=None):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None, 0, []

    try:
        data = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid point JSON: {exc}") from exc

    if isinstance(data, dict) and "points" in data:
        points = data.get("points") or []
        return (points or None), len(points), []
    if not isinstance(data, list):
        raise ValueError("Point coordinates must be a JSON list or an object containing 'points'.")

    height = image_shape[1] if image_shape is not None else None
    width = image_shape[2] if image_shape is not None else None
    points = []
    errors = []
    for index, item in enumerate(data):
        try:
            x, y = float(item["x"]), float(item["y"])
            if x < 0 or y < 0:
                raise ValueError("coordinates must be non-negative")
            if width is not None and height is not None:
                if x >= width or y >= height:
                    raise ValueError(f"point is outside {width}x{height}")
                x, y = x / width, y / height
            points.append([x, y])
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"Point {index}: {exc}")
    return (points or None), len(points), errors


def parse_bbox(value, image_shape=None):
    if value is None:
        return None, 0
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid bbox JSON: {exc}") from exc

    if isinstance(value, dict) and "boxes" in value:
        boxes = value.get("boxes") or []
        return (boxes or None), len(boxes)

    if isinstance(value, dict):
        values = [value]
    elif isinstance(value, (list, tuple)) and len(value) == 4 and all(
        isinstance(item, (int, float)) for item in value
    ):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        raise ValueError(f"Unsupported bbox type: {type(value).__name__}")

    boxes = []
    for item in values:
        if isinstance(item, dict):
            coords = [item["startX"], item["startY"], item["endX"], item["endY"]]
        else:
            coords = list(item)
        if len(coords) != 4:
            raise ValueError(f"A bbox needs four values, got {coords}")
        x1, y1, x2, y2 = (float(number) for number in coords)
        if x2 < x1 or y2 < y1:
            x2, y2 = x1 + x2, y1 + y2
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bbox coordinates: {[x1, y1, x2, y2]}")

        if image_shape is not None:
            height, width = image_shape[1], image_shape[2]
            nx1, ny1, nx2, ny2 = x1 / width, y1 / height, x2 / width, y2 / height
            boxes.append([(nx1 + nx2) / 2, (ny1 + ny2) / 2, nx2 - nx1, ny2 - ny1])
        else:
            boxes.append([x1, y1, x2, y2])
    return (boxes or None), len(boxes)
