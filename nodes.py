"""Standalone KK SAM3 image segmentation node.

Derived from the image segmentation node in ComfyUI-Easy-Sam3 (Apache-2.0).
This version deliberately pads variable per-frame detections before returning
auxiliary outputs, so a video batch may contain 3 detections in one frame and
0/1/2 detections in another without failing at torch.stack().
"""

from __future__ import annotations

import logging
from contextlib import nullcontext

import torch
import comfy.model_management as mm
import comfy.utils
from comfy_api.latest import io

from .utils import (
    join_image_with_alpha,
    masks_to_tensor,
    parse_bbox,
    parse_points,
    pil_to_tensor,
    tensor_to_pil,
)


logger = logging.getLogger("KK-SAM3")


def _pad_first_dimension(tensor: torch.Tensor, size: int) -> torch.Tensor:
    """Pad the object dimension with zero-valued missing detections."""
    tensor = tensor.detach().float().cpu()
    if tensor.shape[0] >= size:
        return tensor[:size]
    padding = torch.zeros((size - tensor.shape[0], *tensor.shape[1:]), dtype=tensor.dtype)
    return torch.cat((tensor, padding), dim=0)


def _align_detection_metadata(boxes, scores):
    """Pad only lightweight box/score metadata to a common object count."""
    object_slots = max(
        1,
        max((item.shape[0] for item in boxes), default=0),
        max((item.shape[0] for item in scores), default=0),
    )

    padded_boxes = []
    padded_scores = []
    for frame_boxes, frame_scores in zip(boxes, scores):
        if frame_boxes.ndim != 2 or frame_boxes.shape[-1] != 4:
            frame_boxes = torch.zeros((0, 4), dtype=torch.float32)
        if frame_scores.ndim != 1:
            frame_scores = frame_scores.reshape(-1)
        padded_boxes.append(_pad_first_dimension(frame_boxes, object_slots))
        padded_scores.append(_pad_first_dimension(frame_scores, object_slots))

    box_batch = torch.stack(padded_boxes, dim=0)
    score_batch = torch.stack(padded_scores, dim=0)
    return box_batch.cpu().tolist(), score_batch.cpu().tolist()


class KKSam3ImageSegmentation(io.ComfyNode):
    """SAM3 image segmentation with video-safe variable detection handling."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KK SAM3",
            display_name="KK SAM3",
            category="KK/SAM3",
            description="SAM3 image segmentation that supports different detection counts per frame.",
            inputs=[
                io.Custom(io_type="EASY_SAM3_MODEL").Input(
                    "sam3_model",
                    display_name="SAM3 Model",
                    tooltip="Compatible SAM3 image model, for example from Load SAM3 Model.",
                ),
                io.Image.Input("images", display_name="图像"),
                io.String.Input(
                    "prompt",
                    display_name="提示词",
                    default="",
                    multiline=True,
                    tooltip="Comma-separated text prompts, such as face, top, jeans.",
                ),
                io.Float.Input(
                    "threshold",
                    display_name="置信度",
                    default=0.40,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                ),
                io.Boolean.Input(
                    "keep_model_loaded",
                    display_name="保持模型加载",
                    default=False,
                ),
                io.Combo.Input(
                    "add_background",
                    display_name="添加背景",
                    options=["none", "black", "white", "grey"],
                    default="none",
                ),
                io.String.Input(
                    "coordinates_positive",
                    display_name="coordinates_positive",
                    optional=True,
                    force_input=True,
                ),
                io.String.Input(
                    "coordinates_negative",
                    display_name="coordinates_negative",
                    optional=True,
                    force_input=True,
                ),
                io.BBOX.Input("bboxes", display_name="边界框", optional=True),
                io.Mask.Input("mask", display_name="重叠遮罩", optional=True),
                io.Int.Input(
                    "detection_limit",
                    display_name="检测对象限制",
                    default=-1,
                    min=-1,
                    max=1000,
                    tooltip="-1 means unlimited. Missing detections are padded automatically.",
                ),
                io.Boolean.Input(
                    "generate_segmented_images",
                    display_name="生成分割预览图（高内存）",
                    default=False,
                    optional=True,
                    tooltip="关闭时图像输出直接透传输入，避免视频批次生成巨大的RGBA副本。",
                ),
                io.Boolean.Input(
                    "output_individual_masks",
                    display_name="输出单独对象遮罩（高内存）",
                    default=False,
                    optional=True,
                    tooltip="关闭时对象遮罩输出复用主遮罩；开启后输出所有对象遮罩。",
                ),
            ],
            outputs=[
                io.Mask.Output("output_masks", display_name="遮罩"),
                io.Image.Output("output_images", display_name="图像"),
                io.Mask.Output("obj_masks", display_name="对象遮罩"),
                io.BBOX.Output("boxes", display_name="边界框"),
                io.Float.Output("scores", display_name="置信度分数"),
            ],
        )

    @classmethod
    def execute(
        cls,
        sam3_model,
        images,
        prompt,
        threshold=0.4,
        keep_model_loaded=False,
        add_background="none",
        detection_limit=-1,
        coordinates_positive=None,
        coordinates_negative=None,
        bboxes=None,
        mask=None,
        generate_segmented_images=False,
        output_individual_masks=False,
    ) -> io.NodeOutput:
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4:
            raise ValueError(f"KK SAM3 expects IMAGE [B,H,W,C], got {tuple(images.shape)}")
        batch_size, height, width, _ = images.shape

        processor = sam3_model.get("processor")
        model = sam3_model.get("model")
        device = sam3_model.get("device", torch.device("cpu"))
        dtype = sam3_model.get("dtype", torch.float32)
        segmentor = sam3_model.get("segmentor", "image")
        if model is None or processor is None or segmentor != "image":
            raise ValueError("KK SAM3 requires a SAM3 model loaded in image mode.")
        if not prompt.strip() and all(value is None for value in (
            coordinates_positive, coordinates_negative, bboxes, mask
        )):
            raise ValueError("Provide at least one text, point, box or mask prompt.")

        positive_points, positive_count, _ = parse_points(coordinates_positive, images.shape)
        negative_points, negative_count, _ = parse_points(coordinates_negative, images.shape)
        points = None
        point_labels = None
        if positive_points or negative_points:
            points = (positive_points or []) + (negative_points or [])
            point_labels = [1] * positive_count + [0] * negative_count

        bounding_boxes, bounding_count = parse_bbox(bboxes, images.shape)
        bounding_labels = [True] * bounding_count if bounding_boxes else None

        processor.set_confidence_threshold(threshold)
        offload_device = mm.unet_offload_device()
        model.to(device)
        input_mask = mask.to(device) if mask is not None else None

        # Preallocate the one output that video compositing actually needs.
        # This avoids retaining a list and allocating another full copy at
        # torch.stack() time.
        combined_masks = torch.empty((batch_size, height, width), dtype=torch.float32)
        segmented_images = None
        if generate_segmented_images:
            image_channels = 4 if add_background == "none" else 3
            segmented_images = torch.empty(
                (batch_size, height, width, image_channels), dtype=torch.float32
            )
        output_raw_masks = [] if output_individual_masks else None
        output_boxes = []
        output_scores = []
        progress = comfy.utils.ProgressBar(batch_size)
        text_prompts = [item.strip() for item in prompt.split(",") if item.strip()]

        autocast = (
            torch.autocast(mm.get_autocast_device(device), dtype=dtype)
            if not mm.is_device_mps(device)
            else nullcontext()
        )
        try:
            with torch.inference_mode(), autocast:
                for index in range(batch_size):
                    # Convert only the current frame. Converting the complete
                    # video to a PIL list would keep another full video copy in
                    # system memory for the duration of inference.
                    pil_image = tensor_to_pil(images[index : index + 1])[0]
                    state = processor.set_image(pil_image)
                    all_masks = []
                    all_boxes = []
                    all_scores = []

                    if text_prompts:
                        # The image backbone is by far the expensive part.  Its
                        # state is independent of the text prompt, and
                        # Sam3Processor.set_text_prompt replaces only the text
                        # features/results. Reuse it for every comma-separated
                        # target instead of encoding the same frame N times.
                        for text_prompt in text_prompts:
                            prompt_state = processor.set_text_prompt(text_prompt, state)
                            prompt_masks = prompt_state.get("masks")
                            if prompt_masks is not None and len(prompt_masks) > 0:
                                all_masks.append(prompt_masks)
                                all_boxes.append(prompt_state["boxes"])
                                all_scores.append(prompt_state["scores"])
                    else:
                        if points:
                            state = processor.add_point_prompt(points, point_labels, state)
                        if bounding_boxes:
                            state = processor.add_boxes_prompts(bounding_boxes, bounding_labels, state)
                        if input_mask is not None:
                            state = processor.add_mask_prompt(input_mask, state)
                        prompt_masks = state.get("masks")
                        if prompt_masks is not None and len(prompt_masks) > 0:
                            all_masks.append(prompt_masks)
                            all_boxes.append(state["boxes"])
                            all_scores.append(state["scores"])

                    if all_masks:
                        frame_masks = torch.cat(all_masks, dim=0)
                        frame_boxes = torch.cat(all_boxes, dim=0)
                        frame_scores = torch.cat(all_scores, dim=0)
                        order = torch.argsort(frame_scores, descending=True)
                        frame_masks = frame_masks[order]
                        frame_boxes = frame_boxes[order]
                        frame_scores = frame_scores[order]
                        if detection_limit >= 0:
                            frame_masks = frame_masks[:detection_limit]
                            frame_boxes = frame_boxes[:detection_limit]
                            frame_scores = frame_scores[:detection_limit]
                    else:
                        frame_masks = torch.zeros((0, height, width), dtype=torch.float32)
                        frame_boxes = torch.zeros((0, 4), dtype=torch.float32)
                        frame_scores = torch.zeros((0,), dtype=torch.float32)

                    raw_masks = masks_to_tensor(frame_masks)
                    if raw_masks is None or raw_masks.shape[0] == 0:
                        raw_masks = torch.zeros((0, height, width), dtype=torch.float32)
                        combined_mask = torch.zeros((height, width), dtype=torch.float32)
                    else:
                        combined_mask = (raw_masks.sum(dim=0) > 0).float()

                    combined_masks[index].copy_(combined_mask)
                    if output_individual_masks:
                        output_raw_masks.append(raw_masks)
                    output_boxes.append(frame_boxes.detach().float().cpu().reshape(-1, 4))
                    output_scores.append(frame_scores.detach().float().cpu().reshape(-1))

                    if generate_segmented_images:
                        rgb_image = pil_to_tensor(pil_image)
                        rgba_image = join_image_with_alpha(rgb_image, combined_mask.unsqueeze(0))
                        if add_background == "none":
                            segmented_images[index].copy_(rgba_image.squeeze(0))
                        else:
                            background_value = {"black": 0.0, "white": 1.0, "grey": 0.5}[add_background]
                            rgb = rgba_image[..., :3]
                            alpha = rgba_image[..., 3:4]
                            background = torch.full_like(rgb, background_value)
                            segmented_images[index].copy_(
                                (rgb * alpha + background * (1.0 - alpha)).squeeze(0)
                            )

                    progress.update_absolute(index + 1, batch_size)

            if not generate_segmented_images:
                # Reuse the caller's tensor without creating an RGBA video copy.
                segmented_images = images
            if output_individual_masks:
                detected_masks = [item for item in output_raw_masks if item.shape[0] > 0]
                object_masks = (
                    torch.cat(detected_masks, dim=0)
                    if detected_masks
                    else torch.zeros((1, height, width), dtype=torch.float32)
                )
            else:
                # Reuse the combined mask tensor; no additional allocation.
                object_masks = combined_masks
            boxes_list, scores_list = _align_detection_metadata(output_boxes, output_scores)
            logger.info(
                "KK SAM3 processed %d frame(s); variable detections padded safely.",
                batch_size,
            )
            return io.NodeOutput(
                combined_masks,
                segmented_images,
                object_masks,
                boxes_list,
                scores_list,
            )
        finally:
            if not keep_model_loaded:
                model.to(offload_device)
                mm.soft_empty_cache()
