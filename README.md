# ComfyUI-KK-SAM3

Standalone `KK SAM3` image-segmentation node for ComfyUI.

## Purpose

This plugin contains only the image segmentation node and its small utility
subset. It does not import `ComfyUI-Easy-Sam3` at runtime. The input accepts a
compatible `EASY_SAM3_MODEL`, such as the output of the existing **Load SAM3
Model** node in image mode.

Unlike the upstream image node, this implementation supports a different
number of detections in every frame. Missing objects are padded with zero masks,
boxes and scores for auxiliary outputs, while the primary combined mask always
contains exactly one mask per input frame.

For comma-separated multi-object prompts, each frame's expensive image-backbone
features are computed once and reused across all text prompts. Ten targets still
require ten text/grounding passes, but do not cause ten repeated image encodes.

The node defaults to a low-memory video mode. It always produces the complete
combined mask batch, but passes the original IMAGE batch through the image
output and reuses the combined masks for the object-mask output. Enable
`generate_segmented_images` or `output_individual_masks` only when those
auxiliary high-memory outputs are actually required.

## Node

- Name: `KK SAM3`
- Category: `KK/SAM3`
- Compatible model input type: `EASY_SAM3_MODEL`

For a video batch, connect the complete IMAGE batch directly to `KK SAM3`.
The primary mask output is already a normal `[frames, height, width]` MASK
batch, so no list-to-batch bridge is required.

## Attribution

The segmentation flow and selected utility behavior are derived from
`ComfyUI-Easy-Sam3`, which is distributed under the Apache License 2.0.
