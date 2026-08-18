from .nodes import KKSam3ImageSegmentation

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override


class KKSam3Extension(ComfyExtension):
    """Register the standalone KK SAM3 image-segmentation node."""

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [KKSam3ImageSegmentation]


async def comfy_entrypoint() -> KKSam3Extension:
    return KKSam3Extension()


__all__ = []
