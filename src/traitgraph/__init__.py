"""TraitEvidenceGraphAgent v2."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import TraitEvidenceGraphPipeline

__all__ = ["TraitEvidenceGraphPipeline"]


def __getattr__(name: str):
    """Keep lightweight submodules independent from the full pipeline."""
    if name == "TraitEvidenceGraphPipeline":
        from .pipeline import TraitEvidenceGraphPipeline

        return TraitEvidenceGraphPipeline
    raise AttributeError(name)
