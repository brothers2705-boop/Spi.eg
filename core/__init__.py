"""
VisionForge AI - Core Engine Package
--------------------------------------
Exposes the primary building blocks of the pipeline:

    SemanticMatcher  -> few-shot retrieval over the golden reference dataset
    ArtDirector      -> creative direction / brand-mood reasoning layer
    PromptBuilder    -> syntax translation into MJ v6 / Flux.1 / SDXL

Kept import-light so the package works identically inside Docker,
a bare virtualenv, or a notebook.
"""

from .semantic_matcher import SemanticMatcher
from .art_director import ArtDirector
from .prompt_builder import PromptBuilder
from .seedance_builder import SeedanceBuilder, CAMERA_MOTION_PRESETS
from .database import CampaignDatabase

__all__ = [
    "SemanticMatcher",
    "ArtDirector",
    "PromptBuilder",
    "SeedanceBuilder",
    "CAMERA_MOTION_PRESETS",
    "CampaignDatabase",
]

__version__ = "1.0.0"
