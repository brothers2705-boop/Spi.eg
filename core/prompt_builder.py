"""
core/prompt_builder.py
========================
Syntax translation layer. Takes a resolved CreativeConcept (produced by
ArtDirector) and renders it into copy-paste-ready generation prompts for:

    - Midjourney v6      (--ar --style raw --v 6.0 --stylize --chaos)
    - Flux.1              (natural language descriptive paragraph framing)
    - Stable Diffusion XL / SD3 (structured Positive / Negative blocks)

Each engine has fundamentally different prompt grammar, so this module
intentionally keeps three independent render functions rather than one
"universal" template -- that is what actually produces high-quality,
idiomatic output per engine instead of a lowest-common-denominator prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .art_director import CreativeConcept

# --------------------------------------------------------------------------- #
# Shared vocabulary
# --------------------------------------------------------------------------- #

_UNIVERSAL_NEGATIVE_TERMS: List[str] = [
    "blurry", "low quality", "low resolution", "jpeg artifacts", "watermark",
    "text", "logo", "signature", "extra limbs", "deformed", "disfigured",
    "oversaturated", "overexposed", "underexposed", "cartoon", "illustration",
    "3d render", "cgi", "plastic looking", "amateur photography", "stock photo",
    "grainy", "noisy", "duplicate", "cropped badly", "out of frame",
]

_CATEGORY_NEGATIVE_ADDITIONS: Dict[str, List[str]] = {
    "watch": ["wrong time on dial", "warped hands", "melted numerals"],
    "perfume bottle": ["leaking liquid", "cracked glass", "warped label text"],
    "perfume flatlay": ["leaking liquid", "cracked glass", "warped label text"],
    "sports car": ["extra wheels", "warped badge", "distorted proportions"],
    "handbag": ["asymmetrical hardware", "warped stitching"],
    "diamond ring": ["cloudy stone", "asymmetrical band", "warped prongs"],
    "smartphone": ["warped screen", "extra buttons", "distorted logo"],
    "sneaker": ["mismatched shoes", "extra laces", "warped sole"],
    "serum jar": ["warped label text", "leaking product"],
    "sunglasses": ["asymmetrical lenses", "warped frame"],
}


@dataclass
class EnginePrompt:
    engine: str
    primary_text: str
    negative_text: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    full_copy_paste: str = ""


class PromptBuilder:
    """Stateless syntax translator: CreativeConcept -> per-engine prompts."""

    # ------------------------------------------------------------------ #
    # Midjourney v6
    # ------------------------------------------------------------------ #
    @staticmethod
    def build_midjourney(
        concept: CreativeConcept,
        aspect_ratio: Optional[str] = None,
        stylize: Optional[int] = None,
        chaos: Optional[int] = None,
        version: str = "6.0",
        use_style_raw: bool = True,
    ) -> EnginePrompt:
        defaults = concept.mj_defaults or {}
        ar = aspect_ratio or defaults.get("ar", "16:9")
        sty = stylize if stylize is not None else defaults.get("stylize", 300)
        chs = chaos if chaos is not None else defaults.get("chaos", 10)

        palette_desc = ", ".join(concept.color_palette)
        style_tag_desc = ", ".join(concept.style_tags) if concept.style_tags else "commercial advertising photography"

        subject_line = (
            f"{concept.brand_name} {concept.camera.get('framing', 'product hero shot')}, "
            f"{concept.texture_material}"
        )
        lighting_line = (
            f"{concept.lighting.get('type', 'studio lighting')}, "
            f"{concept.lighting.get('setup', 'controlled key and fill')}, "
            f"{concept.lighting.get('mood', 'premium')} mood"
        )
        camera_line = (
            f"shot on {concept.camera.get('lens_mm', 50)}mm lens at {concept.camera.get('aperture', 'f/4.0')}, "
            f"{concept.camera.get('angle', 'eye-level')} angle"
        )
        mood_line = ", ".join(concept.mood_keywords)

        prompt_body = (
            f"{subject_line}, {lighting_line}, {camera_line}, "
            f"color palette of {palette_desc}, {mood_line}, {style_tag_desc}, "
            "hyperrealistic, editorial commercial photography, 8k, ultra-detailed"
        )

        param_block = f"--ar {ar} --stylize {sty} --chaos {chs} --v {version}"
        if use_style_raw:
            param_block += " --style raw"

        full = f"{prompt_body} {param_block}"

        return EnginePrompt(
            engine="Midjourney v6",
            primary_text=prompt_body,
            parameters={"ar": ar, "stylize": sty, "chaos": chs, "v": version, "style": "raw" if use_style_raw else "default"},
            full_copy_paste=full,
        )

    # ------------------------------------------------------------------ #
    # Flux.1 (natural-language descriptive framing)
    # ------------------------------------------------------------------ #
    @staticmethod
    def build_flux(concept: CreativeConcept) -> EnginePrompt:
        palette_desc = ", ".join(concept.color_palette)
        text = (
            f"A hyperrealistic commercial advertising photograph of {concept.brand_name}, "
            f"captured as a {concept.camera.get('framing', 'centered hero shot')} from a "
            f"{concept.camera.get('angle', 'eye-level')} perspective. The scene is illuminated with "
            f"{concept.lighting.get('type', 'studio lighting')}: {concept.lighting.get('setup', 'a controlled key and fill setup')}, "
            f"producing a {concept.lighting.get('mood', 'premium')} atmosphere. The camera is a "
            f"{concept.camera.get('lens_mm', 50)}mm lens shot wide open around {concept.camera.get('aperture', 'f/4.0')}, "
            f"rendering {concept.texture_material} in crisp, tactile detail. The color story moves through "
            f"{palette_desc}, reinforcing a {', '.join(concept.mood_keywords)} emotional tone. "
            f"{concept.composition_notes} The overall style reads as {', '.join(concept.style_tags) if concept.style_tags else 'high-end editorial advertising'}, "
            "photographed for a global luxury ad campaign, sharp focus on the product, natural physically "
            "accurate light falloff, no text, no watermark, professional color grade."
        )
        return EnginePrompt(engine="Flux.1", primary_text=text, full_copy_paste=text)

    # ------------------------------------------------------------------ #
    # Stable Diffusion XL / SD3 (Positive / Negative structured blocks)
    # ------------------------------------------------------------------ #
    @staticmethod
    def build_sdxl(concept: CreativeConcept, category: Optional[str] = None) -> EnginePrompt:
        palette_desc = ", ".join(concept.color_palette)

        positive_tags = [
            f"{concept.brand_name} {concept.camera.get('framing', 'product hero shot')}",
            concept.texture_material,
            concept.lighting.get("type", "studio lighting"),
            concept.lighting.get("setup", "controlled key and fill"),
            f"{concept.lighting.get('mood', 'premium')} mood",
            f"{concept.camera.get('lens_mm', 50)}mm lens",
            concept.camera.get("aperture", "f/4.0"),
            concept.camera.get("angle", "eye-level angle"),
            f"color palette {palette_desc}",
            ", ".join(concept.mood_keywords),
            ", ".join(concept.style_tags) if concept.style_tags else "commercial advertising photography",
            "hyperrealistic", "8k uhd", "ultra-detailed", "professional color grading",
            "sharp focus", "physically accurate lighting", "award-winning advertising photography",
        ]
        positive = ", ".join(dict.fromkeys(t for t in positive_tags if t))

        negative_terms = list(_UNIVERSAL_NEGATIVE_TERMS)
        if category and category in _CATEGORY_NEGATIVE_ADDITIONS:
            negative_terms.extend(_CATEGORY_NEGATIVE_ADDITIONS[category])
        negative = ", ".join(dict.fromkeys(negative_terms))

        full = f"POSITIVE:\n{positive}\n\nNEGATIVE:\n{negative}"

        return EnginePrompt(
            engine="SDXL / SD3",
            primary_text=positive,
            negative_text=negative,
            full_copy_paste=full,
        )

    # ------------------------------------------------------------------ #
    # Convenience: build all three at once
    # ------------------------------------------------------------------ #
    @classmethod
    def build_all(
        cls,
        concept: CreativeConcept,
        category: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        stylize: Optional[int] = None,
        chaos: Optional[int] = None,
    ) -> Dict[str, EnginePrompt]:
        return {
            "midjourney": cls.build_midjourney(concept, aspect_ratio=aspect_ratio, stylize=stylize, chaos=chaos),
            "flux": cls.build_flux(concept),
            "sdxl": cls.build_sdxl(concept, category=category),
        }
