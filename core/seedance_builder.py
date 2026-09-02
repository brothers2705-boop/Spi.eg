"""
core/seedance_builder.py
===========================
Seedance Motion Prompt Generator.

Translates a resolved CreativeConcept (still-image Art Director output)
into a Seedance-native video generation prompt: camera movement vector,
motion dynamics, temporal pacing, and a video-to-video product-lock
directive for brand-safe consistency between the static key visual and
the motion output.

Design principles
------------------
1. Grounded, not invented: every motion choice is retrieved from the
   `seedance` block of the best-matching golden reference entry (via
   SemanticMatcher), so camera moves and pacing stay physically coherent
   with the lighting/lens choices already made for the still image.
2. Motion Intensity is a first-class, user-controllable dial (1-10) that
   scales chaos/particle-density/speed language without breaking the
   underlying cinematography -- this backs the Streamlit slider directly.
3. Output is engine-native text: a single, dense, comma/clause-separated
   prompt block plus a discrete camera/motion parameter summary so the
   UI can render both a "copy paste" block and a structured inspector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .art_director import CreativeConcept
from .semantic_matcher import MatchResult, SemanticMatcher

# --------------------------------------------------------------------------- #
# Camera movement presets exposed directly to the Streamlit UI
# --------------------------------------------------------------------------- #

CAMERA_MOTION_PRESETS: Dict[str, str] = {
    "Auto (Art Director Choice)": "",  # falls back to the matched reference's camera_movement
    "360 Spin": "360 Product Spin (turntable orbital, locked focal distance, constant angular velocity)",
    "Dynamic Reveal": "Crane Shot with Dutch-tilt Reveal, camera rises and rotates to unveil the subject",
    "Cinematic Zoom": "Slow Push-In Dolly Zoom, foreground/background compression building tension",
    "Orbital Shot": "Slow Orbital Shot (180°, locked focal distance)",
    "FPV Drone": "FPV Drone Whip Pan, aggressive velocity with motion-trail streaks",
    "Crane Rise": "Gentle Crane Rise, revealing environment and scale",
    "Tracking Shot": "Handheld Tracking Shot, motivated subject-led movement",
    "Static Lock": "Static Locked Camera with subtle Dolly Zoom, minimal movement, maximum stability",
}

# Motion dynamics vocabulary the intensity dial modulates
_MOTION_DYNAMICS_LEXICON: Dict[str, List[str]] = {
    "low": ["subtle ambient drift", "gentle volumetric light shift", "micro product rotation"],
    "medium": ["fluid physics on fabric/liquid", "steady product rotation", "layered volumetric lighting shift"],
    "high": ["explosive particle dynamics", "high-speed phantom slow-mo", "aggressive fluid/impact simulation"],
}

_SEAMLESS_LOOP_HINT = (
    "engineered as a seamless loop: final frame matches first frame in framing, lighting, and subject pose"
)
_NON_LOOP_HINT = (
    "engineered as a single continuous non-looping shot with a clear beginning and hero end-frame"
)


@dataclass
class SeedancePrompt:
    concept_label: str
    brand_name: str
    camera_movement: str
    motion_dynamics: List[str]
    motion_intensity: int  # 1-10, user-controllable
    motion_speed: str
    lighting_transition: str
    frame_rate_feel: str
    temporal_pacing: str
    duration_seconds: int
    resolution_hint: str
    product_lock_directive: str
    prompt_text: str = ""
    negative_text: str = ""
    reference_match: Optional[str] = None


class SeedanceBuilder:
    """Builds Seedance-native motion prompts from Art Director creative
    concepts, grounded by the semantic matcher's reference dataset."""

    def __init__(self, matcher: Optional[SemanticMatcher] = None) -> None:
        self.matcher = matcher or SemanticMatcher()

    # ------------------------------------------------------------------ #
    # Retrieval helper
    # ------------------------------------------------------------------ #
    def _best_seedance_reference(self, concept: CreativeConcept, query_text: str) -> Optional[Dict[str, Any]]:
        matches: List[MatchResult] = self.matcher.match(
            query_text, concept_type=concept.concept_type, top_k=1
        )
        if not matches:
            return None
        seedance_meta = matches[0].entry.get("seedance")
        if seedance_meta is None:
            return None
        seedance_meta = dict(seedance_meta)
        seedance_meta["id"] = matches[0].entry.get("id")
        return seedance_meta

    # ------------------------------------------------------------------ #
    # Intensity scaling
    # ------------------------------------------------------------------ #
    @staticmethod
    def _intensity_bucket(intensity: int) -> str:
        if intensity <= 3:
            return "low"
        if intensity <= 7:
            return "medium"
        return "high"

    @classmethod
    def _scaled_motion_dynamics(cls, base_dynamics: List[str], intensity: int) -> List[str]:
        bucket = cls._intensity_bucket(intensity)
        lexicon_terms = _MOTION_DYNAMICS_LEXICON[bucket]
        # Blend the reference's own grounded dynamics with the intensity-appropriate
        # vocabulary so output stays specific to the product AND scales with the dial.
        combined = list(dict.fromkeys([*base_dynamics, *lexicon_terms]))
        return combined[:4]

    @staticmethod
    def _scaled_speed_language(base_speed: str, intensity: int) -> str:
        if intensity <= 2:
            return f"{base_speed}; near-static micro-movement only"
        if intensity <= 5:
            return base_speed
        if intensity <= 8:
            return f"{base_speed}; movement amplitude increased for a more energetic read"
        return f"{base_speed}; maximum kinetic energy, motion-trail streaking permitted"

    # ------------------------------------------------------------------ #
    # Core builder
    # ------------------------------------------------------------------ #
    def build(
        self,
        concept: CreativeConcept,
        query_text: str = "",
        camera_preset: str = "Auto (Art Director Choice)",
        motion_intensity: int = 5,
        duration_seconds: int = 6,
        loop: Optional[bool] = None,
        resolution_hint: str = "1080x1920 vertical, 4K master",
        product_lock: bool = True,
    ) -> SeedancePrompt:
        """Builds a full SeedancePrompt for one creative concept.

        Parameters
        ----------
        concept : the resolved CreativeConcept from ArtDirector
        query_text : original brief text, used to re-query the reference
            dataset for the seedance-specific metadata block
        camera_preset : one of CAMERA_MOTION_PRESETS keys
        motion_intensity : 1 (near-static) to 10 (maximum kinetic energy)
        duration_seconds : target clip duration
        loop : force seamless-loop framing; if None, inferred from the
            concept archetype (minimalist/luxury defaults to loop=True,
            narrative/conceptual default to loop=False)
        resolution_hint : target output resolution/aspect descriptor
        product_lock : whether to include a video-to-video product
            consistency directive (recommended for brand/product ads)
        """
        motion_intensity = max(1, min(10, int(motion_intensity)))
        seedance_ref = self._best_seedance_reference(concept, query_text or concept.brand_name) or {}

        base_camera_movement = seedance_ref.get("camera_movement", "Slow Orbital Shot (180°, locked focal distance)")
        preset_override = CAMERA_MOTION_PRESETS.get(camera_preset, "")
        camera_movement = preset_override or base_camera_movement

        base_dynamics = seedance_ref.get("motion_dynamics", ["Fluid Physics", "Volumetric Lighting Shift"])
        motion_dynamics = self._scaled_motion_dynamics(base_dynamics, motion_intensity)

        base_speed = seedance_ref.get("motion_speed", "naturalistic real-time (1.0x)")
        motion_speed = self._scaled_speed_language(base_speed, motion_intensity)

        lighting_transition = seedance_ref.get(
            "lighting_transition",
            f"{concept.lighting.get('type', 'key light')} sustains a {concept.lighting.get('mood', 'controlled')} mood across the shot",
        )
        frame_rate_feel = seedance_ref.get("frame_rate_feel", "24fps cinematic, natural motion characteristics")

        if loop is None:
            loop = concept.concept_type == "Ultra-Minimalist & Luxury"
        pacing_hint = _SEAMLESS_LOOP_HINT if loop else _NON_LOOP_HINT
        temporal_pacing = f"{seedance_ref.get('temporal_pacing', pacing_hint)}; {pacing_hint}"

        product_lock_directive = (
            f"video-to-video product preservation lock: maintain exact geometry, logo placement, "
            f"proportions, and material finish of the {concept.brand_name} product frame-to-frame; "
            "no morphing, warping, or brand-mark distortion at any point in the sequence."
            if product_lock
            else ""
        )

        prompt_text = self._render_prompt_text(
            concept=concept,
            camera_movement=camera_movement,
            motion_dynamics=motion_dynamics,
            motion_speed=motion_speed,
            lighting_transition=lighting_transition,
            frame_rate_feel=frame_rate_feel,
            temporal_pacing=temporal_pacing,
            duration_seconds=duration_seconds,
            resolution_hint=resolution_hint,
            product_lock_directive=product_lock_directive,
        )

        negative_text = self._render_negative_text()

        return SeedancePrompt(
            concept_label=concept.label,
            brand_name=concept.brand_name,
            camera_movement=camera_movement,
            motion_dynamics=motion_dynamics,
            motion_intensity=motion_intensity,
            motion_speed=motion_speed,
            lighting_transition=lighting_transition,
            frame_rate_feel=frame_rate_feel,
            temporal_pacing=temporal_pacing,
            duration_seconds=duration_seconds,
            resolution_hint=resolution_hint,
            product_lock_directive=product_lock_directive,
            prompt_text=prompt_text,
            negative_text=negative_text,
            reference_match=seedance_ref.get("id"),
        )

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    @staticmethod
    def _render_prompt_text(
        concept: CreativeConcept,
        camera_movement: str,
        motion_dynamics: List[str],
        motion_speed: str,
        lighting_transition: str,
        frame_rate_feel: str,
        temporal_pacing: str,
        duration_seconds: int,
        resolution_hint: str,
        product_lock_directive: str,
    ) -> str:
        palette_desc = ", ".join(concept.color_palette)
        dynamics_desc = ", ".join(motion_dynamics)
        mood_desc = ", ".join(concept.mood_keywords)

        segments = [
            f"[SUBJECT] {concept.brand_name} — {concept.camera.get('framing', 'hero product shot')}, "
            f"{concept.texture_material}",
            f"[CAMERA] {camera_movement}, base lens {concept.camera.get('lens_mm', 50)}mm at "
            f"{concept.camera.get('aperture', 'f/4.0')}, {concept.camera.get('angle', 'eye-level')} starting angle",
            f"[MOTION] {dynamics_desc}; playback character: {motion_speed}",
            f"[LIGHTING] {lighting_transition}",
            f"[PACING] {duration_seconds}s duration, {frame_rate_feel}; {temporal_pacing}",
            f"[COLOR & MOOD] palette {palette_desc}; tone: {mood_desc}",
            f"[OUTPUT] {resolution_hint}, commercial advertising grade, no on-screen text or watermark",
        ]
        if product_lock_directive:
            segments.append(f"[PRODUCT LOCK] {product_lock_directive}")

        return "\n".join(segments)

    @staticmethod
    def _render_negative_text() -> str:
        terms = [
            "warping", "morphing product geometry", "logo distortion", "flickering artifacts",
            "frame stutter", "inconsistent lighting between frames", "text overlays", "watermark",
            "extra limbs", "melting materials", "unnatural physics", "low frame quality",
            "banding", "temporal incoherence", "duplicate objects",
        ]
        return ", ".join(terms)

    # ------------------------------------------------------------------ #
    # Batch helper: build motion prompts for all 3 campaign concepts
    # ------------------------------------------------------------------ #
    def build_for_campaign(
        self,
        concepts: List[CreativeConcept],
        query_text: str = "",
        camera_preset: str = "Auto (Art Director Choice)",
        motion_intensity: int = 5,
        duration_seconds: int = 6,
        resolution_hint: str = "1080x1920 vertical, 4K master",
        product_lock: bool = True,
    ) -> Dict[str, SeedancePrompt]:
        return {
            concept.label: self.build(
                concept,
                query_text=query_text,
                camera_preset=camera_preset,
                motion_intensity=motion_intensity,
                duration_seconds=duration_seconds,
                resolution_hint=resolution_hint,
                product_lock=product_lock,
            )
            for concept in concepts
        }
