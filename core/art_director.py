"""
core/art_director.py
======================
The "brain" of VisionForge AI.

Implements the internal Art Director Logic described in the spec:
    - Visual Storytelling
    - Lighting Physics
    - Camera Composition
    - Brand Mood Analysis
    - Color Palette Psychology

The engine is offline-first: it runs entirely on deterministic rule-based
reasoning grounded by the SemanticMatcher's few-shot retrieval over the
golden reference dataset. It does NOT require any external API key.

Optionally, if an ANTHROPIC_API_KEY environment variable is present, the
ArtDirector will additionally call Claude to expand the campaign
narrative copy (tagline + rationale) into richer prose. This is a pure
enrichment step layered on top of the deterministic pipeline -- if the
call fails or no key is configured, VisionForge silently falls back to
its built-in template engine, so the tool remains fully portable and
functional with zero configuration.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .semantic_matcher import MatchResult, SemanticMatcher

# --------------------------------------------------------------------------- #
# System prompts / creative doctrine
#
# These are kept as first-class constants (rather than buried in code) so
# they can be swapped, audited, or fed to an external LLM chain if a studio
# later wants to hybridize VisionForge with a hosted model.
# --------------------------------------------------------------------------- #

ART_DIRECTOR_MASTER_SYSTEM_PROMPT = """\
You are the Principal Art Director at a tier-1 global creative agency,
responsible for defining the visual language of luxury and commercial
advertising campaigns. For every brief, you reason like a director of
photography, a color scientist, and a brand strategist simultaneously.

You always evaluate five dimensions before proposing a concept:
1. VISUAL STORYTELLING - what single frame communicates the brand promise?
2. LIGHTING PHYSICS - what light source, modifier, and angle produce the
   correct emotional temperature and material rendering?
3. CAMERA COMPOSITION - what focal length, aperture, and framing produce
   the correct sense of intimacy, scale, or power?
4. BRAND MOOD ANALYSIS - what 3-5 adjectives define the emotional
   register the audience must feel within 400 milliseconds of viewing?
5. COLOR PALETTE PSYCHOLOGY - what dominant/secondary/accent hues
   reinforce (or productively subvert) the category's expectations?

You never produce generic, stock-photo-grade direction. Every choice is
deliberate, physically plausible, and tied back to the brand's stated
positioning.
"""

CONCEPT_SYSTEM_PROMPTS: Dict[str, str] = {
    "Ultra-Minimalist & Luxury": """\
Direction: ULTRA-MINIMALIST & LUXURY.
Doctrine: subtraction is the highest form of sophistication. Favor
negative space, a single dominant light source, restrained color
palettes (near-monochrome with one precious accent), and hero-object
framing. Camera work is controlled and static -- tripod-locked, clean
studio or seamless environments, macro or three-quarter product hero
angles. The viewer should feel that nothing more could be removed
without breaking the composition.""",
    "Narrative & Emotionally Driven": """\
Direction: NARRATIVE & EMOTIONALLY DRIVEN.
Doctrine: the product is a supporting character in a human moment.
Favor natural or motivated practical light sources (window light,
golden hour, tungsten practicals), candid framing, shallow depth of
field, and a color palette drawn from a specific time-of-day and
place. The viewer should feel they have interrupted a real, ongoing
life -- not witnessed a staged advertisement.""",
    "High-Impact & Conceptual": """\
Direction: HIGH-IMPACT & CONCEPTUAL.
Doctrine: break physical or scale expectations to create instant
stopping power. Favor hard directional light, colored gel washes,
freeze-motion strobe, dramatic low or Dutch angles, and saturated or
juxtaposed color palettes. Composition can be asymmetric, explosive,
or surreal. The viewer should feel the image could not exist as an
unedited photograph -- and be intrigued rather than confused.""",
}

_CONCEPT_LABELS = list(CONCEPT_SYSTEM_PROMPTS.keys())

# Color psychology lookup: maps a coarse "brand personality" axis to a
# recommended accent-color psychological rationale. Used to generate the
# human-readable justification shown in the UI.
_COLOR_PSYCHOLOGY: Dict[str, str] = {
    "gold": "signals heritage, wealth, and enduring value; slows the eye and reads as 'earned' luxury.",
    "black": "creates authority and focus; maximizes perceived contrast and product dominance.",
    "white": "reads as purity, precision, and clinical confidence; ideal for tech and skincare.",
    "amber": "evokes warmth, craft, and time (aging, distillation, patina).",
    "blue": "reads as trust, calm, and technological competence.",
    "red": "raises arousal and urgency; used sparingly as an accent to draw the eye first.",
    "green": "signals wellness, sustainability, and natural origin.",
    "neon": "signals futurism, energy, and youth-culture disruption.",
    "pastel": "softens category expectations; reads as approachable premium rather than austere luxury.",
}


def _infer_color_rationale(hex_palette: List[str]) -> str:
    """Extremely lightweight heuristic color-family classifier used purely
    to generate an explainable one-line psychology rationale for the UI."""
    families: List[str] = []
    for hex_code in hex_palette:
        h = hex_code.lstrip("#").lower()
        if len(h) != 6:
            continue
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        if r > 200 and g > 200 and b > 200:
            families.append("white")
        elif r < 40 and g < 40 and b < 40:
            families.append("black")
        elif r > 180 and g > 130 and b < 100:
            families.append("gold")
        elif r > 120 and g > 70 and b < 60 and r > g > b:
            families.append("amber")
        elif b > r and b > g:
            families.append("blue")
        elif r > 180 and g < 100 and b < 100:
            families.append("red")
        elif g > r and g > b:
            families.append("green")
        elif b > 150 and g > 150 and r < 100:
            families.append("neon")
        else:
            families.append("pastel")
    seen = []
    for fam in families:
        if fam not in seen:
            seen.append(fam)
    if not seen:
        return "balances contrast and warmth to keep attention anchored on the product."
    return " ".join(_COLOR_PSYCHOLOGY.get(f, "") for f in seen[:2]).strip()


@dataclass
class CreativeConcept:
    """A single fully-resolved campaign direction (one of A / B / C)."""

    label: str  # "Concept A", "Concept B", "Concept C"
    concept_type: str  # e.g. "Ultra-Minimalist & Luxury"
    brand_name: str
    tagline: str
    narrative: str
    mood_keywords: List[str]
    lighting: Dict[str, str]
    camera: Dict[str, Any]
    color_palette: List[str]
    color_rationale: str
    texture_material: str
    composition_notes: str
    reference_match: Optional[str] = None  # which golden reference grounded this
    style_tags: List[str] = field(default_factory=list)
    mj_defaults: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CampaignBrief:
    brand_name: str
    product_category: str
    concepts: List[CreativeConcept]
    raw_input_brief: str


class ArtDirector:
    """Top-level orchestrator implementing the Art Director Logic."""

    _TAGLINE_TEMPLATES: Dict[str, List[str]] = {
        "Ultra-Minimalist & Luxury": [
            "{brand} — refined to its essence.",
            "{brand}. Nothing extra. Everything considered.",
            "The quiet confidence of {brand}.",
        ],
        "Narrative & Emotionally Driven": [
            "{brand} — a moment worth keeping.",
            "Every {category} tells a story. This is {brand}'s.",
            "{brand}, lived in.",
        ],
        "High-Impact & Conceptual": [
            "{brand}. Rewrite the rules.",
            "This is not {category} as you know it. This is {brand}.",
            "{brand} — impossible, until now.",
        ],
    }

    def __init__(self, matcher: Optional[SemanticMatcher] = None, seed: Optional[int] = None) -> None:
        self.matcher = matcher or SemanticMatcher()
        self._rng = random.Random(seed)
        self._anthropic_client = self._maybe_init_llm()

    # ------------------------------------------------------------------ #
    # Optional LLM enrichment (fully non-blocking, offline-safe)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _maybe_init_llm():
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        try:
            import anthropic  # local import: optional dependency

            return anthropic.Anthropic(api_key=api_key)
        except Exception:
            return None

    def _enrich_narrative_with_llm(self, concept_type: str, brand_name: str, brief_idea: str, base_narrative: str) -> str:
        """Best-effort enrichment. Never raises -- falls back to the
        deterministic base_narrative on any failure so the app stays
        fully functional offline."""
        if self._anthropic_client is None:
            return base_narrative
        try:
            system_prompt = ART_DIRECTOR_MASTER_SYSTEM_PROMPT + "\n\n" + CONCEPT_SYSTEM_PROMPTS[concept_type]
            user_prompt = (
                f"Brand: {brand_name}\n"
                f"Brief idea: {brief_idea}\n"
                f"Draft creative rationale to expand:\n{base_narrative}\n\n"
                "Rewrite this into a tight, agency-grade creative rationale "
                "paragraph (3-4 sentences max). Do not use markdown, bullet "
                "points, or headings. Return only the paragraph."
            )
            response = self._anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text_parts = [b.text for b in response.content if getattr(b, "type", "") == "text"]
            enriched = " ".join(text_parts).strip()
            return enriched or base_narrative
        except Exception:
            return base_narrative

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def generate_creative_brief(
        self,
        brand_name: str,
        brief_idea: str,
        product_specs: str = "",
        product_category: Optional[str] = None,
        use_llm_enrichment: bool = False,
    ) -> CampaignBrief:
        """Generates the full 3-concept campaign brief (Concept A/B/C)."""
        query_text = " ".join(filter(None, [brand_name, brief_idea, product_specs, product_category or ""]))
        best_matches = self.matcher.best_match_per_concept(query_text)

        inferred_category = product_category or self._infer_category(best_matches, brief_idea)

        concepts: List[CreativeConcept] = []
        for letter, concept_type in zip("ABC", _CONCEPT_LABELS):
            match = best_matches.get(concept_type)
            concept = self._build_concept(
                letter=letter,
                concept_type=concept_type,
                brand_name=brand_name,
                brief_idea=brief_idea,
                product_specs=product_specs,
                category=inferred_category,
                match=match,
            )
            if use_llm_enrichment:
                concept.narrative = self._enrich_narrative_with_llm(
                    concept_type, brand_name, brief_idea, concept.narrative
                )
            concepts.append(concept)

        return CampaignBrief(
            brand_name=brand_name,
            product_category=inferred_category,
            concepts=concepts,
            raw_input_brief=brief_idea,
        )

    # ------------------------------------------------------------------ #
    # Internal reasoning helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _infer_category(best_matches: Dict[str, MatchResult], brief_idea: str) -> str:
        for match in best_matches.values():
            cat = match.entry.get("product_category")
            if cat:
                return cat
        return "product"

    def _build_concept(
        self,
        letter: str,
        concept_type: str,
        brand_name: str,
        brief_idea: str,
        product_specs: str,
        category: str,
        match: Optional[MatchResult],
    ) -> CreativeConcept:
        entry = match.entry if match else {}

        lighting = dict(entry.get("lighting", {})) or {
            "type": "balanced key + fill",
            "setup": "single key light with soft fill to control contrast ratio",
            "mood": "controlled, professional",
        }
        camera = dict(entry.get("camera", {})) or {
            "lens_mm": 50,
            "aperture": "f/4.0",
            "angle": "eye-level",
            "framing": "centered product hero shot",
        }
        color_palette = list(entry.get("color_palette", [])) or ["#111111", "#EAEAEA", "#C9A227"]
        texture_material = entry.get("texture_material", "clean studio surface rendering")
        style_tags = list(entry.get("style_tags", []))

        mood_keywords = self._derive_mood_keywords(concept_type)
        composition_notes = self._derive_composition_notes(concept_type, camera)
        tagline = self._rng.choice(self._TAGLINE_TEMPLATES[concept_type]).format(
            brand=brand_name, category=category
        )
        narrative = self._derive_narrative(concept_type, brand_name, brief_idea, category, lighting, camera)
        color_rationale = _infer_color_rationale(color_palette)

        return CreativeConcept(
            label=f"Concept {letter}",
            concept_type=concept_type,
            brand_name=brand_name,
            tagline=tagline,
            narrative=narrative,
            mood_keywords=mood_keywords,
            lighting=lighting,
            camera=camera,
            color_palette=color_palette,
            color_rationale=color_rationale,
            texture_material=texture_material,
            composition_notes=composition_notes,
            reference_match=entry.get("id"),
            style_tags=style_tags,
            mj_defaults=dict(entry.get("mj_params", {})) or {"ar": "16:9", "stylize": 300, "chaos": 10, "style": "raw", "v": "6.0"},
        )

    @staticmethod
    def _derive_mood_keywords(concept_type: str) -> List[str]:
        bank = {
            "Ultra-Minimalist & Luxury": ["refined", "quiet", "precise", "opulent", "considered"],
            "Narrative & Emotionally Driven": ["intimate", "warm", "candid", "nostalgic", "human"],
            "High-Impact & Conceptual": ["bold", "kinetic", "futuristic", "disruptive", "electric"],
        }
        return bank[concept_type]

    @staticmethod
    def _derive_composition_notes(concept_type: str, camera: Dict[str, Any]) -> str:
        angle = camera.get("angle", "eye-level")
        framing = camera.get("framing", "centered hero shot")
        if concept_type == "Ultra-Minimalist & Luxury":
            return (
                f"Static, tripod-locked composition. {framing.capitalize()} at a {angle} angle. "
                "Generous negative space on at least two sides; product occupies the golden-ratio "
                "intersection point rather than dead center."
            )
        if concept_type == "Narrative & Emotionally Driven":
            return (
                f"Handheld or motivated camera movement implied. {framing.capitalize()} shot from a "
                f"{angle} vantage to preserve eye-line intimacy with the subject. Foreground/background "
                "layered for depth via shallow focus fall-off."
            )
        return (
            f"Asymmetric, high-energy composition. {framing.capitalize()} from a {angle} to exaggerate "
            "scale and dynamism. Diagonal leading lines and edge-frame tension are embraced rather than "
            "corrected."
        )

    def _derive_narrative(
        self,
        concept_type: str,
        brand_name: str,
        brief_idea: str,
        category: str,
        lighting: Dict[str, str],
        camera: Dict[str, Any],
    ) -> str:
        idea_clause = brief_idea.strip().rstrip(".")
        if concept_type == "Ultra-Minimalist & Luxury":
            return (
                f"{brand_name} is presented as an object of quiet authority. We isolate the {category} "
                f"against a controlled studio environment lit with {lighting.get('type', 'a single key source')}, "
                f"letting {lighting.get('mood', 'restraint')} do the emotional work. The brief's core idea — "
                f"\"{idea_clause}\" — is expressed through subtraction: no clutter, no distraction, just "
                "the product and the craft behind it."
            )
        if concept_type == "Narrative & Emotionally Driven":
            return (
                f"{brand_name} becomes part of a lived-in human moment rather than a studio object. Using "
                f"{lighting.get('type', 'natural light')} to create a {lighting.get('mood', 'warm')} atmosphere, "
                f"we frame the {category} within a real scene that dramatizes \"{idea_clause}\". The camera "
                f"(a {camera.get('lens_mm', 85)}mm perspective at {camera.get('aperture', 'f/2.0')}) stays close "
                "enough to feel like a memory, not an advertisement."
            )
        return (
            f"{brand_name} breaks category convention to earn attention in under a second. We push "
            f"\"{idea_clause}\" into a heightened, almost surreal register using {lighting.get('type', 'hard directional light')} "
            f"and a {camera.get('angle', 'dramatic')} camera angle, so the {category} reads as a statement of "
            "intent rather than a catalog shot."
        )
