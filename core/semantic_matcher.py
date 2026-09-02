"""
core/semantic_matcher.py
==========================
Lightweight, dependency-minimal semantic matcher used to implement the
"Few-Shot Pattern Learning Engine" described in the VisionForge AI spec.

Design goals
------------
1. Fully offline: no external API calls, no downloaded embedding models.
2. Fast enough to run per-keystroke in a Streamlit app (pure numpy/sklearn
   TF-IDF cosine similarity over a ~20-40 row corpus is sub-millisecond).
3. Deterministic and explainable: every match returns *why* it matched
   (shared tokens) so the Art Director layer can justify its choices.

The matcher indexes each reference prompt's `keywords` + `brand_niche` +
`product_category` + `style_tags` into a single bag-of-words document,
then ranks reference prompts against the free-text user brief using
TF-IDF cosine similarity. A tag-overlap boost is added on top so that
literal category matches (e.g. user typed "watch" and a reference is
tagged "watch") are never out-ranked by loose textual similarity.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _flatten_reference_text(entry: Dict[str, Any]) -> str:
    """Builds the bag-of-words document used for TF-IDF indexing of one
    reference prompt entry."""
    parts: List[str] = [
        entry.get("brand_niche", ""),
        entry.get("product_category", ""),
        entry.get("concept_type", ""),
    ]
    parts.extend(entry.get("keywords", []))
    parts.extend(entry.get("style_tags", []))
    lighting = entry.get("lighting", {})
    parts.append(lighting.get("type", ""))
    parts.append(lighting.get("mood", ""))
    parts.append(entry.get("texture_material", ""))
    return " ".join(p for p in parts if p)


@dataclass
class MatchResult:
    entry: Dict[str, Any]
    score: float
    matched_tokens: List[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.entry.get("id", "UNKNOWN")


class SemanticMatcher:
    """TF-IDF + tag-overlap semantic matcher over the golden reference
    prompt dataset."""

    def __init__(self, dataset_path: Optional[str] = None) -> None:
        self.dataset_path = Path(
            dataset_path
            or Path(__file__).resolve().parent.parent / "dataset" / "reference_prompts.json"
        )
        self.entries: List[Dict[str, Any]] = []
        self._corpus: List[str] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None
        self._load()

    # ------------------------------------------------------------------ #
    # Loading / indexing
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Reference prompt dataset not found at: {self.dataset_path}"
            )
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.entries = data.get("reference_prompts", [])
        if not self.entries:
            raise ValueError("Reference prompt dataset is empty.")
        self._corpus = [_flatten_reference_text(e) for e in self.entries]
        self._vectorizer = TfidfVectorizer(
            tokenizer=_tokenize,
            lowercase=False,  # already lowercased by _tokenize
            token_pattern=None,
        )
        self._matrix = self._vectorizer.fit_transform(self._corpus)

    def reload(self) -> None:
        """Re-reads the dataset from disk (useful if the JSON was edited
        live, e.g. via an admin panel)."""
        self._load()

    # ------------------------------------------------------------------ #
    # Matching
    # ------------------------------------------------------------------ #
    def match(
        self,
        query_text: str,
        concept_type: Optional[str] = None,
        top_k: int = 3,
        tag_boost: float = 0.25,
    ) -> List[MatchResult]:
        """Returns the top_k reference prompts most similar to query_text.

        Parameters
        ----------
        query_text : free-text brief (brand name + brief idea + specs)
        concept_type : optional hard filter restricting candidates to a
            specific concept archetype (Ultra-Minimalist & Luxury, etc.)
        top_k : number of results to return
        tag_boost : additive score bonus per literal keyword overlap,
            ensures category-exact matches always float to the top.
        """
        if not query_text or not query_text.strip():
            query_text = "premium product advertising"

        query_vec = self._vectorizer.transform([query_text])
        sims = cosine_similarity(query_vec, self._matrix)[0]

        query_tokens = set(_tokenize(query_text))

        candidates: List[MatchResult] = []
        for idx, entry in enumerate(self.entries):
            if concept_type and entry.get("concept_type") != concept_type:
                continue
            entry_tokens = set(_tokenize(_flatten_reference_text(entry)))
            overlap = query_tokens & entry_tokens
            boosted_score = float(sims[idx]) + tag_boost * len(overlap)
            candidates.append(
                MatchResult(entry=entry, score=boosted_score, matched_tokens=sorted(overlap))
            )

        candidates.sort(key=lambda c: c.score, reverse=True)

        if not candidates and concept_type:
            # Fallback: no entries tagged with that concept_type exist;
            # widen the search rather than returning nothing.
            return self.match(query_text, concept_type=None, top_k=top_k, tag_boost=tag_boost)

        return candidates[:top_k]

    def best_match_per_concept(self, query_text: str) -> Dict[str, MatchResult]:
        """Convenience helper: returns the single best reference match for
        each of the three campaign concept archetypes."""
        concept_types = [
            "Ultra-Minimalist & Luxury",
            "Narrative & Emotionally Driven",
            "High-Impact & Conceptual",
        ]
        result: Dict[str, MatchResult] = {}
        for ct in concept_types:
            matches = self.match(query_text, concept_type=ct, top_k=1)
            if matches:
                result[ct] = matches[0]
        return result

    def all_niches(self) -> List[str]:
        return sorted({e.get("brand_niche", "") for e in self.entries if e.get("brand_niche")})
