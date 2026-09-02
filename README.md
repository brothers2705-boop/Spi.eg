# VisionForge AI

Enterprise-grade, portable AI Brand Identity & Advertising Pipeline. Takes a
minimal brief (brand/product name + idea + optional specs) and produces three
distinct campaign directions, each fully translated into copy-paste-ready
prompts for **Midjourney v6**, **Flux.1**, **Stable Diffusion XL / SD3**
(static key visuals), and **Seedance AI** (motion / video-to-video prompts).

---

## Architecture

```
visionforge/
├── app.py                      # Streamlit UI (dark theme, tabs, sliders)
├── core/
│   ├── art_director.py         # Art Director Logic: lighting, camera,
│   │                           #   color psychology, brand mood, narrative
│   ├── semantic_matcher.py     # TF-IDF few-shot retrieval over the golden
│   │                           #   reference dataset
│   ├── prompt_builder.py       # Static engine syntax translation
│   │                           #   (Midjourney v6 / Flux.1 / SDXL-SD3)
│   ├── seedance_builder.py     # Seedance motion-prompt generator: camera
│   │                           #   vectors, motion scale, temporal pacing
│   └── database.py             # SQLite offline campaign history store
├── dataset/
│   └── reference_prompts.json  # 21 golden reference prompts, each with a
│                                #   still-image spec AND a nested Seedance
│                                #   motion metadata block
├── .streamlit/config.toml      # Dark theme
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

### Pipeline flow

1. **Input** — brand name, brief idea, optional product specs/category.
2. **Analysis (Art Director Logic)** — `SemanticMatcher` retrieves the
   closest golden reference prompt per concept archetype; `ArtDirector`
   resolves lighting physics, camera composition, color palette psychology,
   and brand mood into three concepts:
   - **Concept A** — Ultra-Minimalist & Luxury
   - **Concept B** — Narrative & Emotionally Driven
   - **Concept C** — High-Impact & Conceptual
3. **Synthesis** — for every concept, two output sets are generated:
   - **Static Key Visual** — Midjourney v6 (`--ar --style raw --v 6.0
     --stylize --chaos`), Flux.1 (natural-language descriptive prompt),
     and SDXL/SD3 (structured Positive/Negative blocks).
   - **Seedance Motion Prompt** — camera movement vector, motion dynamics,
     motion-intensity-scaled speed language, lighting transition, temporal
     pacing/looping behavior, and a video-to-video product-lock directive.
4. **Persistence** — campaigns can be saved to a local SQLite database and
   reloaded from the sidebar history panel. Fully offline, no cloud
   dependency.

### Optional LLM enrichment

If an `ANTHROPIC_API_KEY` environment variable is present, VisionForge will
additionally call Claude to expand the deterministic creative-direction
narrative into richer agency-grade prose. This is a pure enrichment layer —
if the key is absent or the call fails for any reason, the app silently
falls back to its built-in offline template engine. **No functionality is
lost by running with zero configuration.**

---

## Running locally (no Docker)

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501`.

## Running via Docker (single command, any OS)

```bash
docker-compose up --build
```

Then open `http://localhost:8501`. The SQLite history persists in the
`visionforge_data` named volume across restarts. To enable optional Claude
enrichment, copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` before
running `docker-compose up`.

To stop:

```bash
docker-compose down
```

---

## Extending the reference dataset

`dataset/reference_prompts.json` is the few-shot pattern library. Each entry
contains a still-image specification (lighting, camera, color palette,
texture) **and** a nested `seedance` block (camera movement, motion
dynamics, motion speed, lighting transition, frame-rate feel, temporal
pacing, default motion intensity). Add new entries following the existing
schema (documented at the top of the file) to extend VisionForge into new
brand niches — no code changes required, the `SemanticMatcher` picks up new
entries automatically on next app restart (or call `.reload()`).
