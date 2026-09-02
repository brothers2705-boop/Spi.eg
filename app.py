"""
app.py
========
VisionForge AI -- Streamlit front end.

A minimalist, dark-themed control room for generating tier-1 commercial
campaign concepts and translating them into copy-paste-ready prompts for
Midjourney v6, Flux.1, SDXL/SD3 (static key visuals) and Seedance AI
(motion / video-to-video prompts).

Run locally:      streamlit run app.py
Run via Docker:    docker-compose up
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.art_director import ArtDirector, CampaignBrief, CreativeConcept
from core.prompt_builder import PromptBuilder, EnginePrompt
from core.seedance_builder import SeedanceBuilder, SeedancePrompt, CAMERA_MOTION_PRESETS
from core.database import CampaignDatabase

# --------------------------------------------------------------------------- #
# Page configuration + dark theme styling
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="VisionForge AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CUSTOM_CSS = """
<style>
    .stApp { background-color: #0B0B0D; }
    section[data-testid="stSidebar"] { background-color: #101013; border-right: 1px solid #232327; }
    h1, h2, h3, h4 { letter-spacing: 0.02em; }
    .vf-eyebrow {
        color: #C9A227; font-size: 0.78rem; text-transform: uppercase;
        letter-spacing: 0.18em; font-weight: 600; margin-bottom: 0.1rem;
    }
    .vf-title { font-size: 2.1rem; font-weight: 700; color: #F5F5F0; margin-top: 0; }
    .vf-subtitle { color: #9A9AA2; font-size: 0.95rem; margin-bottom: 1.4rem; }
    .vf-card {
        background-color: #131316; border: 1px solid #232327; border-radius: 10px;
        padding: 1.1rem 1.3rem; margin-bottom: 1rem;
    }
    .vf-tagline { color: #F5F5F0; font-size: 1.15rem; font-weight: 600; font-style: italic; }
    .vf-label { color: #C9A227; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 700; }
    .vf-swatch-row { display: flex; gap: 6px; margin-top: 4px; margin-bottom: 4px; }
    .vf-swatch { width: 34px; height: 34px; border-radius: 6px; border: 1px solid #2c2c30; }
    .vf-pill {
        display: inline-block; background-color: #1D1D21; color: #D8D8DC; border: 1px solid #2c2c30;
        border-radius: 999px; padding: 2px 10px; font-size: 0.75rem; margin: 2px 4px 2px 0;
    }
    .vf-ref-tag { color: #6E6E76; font-size: 0.72rem; margin-top: 0.4rem; }
    div[data-testid="stMetricValue"] { color: #F5F5F0; }
</style>
"""
st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Cached resources (engine singletons)
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner=False)
def get_art_director() -> ArtDirector:
    return ArtDirector()


@st.cache_resource(show_spinner=False)
def get_seedance_builder(_art_director: ArtDirector) -> SeedanceBuilder:
    return SeedanceBuilder(matcher=_art_director.matcher)


@st.cache_resource(show_spinner=False)
def get_database() -> CampaignDatabase:
    return CampaignDatabase()


art_director = get_art_director()
seedance_builder = get_seedance_builder(art_director)
db = get_database()

# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #

if "brief" not in st.session_state:
    st.session_state.brief: CampaignBrief | None = None
if "query_text" not in st.session_state:
    st.session_state.query_text = ""

# --------------------------------------------------------------------------- #
# Sidebar -- inputs, parameters, history
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.markdown('<div class="vf-eyebrow">VisionForge AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="vf-title" style="font-size:1.4rem;">Campaign Brief</div>', unsafe_allow_html=True)

    brand_name = st.text_input("Brand / Product Name", placeholder="e.g. Aurum & Co.")
    brief_idea = st.text_area(
        "Brief Idea",
        placeholder="e.g. Launch campaign for a new titanium dive watch built for adventurers.",
        height=110,
    )
    product_specs = st.text_area(
        "Product / Image Specs (optional)",
        placeholder="e.g. 42mm titanium case, sapphire crystal, orange accent bezel.",
        height=80,
    )
    product_category_override = st.text_input(
        "Product Category (optional override)", placeholder="e.g. watch, perfume bottle, sneaker"
    )

    st.divider()
    st.markdown('<div class="vf-eyebrow">Static Engine Parameters</div>', unsafe_allow_html=True)
    aspect_ratio = st.selectbox("Aspect Ratio (Midjourney)", ["16:9", "1:1", "4:5", "3:4", "3:2", "9:16"], index=0)
    stylize_val = st.slider("Stylize (--stylize)", min_value=0, max_value=1000, value=300, step=10)
    chaos_val = st.slider("Chaos (--chaos)", min_value=0, max_value=100, value=10, step=5)
    mj_version = st.selectbox("Midjourney Version", ["6.0", "6.1"], index=0)
    use_style_raw = st.checkbox("Use --style raw", value=True)

    st.divider()
    st.markdown('<div class="vf-eyebrow">Seedance Video Parameters</div>', unsafe_allow_html=True)
    camera_preset = st.selectbox("Camera Motion Preset", list(CAMERA_MOTION_PRESETS.keys()), index=0)
    motion_intensity = st.slider("Motion Intensity", min_value=1, max_value=10, value=5)
    duration_seconds = st.slider("Clip Duration (seconds)", min_value=2, max_value=15, value=6)
    resolution_hint = st.selectbox(
        "Output Resolution", ["1080x1920 vertical, 4K master", "1920x1080 horizontal, 4K master", "1080x1080 square, 4K master"], index=0
    )
    product_lock = st.checkbox("Enable Product Lock (video-to-video consistency)", value=True)

    st.divider()
    use_llm_enrichment = st.checkbox(
        "Enrich narratives with Claude (requires ANTHROPIC_API_KEY)", value=False,
        help="Optional. If no API key is configured, VisionForge silently falls back to the built-in offline template engine.",
    )

    generate_clicked = st.button("⚡ Generate Campaign Concepts", type="primary", use_container_width=True)

    st.divider()
    st.markdown('<div class="vf-eyebrow">Campaign History</div>', unsafe_allow_html=True)
    history_rows = db.list_campaigns(limit=15)
    if not history_rows:
        st.caption("No saved campaigns yet.")
    else:
        for row in history_rows:
            label = f"{row['brand_name']} — {row['created_at'][:16].replace('T', ' ')}"
            if st.button(label, key=f"hist_{row['id']}", use_container_width=True):
                st.session_state["_load_campaign_id"] = row["id"]

# --------------------------------------------------------------------------- #
# Handle history load
# --------------------------------------------------------------------------- #

if "_load_campaign_id" in st.session_state:
    record = db.get_campaign(st.session_state.pop("_load_campaign_id"))
    if record:
        st.info(f"Loaded saved campaign for **{record['brand_name']}** from history. Regenerate to make live edits.")
        st.json(record["payload"], expanded=False)

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #

st.markdown('<div class="vf-eyebrow">Enterprise AI Brand Identity & Advertising Pipeline</div>', unsafe_allow_html=True)
st.markdown('<div class="vf-title">VisionForge AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="vf-subtitle">Static key visuals and Seedance motion prompts, synthesized from one brief.</div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #

if generate_clicked:
    if not brand_name.strip() or not brief_idea.strip():
        st.error("Please provide at least a Brand/Product Name and a Brief Idea.")
    else:
        with st.spinner("Running Art Director Logic — lighting physics, composition, color psychology..."):
            brief = art_director.generate_creative_brief(
                brand_name=brand_name.strip(),
                brief_idea=brief_idea.strip(),
                product_specs=product_specs.strip(),
                product_category=product_category_override.strip() or None,
                use_llm_enrichment=use_llm_enrichment,
            )
            st.session_state.brief = brief
            st.session_state.query_text = " ".join(
                filter(None, [brand_name, brief_idea, product_specs, product_category_override])
            )

# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #

def render_creative_direction(concept: CreativeConcept) -> None:
    st.markdown(f'<div class="vf-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="vf-label">{concept.concept_type}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="vf-tagline">“{concept.tagline}”</div>', unsafe_allow_html=True)
    st.write(concept.narrative)

    cols = st.columns([1, 1, 1])
    with cols[0]:
        st.markdown('<div class="vf-label">Lighting</div>', unsafe_allow_html=True)
        st.caption(f"**{concept.lighting.get('type', '—')}**")
        st.caption(concept.lighting.get("setup", ""))
        st.caption(f"Mood: {concept.lighting.get('mood', '—')}")
    with cols[1]:
        st.markdown('<div class="vf-label">Camera</div>', unsafe_allow_html=True)
        st.caption(f"{concept.camera.get('lens_mm', '—')}mm @ {concept.camera.get('aperture', '—')}")
        st.caption(f"Angle: {concept.camera.get('angle', '—')}")
        st.caption(f"Framing: {concept.camera.get('framing', '—')}")
    with cols[2]:
        st.markdown('<div class="vf-label">Color Palette</div>', unsafe_allow_html=True)
        swatch_html = '<div class="vf-swatch-row">' + "".join(
            f'<div class="vf-swatch" style="background-color:{c};" title="{c}"></div>' for c in concept.color_palette
        ) + "</div>"
        st.markdown(swatch_html, unsafe_allow_html=True)
        st.caption(concept.color_rationale)

    st.markdown('<div class="vf-label" style="margin-top:0.6rem;">Mood</div>', unsafe_allow_html=True)
    st.markdown(
        "".join(f'<span class="vf-pill">{m}</span>' for m in concept.mood_keywords),
        unsafe_allow_html=True,
    )
    if concept.reference_match:
        st.markdown(f'<div class="vf-ref-tag">Grounded by golden reference {concept.reference_match}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_static_prompts(concept: CreativeConcept, category: str) -> Dict[str, EnginePrompt]:
    prompts = PromptBuilder.build_all(
        concept,
        category=category,
        aspect_ratio=aspect_ratio,
        stylize=stylize_val,
        chaos=chaos_val,
    )
    # Rebuild MJ prompt with the exact version/style-raw toggle from the sidebar
    prompts["midjourney"] = PromptBuilder.build_midjourney(
        concept, aspect_ratio=aspect_ratio, stylize=stylize_val, chaos=chaos_val,
        version=mj_version, use_style_raw=use_style_raw,
    )

    mj_tab, flux_tab, sdxl_tab = st.tabs(["🖼️ Midjourney v6", "🌊 Flux.1", "🎛️ SDXL / SD3"])
    with mj_tab:
        st.code(prompts["midjourney"].full_copy_paste, language="text")
        st.caption(f"Parameters: {prompts['midjourney'].parameters}")
    with flux_tab:
        st.code(prompts["flux"].full_copy_paste, language="text")
    with sdxl_tab:
        st.markdown("**Positive**")
        st.code(prompts["sdxl"].primary_text, language="text")
        st.markdown("**Negative**")
        st.code(prompts["sdxl"].negative_text, language="text")

    return prompts


def render_seedance_studio(concept: CreativeConcept, query_text: str) -> SeedancePrompt:
    seedance_prompt = seedance_builder.build(
        concept,
        query_text=query_text,
        camera_preset=camera_preset,
        motion_intensity=motion_intensity,
        duration_seconds=duration_seconds,
        resolution_hint=resolution_hint,
        product_lock=product_lock,
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Motion Intensity", f"{seedance_prompt.motion_intensity}/10")
    metric_cols[1].metric("Duration", f"{seedance_prompt.duration_seconds}s")
    metric_cols[2].metric("Camera Move", seedance_prompt.camera_movement.split("(")[0].strip())
    metric_cols[3].metric("Product Lock", "ON" if product_lock else "OFF")

    st.markdown('<div class="vf-label">Seedance Motion Prompt</div>', unsafe_allow_html=True)
    st.code(seedance_prompt.prompt_text, language="text")

    with st.expander("Negative Prompt / Exclusions"):
        st.code(seedance_prompt.negative_text, language="text")

    with st.expander("Motion Dynamics Breakdown"):
        st.write(f"**Motion speed:** {seedance_prompt.motion_speed}")
        st.write(f"**Lighting transition:** {seedance_prompt.lighting_transition}")
        st.write(f"**Frame-rate feel:** {seedance_prompt.frame_rate_feel}")
        st.write(f"**Temporal pacing:** {seedance_prompt.temporal_pacing}")
        st.markdown(
            "".join(f'<span class="vf-pill">{d}</span>' for d in seedance_prompt.motion_dynamics),
            unsafe_allow_html=True,
        )
        if seedance_prompt.reference_match:
            st.markdown(
                f'<div class="vf-ref-tag">Grounded by Seedance reference {seedance_prompt.reference_match}</div>',
                unsafe_allow_html=True,
            )

    return seedance_prompt


# --------------------------------------------------------------------------- #
# Main output area
# --------------------------------------------------------------------------- #

brief: CampaignBrief | None = st.session_state.brief

if brief is None:
    st.info("Fill in the campaign brief in the sidebar and click **Generate Campaign Concepts** to begin.")
else:
    concept_tabs = st.tabs([f"{c.label} — {c.concept_type}" for c in brief.concepts])
    all_payload = {"brand_name": brief.brand_name, "product_category": brief.product_category, "concepts": []}

    for tab, concept in zip(concept_tabs, brief.concepts):
        with tab:
            render_creative_direction(concept)

            studio_tabs = st.tabs(["🖌️ Static Key Visual Studio", "🎬 Seedance Video Studio"])
            with studio_tabs[0]:
                static_prompts = render_static_prompts(concept, brief.product_category)
            with studio_tabs[1]:
                seedance_prompt = render_seedance_studio(concept, st.session_state.query_text)

            all_payload["concepts"].append(
                {
                    "concept": concept.__dict__,
                    "static_prompts": {k: v.__dict__ for k, v in static_prompts.items()},
                    "seedance_prompt": seedance_prompt.__dict__,
                }
            )

    st.divider()
    save_col, _ = st.columns([1, 3])
    with save_col:
        if st.button("💾 Save Campaign to History", use_container_width=True):
            campaign_id = db.save_campaign(
                brand_name=brief.brand_name,
                product_category=brief.product_category,
                brief_idea=brief.raw_input_brief,
                payload=all_payload,
            )
            st.success(f"Saved campaign `{campaign_id[:8]}` to local history.")
            st.rerun()

st.markdown(
    '<div style="text-align:center; color:#4A4A50; font-size:0.75rem; margin-top:2.5rem;">'
    "VisionForge AI · Offline-first · Runs entirely on-device via Docker</div>",
    unsafe_allow_html=True,
)
