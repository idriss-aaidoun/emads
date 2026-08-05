"""
EMADS UI theme
====================================

Pure presentation layer for the Streamlit dashboard: design tokens, the
injected CSS block, and small HTML-building helpers (metric cards, the
pipeline stepper, styled tables). Nothing here reads or writes EMADSState —
`streamlit_app.py` decides WHAT to show, this module only decides HOW it
looks. Kept separate purely because the CSS block got long enough to crowd
out the actual page logic, not because it needs its own abstraction layer.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------
# Design tokens — one palette per mode. PRIMARY/ACCENT stay constant (brand
# identity shouldn't shift with the theme); everything that touches a
# background or body text swaps between the two.
# ----------------------------------------------------------------------
PRIMARY = "#4338CA"        # indigo — brand color, matches .streamlit/config.toml
PRIMARY_LIGHT = "#6366F1"
ACCENT = "#06B6D4"         # cyan

_PALETTES = {
    "light": dict(
        PRIMARY_SOFT="#EEF2FF",
        SUCCESS="#10B981", SUCCESS_SOFT="#ECFDF5",
        WARNING="#F59E0B", WARNING_SOFT="#FFFBEB",
        ERROR="#F87171", ERROR_SOFT="#FEF2F2",
        TEXT="#1E293B", TEXT_MUTED="#64748B",
        BORDER="#E2E8F0", BG_CARD="#FFFFFF", BG_SUBTLE="#F8FAFC", BG_APP="#FFFFFF",
    ),
    "dark": dict(
        PRIMARY_SOFT="#1E1B4B",
        SUCCESS="#34D399", SUCCESS_SOFT="#052E22",
        WARNING="#FBBF24", WARNING_SOFT="#3A2A05",
        ERROR="#F87171", ERROR_SOFT="#3B1414",
        TEXT="#E2E8F0", TEXT_MUTED="#94A3B8",
        BORDER="#334155", BG_CARD="#1E293B", BG_SUBTLE="#0F172A", BG_APP="#0B1220",
    ),
}

# Mutable at runtime by inject_theme() — every helper below (metric_card,
# badge, style_* ) reads colors through this instead of a fixed module-level
# constant, so a single call flips the whole dashboard's palette.
_palette = dict(_PALETTES["light"])


def _status_colors() -> dict:
    return {
        "good": (_palette["SUCCESS"], _palette["SUCCESS_SOFT"]),
        "warn": (_palette["WARNING"], _palette["WARNING_SOFT"]),
        "bad": (_palette["ERROR"], _palette["ERROR_SOFT"]),
        None: (_palette["BORDER"], _palette["BG_CARD"]),
    }


def _build_css(mode: str) -> str:
    p = _PALETTES[mode]
    PRIMARY_SOFT, SUCCESS, SUCCESS_SOFT = p["PRIMARY_SOFT"], p["SUCCESS"], p["SUCCESS_SOFT"]
    WARNING, WARNING_SOFT, ERROR, ERROR_SOFT = p["WARNING"], p["WARNING_SOFT"], p["ERROR"], p["ERROR_SOFT"]
    TEXT, TEXT_MUTED, BORDER = p["TEXT"], p["TEXT_MUTED"], p["BORDER"]
    BG_CARD, BG_SUBTLE, BG_APP = p["BG_CARD"], p["BG_SUBTLE"], p["BG_APP"]

    # Dark mode also needs to repaint Streamlit's own widgets (inputs,
    # selects, dataframes, code blocks, ...) since .streamlit/config.toml
    # pins a light native theme — our custom cards aren't the only surface.
    # Streamlit's own emotion-generated classes (one per widget instance,
    # re-rendered by React after our <style> tag is injected) win the cascade
    # over a plain attribute selector — hence !important everywhere here,
    # including on the sidebar itself, which otherwise stays stuck on the
    # light secondaryBackgroundColor from .streamlit/config.toml.
    dark_widget_overrides = f"""
    .stApp, .stApp [data-testid="stAppViewContainer"], .stApp [data-testid="stHeader"] {{
        background-color: {BG_APP} !important; color: {TEXT} !important;
    }}
    .stMarkdown, .stMarkdown p, label, .stCaption, span, p, li {{ color: {TEXT} !important; }}

    /* ---- Sidebar shell + every native widget inside it ------------------ */
    [data-testid="stSidebar"], [data-testid="stSidebarContent"],
    [data-testid="stSidebarUserContent"] {{
        background-color: {BG_SUBTLE} !important; color: {TEXT} !important;
        border-right: 1px solid {BORDER};
    }}
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
        background-color: {BG_CARD} !important; border: 1px dashed {BORDER} !important; color: {TEXT} !important;
    }}
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stSidebar"] small {{ color: {TEXT_MUTED} !important; }}
    [data-testid="stSidebar"] .stButton > button, [data-testid="stSidebar"] .stDownloadButton > button {{
        background-color: {BG_CARD} !important; color: {TEXT} !important; border: 1px solid {BORDER} !important;
    }}
    [data-testid="stAlert"] {{ background-color: {BG_CARD} !important; }}

    div[data-testid="stMetric"] {{ background-color: {BG_CARD}; border-radius: 12px; padding: 8px; }}
    div[data-testid="stExpander"] {{ background-color: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 12px; }}
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div,
    .stTextArea textarea {{
        background-color: {BG_CARD} !important; color: {TEXT} !important; border-color: {BORDER} !important;
    }}
    div[data-testid="stDataFrame"], div[data-testid="stTable"] {{ background-color: {BG_CARD}; }}
    .stTabs [data-baseweb="tab-list"] {{ background-color: transparent; }}
    .stTabs [data-baseweb="tab"] {{ color: {TEXT_MUTED}; }}
    .stTabs [aria-selected="true"] {{ color: {PRIMARY}; }}
    code {{ background-color: {BG_SUBTLE}; color: {TEXT}; }}
    """ if mode == "dark" else ""

    return f"""
<style>
    html, body, [class*="css"] {{
        font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
    }}
    .stApp {{ background-color: {BG_APP}; }}
    {dark_widget_overrides}

    /* ---- Header ------------------------------------------------------- */
    .main-header {{
        background: linear-gradient(135deg, {PRIMARY} 0%, {ACCENT} 100%);
        padding: 30px 34px; border-radius: 16px; margin-bottom: 28px;
        box-shadow: 0 1px 2px rgba(15,23,42,0.06), 0 8px 24px rgba(67,56,202,0.18);
    }}
    .main-header h1 {{ color: white; font-size: 30px; font-weight: 800; margin: 0; letter-spacing: -0.02em; }}
    .main-header p {{ color: rgba(255,255,255,0.88); font-size: 15px; margin-top: 6px; }}

    /* ---- Section titles ------------------------------------------------ */
    .section-title {{
        font-size: 18px; font-weight: 700; color: {TEXT}; margin: 22px 0 12px 0;
        padding-bottom: 8px; border-bottom: 2px solid {BORDER};
    }}

    /* ---- Cards (metric, decision, sidebar, empty-state) ---------------- */
    .metric-card, .decision-card, .sidebar-card, .step-preview-card {{
        background-color: {BG_CARD};
        border-radius: 16px;
        border: 1px solid {BORDER};
        box-shadow: 0 1px 2px rgba(15,23,42,0.04), 0 4px 10px rgba(15,23,42,0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .metric-card:hover, .decision-card:hover, .step-preview-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 2px 4px rgba(15,23,42,0.06), 0 12px 24px rgba(15,23,42,0.10);
    }}

    .metric-card {{
        padding: 18px 20px; border-left: 4px solid var(--status-color, {BORDER});
    }}
    .metric-card .icon {{ font-size: 20px; margin-bottom: 6px; }}
    .metric-card .label {{
        color: {TEXT_MUTED}; font-size: 12px; text-transform: uppercase;
        letter-spacing: 0.06em; font-weight: 600;
    }}
    .metric-card .value {{ color: {TEXT}; font-size: 26px; font-weight: 800; margin-top: 4px; }}

    .decision-card {{
        padding: 14px 18px; margin-bottom: 12px; border-left: 4px solid {ACCENT};
    }}
    .decision-agent {{ color: {PRIMARY}; font-weight: 700; font-size: 13px; }}
    .decision-reason {{ color: {TEXT_MUTED}; font-size: 13px; margin-top: 4px; }}

    /* ---- Badges ---------------------------------------------------------- */
    .badge {{
        display: inline-block; padding: 3px 11px; border-radius: 999px;
        font-size: 11px; font-weight: 700; letter-spacing: 0.02em;
    }}
    .badge-good {{ color: #065F46; background-color: {SUCCESS_SOFT}; border: 1px solid {SUCCESS}55; }}
    .badge-warn {{ color: #92400E; background-color: {WARNING_SOFT}; border: 1px solid {WARNING}55; }}
    .badge-bad  {{ color: #991B1B; background-color: {ERROR_SOFT};   border: 1px solid {ERROR}55; }}
    .badge-neutral {{ color: {PRIMARY}; background-color: {PRIMARY_SOFT}; border: 1px solid {PRIMARY}33; }}

    /* ---- Sidebar ----------------------------------------------------------- */
    div[data-testid="stSidebar"] {{ background-color: {BG_SUBTLE}; border-right: 1px solid {BORDER}; }}
    .sidebar-card {{ padding: 14px 16px; margin-bottom: 14px; }}
    .sidebar-section-label {{
        color: {TEXT_MUTED}; font-size: 11px; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.07em; margin: 4px 0 8px 0;
    }}

    /* ---- Buttons ------------------------------------------------------------ */
    .stButton > button, .stDownloadButton > button {{
        border-radius: 10px; transition: transform 0.15s ease, box-shadow 0.15s ease;
        font-weight: 600;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 6px 14px rgba(67,56,202,0.18);
    }}

    /* ---- Progress bar -------------------------------------------------------- */
    div[data-testid="stProgress"] > div > div > div {{
        background: linear-gradient(90deg, {PRIMARY} 0%, {ACCENT} 100%);
        border-radius: 999px;
    }}
    div[data-testid="stProgress"] > div > div {{ border-radius: 999px; height: 10px; }}

    /* ---- Pipeline stepper ------------------------------------------------------ */
    .stepper {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0 18px 0; }}
    .stepper-item {{
        flex: 1 1 100px; min-width: 100px; text-align: center; padding: 10px 8px;
        border-radius: 12px; font-size: 12px; font-weight: 600;
        border: 1px solid {BORDER}; background-color: {BG_CARD}; color: {TEXT_MUTED};
        transition: all 0.2s ease;
    }}
    .stepper-item .step-icon {{ display: block; font-size: 16px; margin-bottom: 4px; }}
    .stepper-item.done {{ background-color: {SUCCESS_SOFT}; border-color: {SUCCESS}66; color: #065F46; }}
    .stepper-item.active {{
        background-color: {PRIMARY_SOFT}; border-color: {PRIMARY}; color: {PRIMARY};
        box-shadow: 0 2px 8px rgba(67,56,202,0.20);
    }}
    .stepper-item.pending {{ opacity: 0.55; }}

    /* ---- Empty-state pipeline preview -------------------------------------------- */
    .step-preview-grid {{
        display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 14px; margin-top: 18px;
    }}
    .step-preview-card {{ padding: 16px 18px; }}
    .step-preview-card .step-num {{
        display: inline-block; width: 22px; height: 22px; line-height: 22px;
        border-radius: 999px; background-color: {PRIMARY_SOFT}; color: {PRIMARY};
        font-size: 11px; font-weight: 800; text-align: center; margin-bottom: 8px;
    }}
    .step-preview-card .step-title {{ font-weight: 700; color: {TEXT}; font-size: 14px; }}
    .step-preview-card .step-desc {{ color: {TEXT_MUTED}; font-size: 12px; margin-top: 4px; }}
</style>
"""


def inject_theme(mode: str = "light") -> None:
    """Injects the CSS for `mode` ("light" or "dark") and remembers the
    palette so metric_card/badge/style_* render with matching colors."""
    global _palette
    _palette = dict(_PALETTES.get(mode, _PALETTES["light"]))
    st.markdown(_build_css(mode if mode in _PALETTES else "light"), unsafe_allow_html=True)


def status_from_thresholds(value: float | None, good: float, warn: float) -> str | None:
    """Maps a 0-1 score to a card status: >=good -> good, >=warn -> warn, else bad."""
    if value is None:
        return None
    if value >= good:
        return "good"
    if value >= warn:
        return "warn"
    return "bad"


def metric_card(col, label: str, value, icon: str = "", status: str | None = None) -> None:
    status_colors = _status_colors()
    color, _ = status_colors.get(status, status_colors[None])
    col.markdown(f"""
        <div class="metric-card" style="--status-color: {color};">
            {f'<div class="icon">{icon}</div>' if icon else ''}
            <div class="label">{label}</div>
            <div class="value">{value}</div>
        </div>
    """, unsafe_allow_html=True)


def badge(text: str, kind: str = "neutral") -> str:
    return f'<span class="badge badge-{kind}">{text}</span>'


def stepper_html(steps: list[tuple[str, str]], current_index: int | None) -> str:
    """Renders the fixed pipeline steps as a horizontal stepper.

    `current_index` is the index of the step currently running (None once the
    whole run has finished, at which point every step already shown is done).
    """
    items = []
    for i, (_, label) in enumerate(steps):
        if current_index is None or i < current_index:
            state_cls, icon = "done", "✔"
        elif i == current_index:
            state_cls, icon = "active", "●"
        else:
            state_cls, icon = "pending", "○"
        items.append(
            f'<div class="stepper-item {state_cls}">'
            f'<span class="step-icon">{icon}</span>{label}'
            f'</div>'
        )
    return f'<div class="stepper">{"".join(items)}</div>'


def style_model_comparison(df: pd.DataFrame, selected_model_name: str | None) -> "pd.io.formats.style.Styler":
    """Highlights the full row of the selected model and applies a color
    gradient to the CV score column, instead of just tinting the max cell."""

    def highlight_selected_row(row: pd.Series):
        if selected_model_name is not None and row.get("Model") == selected_model_name:
            return [f"background-color: {_palette['PRIMARY_SOFT']}; font-weight: 700;"] * len(row)
        return [""] * len(row)

    styler = (
        df.style
        .apply(highlight_selected_row, axis=1)
        .background_gradient(subset=["Mean CV Score"], cmap="BuGn")
        .format({"Mean CV Score": "{:.4f}", "Std Dev": "{:.4f}"})
    )
    return styler


def style_metrics_table(df: pd.DataFrame, pct_cols: list[str] | None = None) -> "pd.io.formats.style.Styler":
    """Generic evaluation-metrics table styling: subtle row striping and
    consistent decimal/percent formatting."""
    fmt = {col: "{:.1%}" for col in (pct_cols or [])}
    bg_card, bg_subtle = _palette["BG_CARD"], _palette["BG_SUBTLE"]
    styler = (
        df.style
        .format(fmt, na_rep="N/A")
        .set_properties(**{"background-color": bg_card, "color": _palette["TEXT"]})
        .apply(
            lambda row: [f"background-color: {bg_subtle if row.name % 2 else bg_card}"] * len(row),
            axis=1,
        )
    )
    return styler
