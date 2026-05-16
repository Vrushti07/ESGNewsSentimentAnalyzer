"""
Explainable ESG News Sentiment Analyzer
Converts the Google Colab FinBERT + SHAP + SentenceBERT pipeline
into a real-time, interactive Streamlit application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import re
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG — must be first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ESG Sentiment Analyzer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:opsz,wght@9..144,300;9..144,600;9..144,700&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3 {
    font-family: 'Fraunces', serif;
}

.stApp {
    background: #0d1117;
    color: #e6edf3;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] * {
    color: #c9d1d9 !important;
}

/* Main Title */
.main-title {
    font-family: 'Fraunces', serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #58a6ff;
    letter-spacing: -0.5px;
    margin-bottom: 0;
}
.main-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #8b949e;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* Score cards */
.score-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
}
.score-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.3rem;
}
.score-value {
    font-family: 'Fraunces', serif;
    font-size: 2rem;
    font-weight: 600;
    color: #e6edf3;
}

/* Sentiment badges */
.badge-positive { color: #3fb950; font-weight: 600; }
.badge-negative { color: #f85149; font-weight: 600; }
.badge-neutral  { color: #d29922; font-weight: 600; }
.badge-esg      { color: #58a6ff; font-weight: 600; }
.badge-nonesg   { color: #8b949e; font-weight: 600; }

/* Section headers */
.section-header {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #58a6ff;
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
}

/* SHAP container */
.shap-container {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1rem;
}

/* Divider */
.custom-divider {
    border: none;
    border-top: 1px solid #21262d;
    margin: 2rem 0;
}

/* Metric override */
[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1rem;
}
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"] { color: #e6edf3 !important; }

/* Text area / input */
textarea, input[type="text"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.9rem !important;
}
.stTextArea label, .stTextInput label {
    color: #8b949e !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

/* Button */
.stButton > button {
    background: #238636;
    color: #ffffff;
    border: 1px solid #2ea043;
    border-radius: 6px;
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    padding: 0.5rem 1.5rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #2ea043;
    border-color: #3fb950;
}

/* Dataframe */
.stDataFrame { border: 1px solid #21262d; border-radius: 8px; }

/* Info boxes */
.stInfo { background: #0d2137 !important; border: 1px solid #1f4566 !important; }
.stWarning { background: #1f1700 !important; border: 1px solid #3d2e00 !important; }
.stSuccess { background: #001a0a !important; border: 1px solid #1a4428 !important; }
.stError { background: #1a0000 !important; border: 1px solid #4a1010 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONSTANTS — identical to your Colab notebook
# ─────────────────────────────────────────────

TICKERS = [
    "Tata Power", "Adani Green Energy", "JSW Energy", "NHPC", "SJVN",
    "ONGC", "Indian Oil Corporation", "BPCL", "GAIL",
    "VA Tech Wabag", "Ion Exchange India", "Antony Waste Handling Cell",
    "Tata Motors", "Olectra Greentech", "Exide Industries",
    "Sterling and Wilson Renewable Energy", "Inox Wind"
]

ESG_POLICY_KEYWORDS = (
    r"CARBON|NET ZERO|RENEWABLE|EMISSIONS|DECARBONIZATION|GREEN HYDROGEN|"
    r"BIOFUEL|EV|SOLAR|WIND|HYDRO|CLEAN|POLLUTION|WASTE|SPILL|FLY ASH|"
    r"EFFLUENT|CLIMATE|BIODIVERSITY|ECOLOGY|SUSTAINABILITY|METHANE|"
    r"SEQUESTRATION|AMMONIA"
)

NOISE_PHRASES = [
    "stocks in news", "sensex", "nifty", "top losers", "top gainers",
    "should you buy", "market news", "closing bell", "opening bell",
    "technical view", "stock pick", "brokerage view"
]

ESG_ANCHOR = "Renewable energy, solar power, carbon reduction, and sustainability."

# Sentiment thresholds (from your Colab logic)
POSITIVE_THRESHOLD = 0.5
NEGATIVE_THRESHOLD = -0.5


# ─────────────────────────────────────────────
# CACHED MODEL LOADERS  (only load once)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=" Loading FinBERT — this takes ~30s on first run...")
def load_finbert():
    from transformers import pipeline as hf_pipeline
    pipe = hf_pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        top_k=None
    )
    return pipe


@st.cache_resource(show_spinner=" Loading SentenceBERT...")
def load_sentence_bert():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


@st.cache_resource(show_spinner=" Setting up SHAP explainer...")
def load_shap_explainer(_finbert_pipe):
    """
    Returns (masker, explainer_finbert).
    The environmental SHAP explainer is built per-call because it depends on
    the SentenceBERT model which must also be loaded — we build it lazily.
    """
    import shap
    masker = shap.maskers.Text(_finbert_pipe.tokenizer)
    explainer_finbert = shap.Explainer(_finbert_pipe)
    return masker, explainer_finbert


# ─────────────────────────────────────────────
# PIPELINE HELPER FUNCTIONS  
# ─────────────────────────────────────────────

def tag_company(headline: str) -> str | None:
    """Detect company from headline using keyword matching (your logic)."""
    headline_lower = headline.lower()
    for company in TICKERS:
        if company.lower() in headline_lower:
            return company
    return None


def is_esg_headline(headline: str) -> bool:
    """Check if headline matches ESG keyword pattern (your esg_policy_keywords)."""
    return bool(re.search(ESG_POLICY_KEYWORDS, headline, re.IGNORECASE))


def is_noise(headline: str) -> bool:
    """Filter out noise phrases (your noise_phrases logic)."""
    h = headline.lower()
    return any(np in h for np in NOISE_PHRASES)


def generate_finbert_scores(headline: str, pipe) -> dict:
    """Run FinBERT and return {positive, negative, neutral} (your generate_scores)."""
    try:
        results = pipe(headline[:512])
        if isinstance(results[0], list):
            return {item["label"].lower(): item["score"] for item in results[0]}
        elif isinstance(results[0], dict):
            output = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
            label = results[0]["label"].lower()
            output[label] = results[0]["score"]
            return output
    except Exception:
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}


def compute_cumulative_score(scores: dict) -> float:
    """cumulative_score = positive - negative (your exact formula)."""
    return scores.get("positive", 0.0) - scores.get("negative", 0.0)


def classify_sentiment(cumulative_score: float) -> str:
    """Map cumulative score to label (your np.select conditions)."""
    if cumulative_score > POSITIVE_THRESHOLD:
        return "Positive"
    elif cumulative_score < NEGATIVE_THRESHOLD:
        return "Negative"
    else:
        return "Neutral"


def compute_nlp_materiality_score(headline: str, sbert_model) -> float:
    """SentenceBERT cosine similarity vs ESG anchor (your nlp_materiality_score)."""
    from sentence_transformers import util
    headline_emb = sbert_model.encode(headline)
    anchor_emb = sbert_model.encode(ESG_ANCHOR)
    return util.cos_sim(headline_emb, anchor_emb).item()


def compute_esg_adjusted_score(
    finbert_scores: dict,
    materiality_score: float
) -> float:
    """
    Your nlp_weighted_predict logic adapted for a single headline.
    direction * materiality_score (cosine similarity used as magnitude).
    """
    pos = finbert_scores.get("positive", 0.0)
    neg = finbert_scores.get("negative", 0.0)
    direction = 1 if pos > neg else -1
    return direction * materiality_score


# ─────────────────────────────────────────────
# SHAP PLOT HELPERS
# ─────────────────────────────────────────────


def _shap_values_to_html(shap_values_slice) -> str:
    """
    Capture the HTML that shap.plots.text() would display in Jupyter
    and return it as a plain string so we can pass it to st.components.
    Works by temporarily replacing the IPython display mechanism.
    """
    import shap
    import html as html_lib

    # shap.plots.text() calls IPython.display.display() internally.
    # We monkey-patch it to grab the HTML object instead.
    captured_html = []

    class _FakeDisplay:
        def __init__(self, obj, *args, **kwargs):
            # shap passes an IPython HTML object; grab its .data attribute
            if hasattr(obj, "data"):
                captured_html.append(obj.data)
            elif isinstance(obj, str):
                captured_html.append(obj)

    try:
        import IPython.display as ipython_display
        original_display = ipython_display.display
        ipython_display.display = _FakeDisplay

        shap.plots.text(shap_values_slice, display=True)   # display=True triggers the call

        ipython_display.display = original_display
    except Exception:
        # Fallback: call with display=False and build a minimal table from raw values
        try:
            ipython_display.display = original_display
        except Exception:
            pass

    if captured_html:
        raw = captured_html[0]

        # Inject dark-mode override so the widget blends with the app theme
        dark_override = """
        <style>
          body, .shap-text { background: #161b22 !important; color: #e6edf3 !important; }
          span { font-family: 'DM Mono', monospace !important; font-size: 0.9rem !important; }
        </style>
        """
        return dark_override + raw

    # ── Last-resort fallback: draw a simple token-score bar chart as SVG ──
    try:
        tokens = shap_values_slice.data          # list of token strings
        values = shap_values_slice.values        # numpy array of SHAP values

        if hasattr(values, "tolist"):
            values = values.tolist()

        max_abs = max(abs(v) for v in values) if values else 1.0
        max_abs = max_abs or 1.0

        bar_height = 28
        width = 700
        padding = 10
        height = padding + len(tokens) * bar_height + padding

        svg_rows = []
        for i, (tok, val) in enumerate(zip(tokens, values)):
            norm = val / max_abs                            # −1 … +1
            bar_w = int(abs(norm) * (width // 2 - 60))
            color = "#3fb950" if val >= 0 else "#f85149"
            cx = width // 2
            y = padding + i * bar_height
            x_bar = cx if val >= 0 else cx - bar_w
            tok_safe = html_lib.escape(str(tok))
            score_label = f"{val:+.3f}"

            svg_rows.append(
                f'<rect x="{x_bar}" y="{y+4}" width="{bar_w}" height="{bar_height-8}" '
                f'fill="{color}" opacity="0.8" rx="3"/>'
                f'<text x="{cx - 8 if val < 0 else cx + 8}" y="{y + bar_height//2 + 4}" '
                f'text-anchor="{"end" if val < 0 else "start"}" fill="{color}" '
                f'font-size="11" font-family="monospace">{score_label}</text>'
                f'<text x="{cx - bar_w - 6 if val < 0 else cx + bar_w + 6}" y="{y + bar_height//2 + 4}" '
                f'text-anchor="{"end" if val >= 0 else "start"}" fill="#c9d1d9" '
                f'font-size="11" font-family="monospace">{tok_safe}</text>'
            )

        mid_line = f'<line x1="{width//2}" y1="0" x2="{width//2}" y2="{height}" stroke="#30363d" stroke-width="1"/>'
        svg_content = "\n".join(svg_rows)
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'style="background:#161b22">{mid_line}{svg_content}</svg>'
        )
        return svg

    except Exception as e:
        return f'<p style="color:#f85149;font-family:monospace;">SHAP render error: {html_lib.escape(str(e))}</p>'


def render_shap_finbert(headline: str, explainer, pipe) -> tuple:
    """
    Compute FinBERT SHAP values and return (html_str, best_label).
    Replaces the old plt.savefig approach — now returns raw HTML.
    """
    import shap
    try:
        shap_values = explainer([headline])

        raw = pipe(headline[:512])
        if isinstance(raw[0], list):
            best_label = max(raw[0], key=lambda x: x["score"])["label"].lower()
        else:
            best_label = raw[0]["label"].lower()

        html_out = _shap_values_to_html(shap_values[0, :, best_label])
        return html_out, best_label

    except Exception as e:
        return None, str(e)


def render_shap_env(headline: str, masker, finbert_pipe, sbert_model) -> tuple:
    """
    Compute environmental SHAP values and return (html_str, env_audit_score, error).
    Replaces the old plt.savefig approach — now returns raw HTML.
    All FinBERT + SentenceBERT logic is preserved exactly.
    """
    import shap
    from sentence_transformers import util

    def _nlp_weighted_predict(texts):
        clean_texts = [str(t) for t in texts]
        results = finbert_pipe(clean_texts)
        headline_embs = sbert_model.encode(clean_texts, convert_to_tensor=True)
        anchor_emb = sbert_model.encode(ESG_ANCHOR, convert_to_tensor=True)
        similarities = util.cos_sim(headline_embs, anchor_emb).flatten().tolist()
        outputs = []
        for i, res in enumerate(results):
            if isinstance(res, list):
                d = {item["label"].lower(): item["score"] for item in res}
            else:
                d = {res["label"].lower(): res["score"]}
            magnitude = similarities[i]
            direction = 1 if d.get("positive", 0) > d.get("negative", 0) else -1
            final_score = direction * magnitude
            outputs.append([0.0, 0.0, final_score])
        return np.array(outputs)

    try:
        explainer_env = shap.Explainer(
            _nlp_weighted_predict,
            masker=masker,
            output_names=["negative", "neutral", "env_impact"]
        )
        shap_values_env = explainer_env([headline])

        # ── Your exact audit score formula ──────────────────
        base_val = shap_values_env.base_values[0, 2]
        shap_sum = shap_values_env.values[0, :, 2].sum()
        env_audit_score = float(base_val + shap_sum)

        html_out = _shap_values_to_html(shap_values_env[0, :, "env_impact"])
        return html_out, env_audit_score, None

    except Exception as e:
        return None, 0.0, str(e)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="font-family:'Fraunces',serif; font-size:1.3rem; color:#58a6ff; font-weight:600; margin-bottom:0.3rem;">
         ESG Analyzer
    </div>
    <div style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#8b949e; text-transform:uppercase; letter-spacing:2px; margin-bottom:1.5rem;">
        Explainability · Materiality · Sentiment
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.82rem; color:#8b949e; line-height:1.6; margin-bottom:1.5rem;">
        Analyze ESG (Environmental, Social, Governance) sentiment in financial news using:
        <ul style="margin-top:0.5rem; padding-left:1.2rem;">
            <li><b style="color:#c9d1d9;">FinBERT</b> — domain-tuned financial sentiment</li>
            <li><b style="color:#c9d1d9;">SentenceBERT</b> — semantic ESG materiality</li>
            <li><b style="color:#c9d1d9;">SHAP</b> — token-level explainability</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    mode = st.radio(
        "ANALYSIS MODE",
        [" Single Headline", " Upload CSV"],
        index=0
    )

    st.markdown("---")

    st.markdown("""
    <div style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#8b949e; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:0.5rem;">
        Coverage Universe
    </div>
    """, unsafe_allow_html=True)

    with st.expander("View Tracked Companies", expanded=False):
        for t in TICKERS:
            st.markdown(f"<span style='font-family:DM Mono,monospace; font-size:0.75rem; color:#8b949e;'>• {t}</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'DM Mono',monospace; font-size:0.62rem; color:#8b949e; text-align:center;">
        Models load on first run (~30–60s)<br>then stay cached in session.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div class="main-title">Explainable ESG News Sentiment Analyzer</div>
<div class="main-subtitle">FinBERT · SentenceBERT · SHAP · India Energy Sector</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODELS (cached)
# ─────────────────────────────────────────────

finbert_pipe = load_finbert()
sbert_model  = load_sentence_bert()
masker, explainer_finbert = load_shap_explainer(finbert_pipe)


# ══════════════════════════════════════════════
# MODE A — SINGLE HEADLINE
# ══════════════════════════════════════════════

if "Single" in mode:

    st.markdown('<div class="section-header">Single Headline Analysis</div>', unsafe_allow_html=True)

    col_input, col_hint = st.columns([3, 1])
    with col_input:
        headline_input = st.text_area(
            "Paste a news headline",
            placeholder="e.g. Tata Power commissions 500MW solar plant, targets net zero by 2045",
            height=90,
            key="headline_input"
        )
    with col_hint:
        st.markdown("""
        <div style="font-family:'DM Mono',monospace; font-size:0.72rem; color:#8b949e; padding-top:1.6rem; line-height:1.8;">
            <b style="color:#58a6ff;">Tips:</b><br>
            • Include company name<br>
            • Use ESG keywords for<br>&nbsp;&nbsp;materiality scoring<br>
            • English only
        </div>
        """, unsafe_allow_html=True)

    run_btn = st.button(" Analyze Headline", use_container_width=False)

    if run_btn and headline_input.strip():

        headline = headline_input.strip()

        # ── Step 1: Company Detection ──────────────────────
        company = tag_company(headline)
        esg_flag = is_esg_headline(headline)
        noise_flag = is_noise(headline)

        # ── Step 2: FinBERT Scores ─────────────────────────
        with st.spinner("Running FinBERT sentiment analysis..."):
            finbert_scores = generate_finbert_scores(headline, finbert_pipe)

        cumulative_score   = compute_cumulative_score(finbert_scores)
        sentiment_label    = classify_sentiment(cumulative_score)

        # ── Step 3: SentenceBERT Materiality ──────────────
        with st.spinner("Computing ESG materiality score..."):
            materiality_score  = compute_nlp_materiality_score(headline, sbert_model)
            esg_adjusted_score = compute_esg_adjusted_score(finbert_scores, materiality_score)

        # ────────────────────────────────────────────────────
        # OUTPUT SECTION
        # ────────────────────────────────────────────────────

        st.markdown('<div class="section-header">Results</div>', unsafe_allow_html=True)

        # Row 1: entity tags
        r1a, r1b, r1c = st.columns(3)
        with r1a:
            st.markdown(f"""
            <div class="score-card">
                <div class="score-label">Detected Company</div>
                <div class="score-value" style="font-size:1.1rem; padding-top:0.3rem;">
                    {"<span class='badge-esg'>" + company + "</span>" if company else "<span class='badge-nonesg'>Not Detected</span>"}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with r1b:
            esg_color = "badge-esg" if esg_flag else "badge-nonesg"
            esg_text  = " Environmental" if esg_flag else " Non-ESG"
            st.markdown(f"""
            <div class="score-card">
                <div class="score-label">ESG Classification</div>
                <div class="score-value" style="font-size:1.1rem; padding-top:0.3rem;">
                    <span class="{esg_color}">{esg_text}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with r1c:
            noise_text = "Market Noise Detected" if noise_flag else " Clean Signal"
            noise_color = "badge-negative" if noise_flag else "badge-positive"
            st.markdown(f"""
            <div class="score-card">
                <div class="score-label">Signal Quality</div>
                <div class="score-value" style="font-size:1.1rem; padding-top:0.3rem;">
                    <span class="{noise_color}">{noise_text}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Row 2: numeric scores
        r2a, r2b, r2c, r2d, r2e = st.columns(5)
        with r2a:
            st.metric("FinBERT Positive", f"{finbert_scores['positive']:.3f}")
        with r2b:
            st.metric("FinBERT Negative", f"{finbert_scores['negative']:.3f}")
        with r2c:
            st.metric("FinBERT Neutral", f"{finbert_scores['neutral']:.3f}")
        with r2d:
            delta_color = "normal" if cumulative_score >= 0 else "inverse"
            st.metric("Cumulative Score", f"{cumulative_score:+.3f}",
                      delta=sentiment_label)
        with r2e:
            st.metric("ESG Materiality", f"{materiality_score:.3f}",
                      delta=f"Adjusted: {esg_adjusted_score:+.3f}")

        # Sentiment gauge (simple matplotlib bar)
        fig_gauge, ax_gauge = plt.subplots(figsize=(6, 0.6))
        fig_gauge.patch.set_facecolor("#161b22")
        ax_gauge.set_facecolor("#161b22")

        score_clamped = max(-1, min(1, cumulative_score))
        bar_color = "#3fb950" if score_clamped > 0.5 else \
                    "#f85149" if score_clamped < -0.5 else "#d29922"

        ax_gauge.barh(0, score_clamped, color=bar_color, height=0.5, left=0 if score_clamped > 0 else score_clamped)
        ax_gauge.axvline(0, color="#8b949e", linewidth=1.2, linestyle="--")
        ax_gauge.set_xlim(-1, 1)
        ax_gauge.set_yticks([])
        ax_gauge.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax_gauge.set_xticklabels(["−1", "−0.5", "0", "+0.5", "+1"],
                                  color="#8b949e", fontsize=8)
        ax_gauge.spines[:].set_color("#21262d")
        ax_gauge.tick_params(colors="#8b949e")
        ax_gauge.set_title(f"Cumulative Score: {cumulative_score:+.3f}  →  {sentiment_label}",
                           color="#e6edf3", fontsize=9, pad=4, loc="left",
                           fontfamily="monospace")

        st.pyplot(fig_gauge, use_container_width=True)
        plt.close(fig_gauge)

        # ────────────────────────────────────────────────────
        # XAI SECTION
        # ────────────────────────────────────────────────────

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">XAI — SHAP Explainability</div>', unsafe_allow_html=True)

        st.info("SHAP values show which tokens pushed the model toward each sentiment. "
                "Red tokens push negative, blue tokens push positive.")

        import streamlit.components.v1 as components

        # Shared wrapper: dark-themed card that houses the SHAP HTML widget
        def _shap_card(html_content: str, height: int = 220) -> None:
            """Embed SHAP HTML inside a dark-themed iframe via st.components."""
            wrapped = f"""
            <!DOCTYPE html>
            <html>
            <head>
            <meta charset="utf-8">
            <style>
              * {{ box-sizing: border-box; }}
              body {{
                margin: 0; padding: 8px;
                background: #161b22;
                color: #e6edf3;
                font-family: 'DM Mono', 'Courier New', monospace;
                font-size: 13px;
              }}
              /* Override SHAP's default white background */
              svg, .shap-text, div {{ background: transparent !important; }}
              span {{
                border-radius: 3px;
                padding: 1px 2px;
              }}
            </style>
            </head>
            <body>{html_content}</body>
            </html>
            """
            components.html(wrapped, height=height, scrolling=True)

        xai_col1, xai_col2 = st.columns(2)

        # (a) FinBERT SHAP
        with xai_col1:
            st.markdown("""
            <div style="font-family:'DM Mono',monospace; font-size:0.72rem; color:#58a6ff;
                        text-transform:uppercase; letter-spacing:1.5px; margin-bottom:0.6rem;">
                (a) FinBERT Sentiment SHAP
            </div>
            """, unsafe_allow_html=True)
            with st.spinner("Generating FinBERT SHAP explanation..."):
                shap_html_finbert, meta_finbert = render_shap_finbert(
                    headline, explainer_finbert, finbert_pipe
                )
            if shap_html_finbert:
                st.caption(f"SHAP token attributions — FinBERT [{meta_finbert}] | "
                           "")
                _shap_card(shap_html_finbert, height=240)
            else:
                st.error(f"SHAP render failed: {meta_finbert}")

        # (b) Environmental SHAP
        with xai_col2:
            st.markdown("""
            <div style="font-family:'DM Mono',monospace; font-size:0.72rem; color:#3fb950;
                        text-transform:uppercase; letter-spacing:1.5px; margin-bottom:0.6rem;">
                (b) Environmental Impact SHAP
            </div>
            """, unsafe_allow_html=True)
            with st.spinner("Generating Environmental SHAP explanation (SentenceBERT × FinBERT)..."):
                shap_html_env, env_audit_score, err_env = render_shap_env(
                    headline, masker, finbert_pipe, sbert_model
                )
            if shap_html_env:
                st.caption(f"SHAP token attributions — Env Impact | "
                           f"Audit Score: {env_audit_score:+.4f}")
                _shap_card(shap_html_env, height=240)
                st.metric("Environmental Audit Score (SHAP)", f"{env_audit_score:+.4f}")
            else:
                st.error(f"Env SHAP render failed: {err_env}")

        # Interpretation summary
        st.markdown('<div class="section-header">Interpretation Summary</div>', unsafe_allow_html=True)

        sentiment_emoji = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}
        interp = f"""
| Field | Value |
|-------|-------|
| Headline | `{headline[:100]}{"..." if len(headline)>100 else ""}` |
| Company | `{company or "Not Detected"}` |
| ESG Flag | `{"Environmental" if esg_flag else "Non-ESG"}` |
| FinBERT Sentiment | `{sentiment_emoji.get(sentiment_label,"")} {sentiment_label}` |
| Cumulative Score (pos−neg) | `{cumulative_score:+.4f}` |
| ESG Materiality (cosine sim) | `{materiality_score:.4f}` |
| ESG-Adjusted Score | `{esg_adjusted_score:+.4f}` |
| Env. Audit Score (SHAP) | `{env_audit_score:+.4f}` |
"""
        st.markdown(interp)

    elif run_btn:
        st.warning("Please enter a headline before clicking Analyze.")


# ══════════════════════════════════════════════
# MODE B — CSV UPLOAD
# ══════════════════════════════════════════════

else:
    st.markdown('<div class="section-header">Batch CSV Analysis</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'DM Mono',monospace; font-size:0.78rem; color:#8b949e; margin-bottom:1rem;">
        Upload a CSV file with a <b style="color:#58a6ff;">Headline</b> column.
        Optional columns: <b style="color:#8b949e;">Date</b>, <b style="color:#8b949e;">Headline link</b>.
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded_file:
        df_raw = pd.read_csv(uploaded_file)

        if "Headline" not in df_raw.columns:
            st.error(" CSV must have a column named **Headline**.")
        else:
            st.success(f" Loaded {len(df_raw):,} rows.")

            # Preview
            with st.expander(" Preview Raw Data", expanded=False):
                st.dataframe(df_raw.head(20), use_container_width=True)

            # ── Apply pipeline ─────────────────────────────
            run_batch = st.button("⚙️  Run Full Pipeline on CSV", use_container_width=False)

            if run_batch:
                progress_bar = st.progress(0, text="Initializing...")
                n = len(df_raw)

                # Working copy
                df = df_raw.copy()
                df["Headline"] = df["Headline"].astype(str)

                # Step 1: Company tagging
                progress_bar.progress(5, text="Detecting companies...")
                df["company"] = df["Headline"].apply(tag_company)

                # Step 2: ESG flag
                progress_bar.progress(10, text="Applying ESG keyword filter...")
                df["esg_flag"] = df["Headline"].apply(is_esg_headline)
                df["noise_flag"] = df["Headline"].apply(is_noise)

                # Step 3: FinBERT scores (batch)
                progress_bar.progress(15, text="Running FinBERT sentiment scoring...")

                BATCH_SIZE = 16
                all_scores = []
                headlines_list = df["Headline"].tolist()

                for i in range(0, n, BATCH_SIZE):
                    batch = headlines_list[i: i + BATCH_SIZE]
                    batch_truncated = [h[:512] for h in batch]
                    try:
                        results = finbert_pipe(batch_truncated)
                        for res in results:
                            if isinstance(res, list):
                                scores = {item["label"].lower(): item["score"] for item in res}
                            else:
                                scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
                                scores[res["label"].lower()] = res["score"]
                            all_scores.append(scores)
                    except Exception:
                        for _ in batch:
                            all_scores.append({"positive": 0.0, "negative": 0.0, "neutral": 1.0})

                    pct = int(15 + 55 * (i + BATCH_SIZE) / n)
                    progress_bar.progress(min(pct, 70), text=f"FinBERT: {min(i+BATCH_SIZE, n)}/{n} rows...")

                scores_df_batch = pd.DataFrame(all_scores)
                df = pd.concat([df.reset_index(drop=True), scores_df_batch], axis=1)
                df["cumulative_score"] = df["positive"] - df["negative"]

                # Step 4: Sentiment labels
                progress_bar.progress(72, text="Classifying sentiments...")
                df["sentiment"] = df["cumulative_score"].apply(classify_sentiment)

                # Step 5: SentenceBERT materiality (batch encode)
                progress_bar.progress(75, text="Computing ESG materiality scores (SentenceBERT)...")
                from sentence_transformers import util as sbert_util

                headlines_enc = sbert_model.encode(
                    headlines_list, batch_size=32, show_progress_bar=False,
                    convert_to_tensor=True
                )
                anchor_enc = sbert_model.encode(ESG_ANCHOR, convert_to_tensor=True)
                sims = sbert_util.cos_sim(headlines_enc, anchor_enc).flatten().tolist()
                df["materiality_score"] = sims

                # Step 6: ESG adjusted score
                df["esg_adjusted_score"] = df.apply(
                    lambda row: compute_esg_adjusted_score(
                        {"positive": row["positive"], "negative": row["negative"]},
                        row["materiality_score"]
                    ), axis=1
                )

                progress_bar.progress(100, text="Done!")
                st.success(f" Pipeline complete on {n:,} headlines.")

                # ── Summary Metrics ─────────────────────────
                st.markdown('<div class="section-header">Summary Metrics</div>', unsafe_allow_html=True)

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Total Headlines", f"{n:,}")
                m2.metric("Environmental", f"{df['esg_flag'].sum():,}",
                          delta=f"{df['esg_flag'].mean()*100:.1f}%")
                m3.metric("Non-ESG", f"{(~df['esg_flag']).sum():,}")
                m4.metric("Avg Cumulative Score", f"{df['cumulative_score'].mean():+.3f}")
                m5.metric("Avg Materiality Score", f"{df['materiality_score'].mean():.3f}")

                # ── Charts ─────────────────────────────────
                st.markdown('<div class="section-header">Visualizations</div>', unsafe_allow_html=True)

                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    # Sentiment distribution
                    fig1, ax1 = plt.subplots(figsize=(5, 3.5))
                    fig1.patch.set_facecolor("#161b22")
                    ax1.set_facecolor("#161b22")

                    sent_counts = df["sentiment"].value_counts()
                    colors_map = {"Positive": "#3fb950", "Negative": "#f85149", "Neutral": "#d29922"}
                    bar_colors = [colors_map.get(s, "#8b949e") for s in sent_counts.index]

                    ax1.bar(sent_counts.index, sent_counts.values, color=bar_colors,
                            edgecolor="#21262d", linewidth=0.8)
                    ax1.set_title("Sentiment Distribution", color="#e6edf3",
                                  fontsize=10, pad=8, fontfamily="monospace")
                    ax1.set_ylabel("Count", color="#8b949e", fontsize=8)
                    ax1.tick_params(colors="#8b949e", labelsize=8)
                    ax1.spines[:].set_color("#21262d")
                    ax1.yaxis.label.set_color("#8b949e")

                    st.pyplot(fig1, use_container_width=True)
                    plt.close(fig1)

                with chart_col2:
                    # Top companies
                    company_counts = df["company"].dropna().value_counts().head(8)
                    if not company_counts.empty:
                        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
                        fig2.patch.set_facecolor("#161b22")
                        ax2.set_facecolor("#161b22")

                        ax2.barh(company_counts.index[::-1], company_counts.values[::-1],
                                 color="#58a6ff", edgecolor="#21262d", linewidth=0.8)
                        ax2.set_title("Top Companies by Headline Count", color="#e6edf3",
                                      fontsize=10, pad=8, fontfamily="monospace")
                        ax2.tick_params(colors="#8b949e", labelsize=7)
                        ax2.spines[:].set_color("#21262d")

                        st.pyplot(fig2, use_container_width=True)
                        plt.close(fig2)
                    else:
                        st.info("No company matches found in the dataset.")

                # Company × Sentiment stacked bar (your original plot)
                company_sent = df.groupby("company")["sentiment"].value_counts().unstack().fillna(0)
                for col in ["Positive", "Neutral", "Negative"]:
                    if col not in company_sent.columns:
                        company_sent[col] = 0
                company_sent["Total"] = company_sent[["Positive", "Neutral", "Negative"]].sum(axis=1)
                top_companies = company_sent.sort_values("Total", ascending=False).head(6)
                plot_df = top_companies[["Negative", "Neutral", "Positive"]]

                if not plot_df.empty:
                    fig3, ax3 = plt.subplots(figsize=(10, 4))
                    fig3.patch.set_facecolor("#161b22")
                    ax3.set_facecolor("#161b22")

                    bottom = np.zeros(len(plot_df))
                    stack_colors = {"Negative": "#f85149", "Neutral": "#d29922", "Positive": "#3fb950"}
                    for col in ["Negative", "Neutral", "Positive"]:
                        if col in plot_df.columns:
                            ax3.bar(plot_df.index, plot_df[col], bottom=bottom,
                                    label=col, color=stack_colors[col],
                                    edgecolor="#161b22", linewidth=0.5)
                            bottom += plot_df[col].values

                    ax3.set_title("Top 6 Energy Companies — Environmental Materiality Audit",
                                  color="#e6edf3", fontsize=11, pad=10, fontfamily="monospace")
                    ax3.set_ylabel("Headline Count", color="#8b949e", fontsize=9)
                    ax3.tick_params(colors="#8b949e", labelsize=8, axis="x", rotation=15)
                    ax3.tick_params(colors="#8b949e", labelsize=8, axis="y")
                    ax3.spines[:].set_color("#21262d")
                    patches = [mpatches.Patch(color=stack_colors[c], label=c)
                               for c in ["Negative", "Neutral", "Positive"]]
                    ax3.legend(handles=patches, loc="upper right",
                               facecolor="#21262d", edgecolor="#30363d",
                               labelcolor="#c9d1d9", fontsize=8)

                    st.pyplot(fig3, use_container_width=True)
                    plt.close(fig3)

                # ESG vs Non-ESG materiality box
                st.markdown('<div class="section-header">ESG vs Non-ESG Materiality</div>', unsafe_allow_html=True)
                box_col1, box_col2 = st.columns(2)

                esg_scores   = df[df["esg_flag"]]["materiality_score"]
                nonesg_scores = df[~df["esg_flag"]]["materiality_score"]

                with box_col1:
                    fig4, ax4 = plt.subplots(figsize=(5, 3))
                    fig4.patch.set_facecolor("#161b22")
                    ax4.set_facecolor("#161b22")
                    ax4.hist(esg_scores, bins=20, color="#3fb950", alpha=0.8, edgecolor="#161b22")
                    ax4.set_title("ESG Headlines — Materiality", color="#e6edf3",
                                  fontsize=9, fontfamily="monospace")
                    ax4.tick_params(colors="#8b949e", labelsize=7)
                    ax4.spines[:].set_color("#21262d")
                    st.pyplot(fig4, use_container_width=True)
                    plt.close(fig4)

                with box_col2:
                    fig5, ax5 = plt.subplots(figsize=(5, 3))
                    fig5.patch.set_facecolor("#161b22")
                    ax5.set_facecolor("#161b22")
                    ax5.hist(nonesg_scores, bins=20, color="#8b949e", alpha=0.8, edgecolor="#161b22")
                    ax5.set_title("Non-ESG Headlines — Materiality", color="#e6edf3",
                                  fontsize=9, fontfamily="monospace")
                    ax5.tick_params(colors="#8b949e", labelsize=7)
                    ax5.spines[:].set_color("#21262d")
                    st.pyplot(fig5, use_container_width=True)
                    plt.close(fig5)

                # ── Result DataFrame ─────────────────────────
                st.markdown('<div class="section-header">Result Data</div>', unsafe_allow_html=True)

                display_cols = [
                    c for c in [
                        "Headline", "company", "esg_flag", "sentiment",
                        "cumulative_score", "materiality_score",
                        "esg_adjusted_score", "positive", "negative", "neutral"
                    ] if c in df.columns
                ]

                st.dataframe(df[display_cols].head(100), use_container_width=True)

                # ── Download ──────────────────────────────────
                csv_out = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=" Download Full Results as CSV",
                    data=csv_out,
                    file_name="esg_sentiment_results.csv",
                    mime="text/csv",
                    use_container_width=False
                )


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
st.markdown("""
<div style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#8b949e;
            text-align:center; padding:0.5rem 0 1rem 0; letter-spacing:1px;">
    FinBERT (ProsusAI) · SentenceBERT (all-MiniLM-L6-v2) · SHAP · India ESG Coverage Universe
</div>
""", unsafe_allow_html=True)