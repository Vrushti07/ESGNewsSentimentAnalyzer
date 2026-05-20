# Explainable ESG News Sentiment Analyzer

> A dual-model NLP pipeline for financial news sentiment scoring and environmental materiality auditing of Indian energy sector stocks — with token-level SHAP explainability.

---

## Overview

Standard sentiment tools fail in financial contexts. A headline like *"ONGC reports record profits from fossil fuel expansion"* scores positive in generic models but is negative from an ESG standpoint. This project solves that by combining **FinBERT** (domain-tuned financial sentiment) with **SentenceBERT** (semantic ESG materiality) into a single auditable score, backed by SHAP explainability so every result is transparent and defensible.

Built as a real-time **Streamlit application**, it supports both single-headline analysis and batch CSV processing — replicating analyst-grade ESG materiality audit workflows.

---

## Problem Statement

ESG (Environmental, Social, Governance) materiality auditing of financial news requires two things that no single off-the-shelf model provides:

1. **Financial-domain sentiment** — understanding what "positive" and "negative" mean in the context of investment news, not social media
2. **Environmental relevance scoring** — distinguishing headlines that are genuinely about ESG topics from those that merely mention a company

This project addresses both, and adds explainability so the scores can be trusted and reported on.

---

## Coverage Universe

18 Indian energy sector companies across four ESG-relevant categories:

| Category | Companies |
|---|---|
| Renewable Energy | Tata Power, Adani Green Energy, JSW Energy, NHPC, SJVN |
| Oil and Gas (Energy Transition) | ONGC, Indian Oil Corporation, BPCL, GAIL |
| Environmental Services | VA Tech Wabag, Ion Exchange India, Antony Waste Handling Cell |
| EV and Clean Mobility | Tata Motors, Olectra Greentech, Exide Industries |
| Green Infrastructure | Sterling and Wilson Renewable Energy, Inox Wind |

---

## Pipeline Architecture

```
Raw Headlines (Economic Times, 2022-2025)
        |
        v
  Noise Filtering
  (bollywood, lifestyle, astrology, market aggregators)
        |
        v
  Company Detection
  (substring matching across 18 tickers)
        |
        v
  ESG Keyword Filter
  (CARBON, RENEWABLE, SOLAR, EMISSIONS, BIODIVERSITY + 20 more)
        |
        v
  FinBERT Sentiment Scoring
  (positive, negative, neutral scores)
        |
        v
  Cumulative Score = positive - negative
        |
        v
  SentenceBERT Materiality Scoring
  (cosine similarity vs ESG anchor sentence)
        |
        v
  ESG-Adjusted Score = direction x materiality
        |
        v
  SHAP Explainability
  (a) FinBERT token attribution
  (b) Environmental impact attribution (custom composite explainer)
        |
        v
  Streamlit Dashboard
  (single headline + batch CSV modes)
```

---

## Models

### FinBERT — Financial Sentiment
- Model: `ProsusAI/finbert`
- A BERT model fine-tuned on financial news corpora
- Outputs three scores per headline: `positive`, `negative`, `neutral` (sum to 1.0)
- **Cumulative score** = `positive − negative` (range: −1 to +1)
- Sentiment label thresholds: `> 0.5` Positive, `< −0.5` Negative, else Neutral

### SentenceBERT — ESG Materiality
- Model: `all-MiniLM-L6-v2`
- Encodes each headline into a semantic vector
- Computes cosine similarity against the ESG anchor:
  *"Renewable energy, solar power, carbon reduction, and sustainability."*
- **Materiality score** range: 0 to 1 (higher = more semantically ESG-relevant)
- **ESG-adjusted score** = `direction × materiality_score`

### SHAP — Explainability
- **(a) FinBERT SHAP:** `shap.Explainer` on the FinBERT pipeline — token-level attribution showing which words pushed financial sentiment
- **(b) Environmental SHAP:** Custom composite explainer over `nlp_weighted_predict` (FinBERT direction × SentenceBERT magnitude) — shows which tokens drove the ESG impact score
- **Environmental Audit Score** = `base_value + Σ(SHAP values)` — the final auditable per-headline number

---

## Features

### Single Headline Mode
- Detects company from the tracked ticker list
- Applies ESG keyword filter and noise filter
- Runs full FinBERT + SentenceBERT pipeline
- Displays all five scores: positive, negative, neutral, cumulative, materiality
- Renders both SHAP visualizations (FinBERT sentiment + environmental impact)
- Outputs an interpretation summary table

### Batch CSV Mode
- Upload any CSV with a `Headline` column
- Batched FinBERT inference (batch size 16) for speed
- Vectorized SentenceBERT encoding (single `.encode()` call across all rows)
- Outputs:
  - Summary metrics (total, ESG vs non-ESG count, average scores)
  - Sentiment distribution bar chart
  - Top companies by headline count
  - Stacked company × sentiment materiality audit chart
  - ESG vs non-ESG materiality histograms
  - Full results dataframe preview
  - Downloadable enriched CSV

---

## Tech Stack

| Component | Technology |
|---|---|
| Application framework | Streamlit |
| Financial sentiment | FinBERT (ProsusAI/finbert via HuggingFace Transformers) |
| Semantic similarity | SentenceBERT (all-MiniLM-L6-v2) |
| Explainability | SHAP |
| Data processing | Pandas, NumPy |
| Visualization | Matplotlib |
| Language | Python 3.10+ |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/esg-sentiment-analyzer.git
cd esg-sentiment-analyzer

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

> **Note:** First run downloads approximately 500 MB of model weights for FinBERT and MiniLM. These are cached in `~/.cache/huggingface/` and reused on subsequent runs. All three models (FinBERT pipeline, SentenceBERT, SHAP explainer) are loaded once per session via `@st.cache_resource`.

---

## Input Format

For CSV batch mode, upload a file with at minimum a `Headline` column. Optional columns `Date` and `Headline link` are preserved in output if present.

```
Headline, Date, Headline link
Tata Power commissions 500MW solar plant, 15-03-2024, /energy/tata-power-solar
ONGC reports record quarterly profit, 20-03-2024, /markets/ongc-results
```

---

## Output Columns

| Column | Description |
|---|---|
| `company` | Detected company from ticker list |
| `esg_flag` | True if headline matches ESG keyword pattern |
| `noise_flag` | True if headline matches market noise phrases |
| `positive` | FinBERT positive score (0–1) |
| `negative` | FinBERT negative score (0–1) |
| `neutral` | FinBERT neutral score (0–1) |
| `cumulative_score` | positive − negative (−1 to +1) |
| `sentiment` | Positive / Neutral / Negative label |
| `materiality_score` | SentenceBERT cosine similarity vs ESG anchor (0–1) |
| `esg_adjusted_score` | direction × materiality (−1 to +1) |

---

## ESG Keywords Monitored

```
CARBON · NET ZERO · RENEWABLE · EMISSIONS · DECARBONIZATION
GREEN HYDROGEN · BIOFUEL · EV · SOLAR · WIND · HYDRO · CLEAN
POLLUTION · WASTE · SPILL · FLY ASH · EFFLUENT · CLIMATE
BIODIVERSITY · ECOLOGY · SUSTAINABILITY · METHANE · SEQUESTRATION · AMMONIA
```

---

## Key Design Decisions

**Why FinBERT over VADER or TextBlob?**
General-purpose sentiment models are trained on social media and product reviews. FinBERT is trained on financial news, earnings calls, and analyst reports — the vocabulary and tone are fundamentally different.

**Why SentenceBERT materiality scoring over keyword matching alone?**
Keyword matching catches explicit ESG terms but misses semantic context. A headline about "Adani Green Energy's debt restructuring" contains a green energy company but has no ESG content. SentenceBERT's cosine similarity against an ESG anchor captures meaning, not just vocabulary.

**Why SHAP?**
ESG audits need to be defensible. A score without attribution is not auditable. SHAP provides mathematically grounded token-level explanations that show exactly which words contributed to each score — making the output suitable for reporting.

**Why `st.components.v1.html()` for SHAP plots?**
`shap.plots.text()` is an IPython HTML widget. It calls `IPython.display.display()` internally and never writes pixels to a matplotlib figure, so `plt.savefig()` always captures a blank canvas. The fix intercepts the HTML output via monkey-patching `IPython.display.display` and renders it inside a dark-themed iframe using `st.components.v1.html()`.

---

## Data Source

Economic Times financial headlines collected across 2022–2025, pre-filtered to remove lifestyle, entertainment, astrology, sports, and bollywood content before ESG analysis.
