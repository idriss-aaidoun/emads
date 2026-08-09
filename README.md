# EMADS – Explainable Multi-Agent Data Science System

## Overview

EMADS is a **sequential multi-agent pipeline** that automates the day-to-day work of a
Data Scientist on a tabular CSV dataset — profiling, cleaning, model selection,
hyperparameter tuning, evaluation, and explanation — while documenting **why** every
automatic decision was made along the way.

Unlike a typical AutoML tool, EMADS is not optimizing purely for the best score. Its
core research question is **explainability and reliability of automated decisions**:
every non-trivial choice the pipeline makes (which column is the target, which model
won, which imputation strategy was used, how confident the system actually is) is
recorded as a structured, human-readable decision with a statistically-grounded
confidence score — not a black box that just spits out a trained model.

It is also **not** a dynamic tool-calling agent. `SupervisorAgent` runs a fixed,
linear [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph` of 9
specialized agents, each reading from and writing to one shared state object
(`EMADSState`). Nothing is re-planned or re-ordered at runtime — the pipeline is
deterministic in *shape*, even where individual decisions rely on statistics or an
LLM.

---

## What it actually does

1. **Upload a CSV** through the Streamlit dashboard.
2. Choose a problem type — **Classification**, **Regression**, **Clustering**,
   **Anomaly Detection**, or leave it on **Auto-detect** and let the pipeline infer
   target column + problem type from the data itself.
3. Click **Run EMADS Pipeline** and watch live, per-agent progress as 9 agents run in
   sequence.
4. Get back: a full interactive dashboard (EDA plots, model comparison table, SHAP
   explanations, a global confidence verdict, a decisions log, an LLM-arbitration log),
   a downloadable **PDF report**, and a downloadable **trained model** (`.pkl`).

---

## The 9-agent pipeline

```
Supervisor (LangGraph StateGraph — linear, fixed order)

 1. DataUnderstandingAgent   — loads the CSV, profiles columns, flags data-quality
                               issues, detects target column + problem type
 2. EDAAgent                 — descriptive stats, outlier detection (IQR), plots,
                               LLM-narrated summary
 3. PreprocessingAgent       — dedup, imputation, categorical encoding, scaling
                               (100% deterministic, no LLM)
 4. ModelSelectionAgent      — compares candidate algorithms via k-fold CV
                               (supervised) or silhouette score (unsupervised)
 5. HyperparameterAgent      — Optuna search on ONLY the algorithm already selected
 6. EvaluationAgent          — holdout metrics + a fresh CV pass for stability
 7. ExplainabilityAgent      — SHAP + permutation importance, cross-checked against
                               each other, LLM-narrated in plain English
 8. MetaEvaluatorAgent       — audits the whole run, produces one global confidence
                               verdict (no LLM — a deterministic weighted formula)
 9. ReportingAgent           — assembles everything into the final PDF report
```

Every agent extends the same `BaseAgent` contract: it reads a shared `EMADSState`,
returns a **partial** state update, and — for any decision worth explaining — appends
an `AgentDecision(agent_name, decision, reasoning, confidence)` to the state's
decisions log. That log is what powers the "Decisions Log" tab and the PDF report.

### Supervised vs. unsupervised, without a conditional graph

Classification, regression, clustering, and anomaly detection all run through the
**same fixed 9 steps** — there is no separate graph or skipped agent per problem type.
The supervised/unsupervised distinction is handled with internal branching *inside*
each concerned agent (e.g. `ModelSelectionAgent` cross-validates for supervised
problems but does a single fit + silhouette score for unsupervised ones, since
DBSCAN/LocalOutlierFactor have no out-of-sample `.predict()`). Choosing clustering or
anomaly detection is **strictly opt-in** through the sidebar — nothing in the pipeline
ever infers "you probably want clustering" from the raw CSV on its own; when a user
leaves the problem-type selector on Auto-detect, the sidebar can only *suggest*
considering clustering (a purely informational hint, see below), never select it.

---

## Explainability, end to end

This is the actual thesis of the project, not a side feature:

- **`AgentDecision` log** — every meaningful automatic choice (target column, problem
  type, imputation strategy, winning model, hyperparameters, final verdict) is
  recorded with a plain-English `reasoning` string and a `confidence` score, visible
  in the UI's "Decisions Log" tab and in the PDF report.
- **Confidence is derived from a real statistic, never hardcoded** — a score margin, a
  Wilcoxon p-value, a missing-value ratio, a cardinality ratio, a CV/holdout gap, etc.
  The only fixed `confidence = 1.0` values left in the codebase are genuine facts
  (e.g. "the user explicitly picked this target column"), never disguised
  heuristics.
- **LLM arbitration is conditional and logged, not decorative** — most of the pipeline
  is fully deterministic. The LLM only steps in at specific, well-defined uncertainty
  points: no target column could be confidently inferred, or two candidate models tied
  statistically (Wilcoxon signed-rank test, `p > 0.05`, 10-fold CV). Every such call is
  recorded in a dedicated `llm_arbitration_log`, visible in its own UI tab, and — when
  it changes the outcome (e.g. picks the statistical runner-up for interpretability
  reasons) — the reasoning text explicitly says so rather than silently claiming the
  promoted model "scored best."
- **SHAP cross-checked against permutation importance** — `ExplainabilityAgent`
  computes both, on the same test set, and reports a Spearman agreement score between
  them. Low agreement is itself a signal, not hidden.
- **`MetaEvaluatorAgent` — one global confidence verdict.** Instead of asking a reader
  to reconcile a dozen scattered per-agent confidence scores, this deterministic
  (no-LLM) agent audits the completed run and produces a single weighted verdict:
  40% evaluation metrics, 30% SHAP/permutation agreement, 30% penalty per LLM
  arbitration logged. Below its threshold, it points at the most plausible root cause
  (e.g. "target detection had low confidence — consider re-running with an explicit
  target column") instead of a vague "something might be off."
- **Runs never fail silently.** `SupervisorAgent` stops the pipeline at the first
  agent that raises, with a clear `RuntimeError` naming the step and the original
  exception, rather than limping forward with an incomplete state.

---

## Engineering robustness work

A sample of the correctness/reliability work behind the pipeline, since a research
project on *reliability* has to hold up to real, messy datasets:

- **XGBoost + non-contiguous class labels.** XGBoost's sklearn API requires class
  labels to be contiguous integers starting at 0 — a real dataset with a 1–5 rating
  target used to make XGBoost silently fail CV and disappear from every model
  comparison. Fixed with an internal `LabelEncoder`-based wrapper
  (`XGBClassifierWithLabelEncoding`) that remaps transparently on fit/predict, kept
  consistent between `ModelSelectionAgent` and `HyperparameterAgent`. Verified
  LightGBM has no equivalent constraint.
- **Wilcoxon significance floor.** With too few CV folds, the Wilcoxon signed-rank
  test used to arbitrate close model comparisons has a hard mathematical floor on how
  small its p-value can get — below 6 folds, it could *never* report significance even
  when every fold agreed. `CV_FOLDS` was raised to 10 to leave real statistical
  headroom, at negligible extra runtime cost (folds parallelize via
  `cross_val_score`'s `n_jobs=-1`).
- **Stop-on-failure, not degrade-silently**, described above — enforced with a
  dedicated regression test (`tests/test_pipeline_edge_cases.py`).
- **A lightweight, purely informational sidebar hint** — when a user leaves the
  problem type on Auto-detect and no column looks like an obvious prediction target,
  the sidebar suggests considering Clustering / Anomaly Detection. It reuses the exact
  same confidence formula as `DataUnderstandingAgent`'s own fallback logic (so the UI
  and the pipeline never disagree about what "looks like a target" means), computed
  instantly client-side on the already-loaded preview — no LLM call, no extra latency,
  and it never presets or decides anything on the user's behalf.

---

## Tech stack

| Layer | Choice |
|---|---|
| Orchestration | LangGraph (`StateGraph`), LangChain |
| UI | Streamlit (custom theme, dark mode, live per-agent progress) |
| Modeling | scikit-learn, XGBoost, LightGBM |
| Hyperparameter search | Optuna (TPE sampler) |
| Explainability | SHAP, permutation importance |
| Stats | SciPy (Wilcoxon signed-rank test) |
| Plotting | Matplotlib, Seaborn |
| PDF report | ReportLab |
| LLM provider | Groq API (with a fully deterministic offline fallback per agent if no key is set) |

No database, no persistent memory layer, no cloud deployment, no dynamic tool-calling
`tools/` directory — all deliberately out of scope for this version (see `CLAUDE.md`).

---

## Project structure

```
app/
├── core/
│   ├── agents/            # the 9 pipeline agents + BaseAgent contract
│   ├── state/              # EMADSState (shared TypedDict) + AgentDecision
│   └── supervisor/         # SupervisorAgent — builds & runs the LangGraph graph
├── services/
│   ├── llm_service.py       # Groq client + deterministic offline fallbacks
│   └── pdf_service.py       # renders the final report (no decision logic)
├── ui/
│   ├── streamlit_app.py     # dashboard: sidebar, live progress, 9 result tabs
│   └── theme.py             # dark/light theme, status badges, styled tables
└── utils/                   # plot_utils, file_utils, logger — thin, no premature abstraction

tests/                       # unittest suite covering agents, edge cases, arbitration
data/                        # uploads, pipeline outputs (plots, cleaned data, model)
logs/                        # per-run log files
reports/                     # generated PDF reports
```

---

## Getting started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (optional) Add LLM credentials — the pipeline runs fine without this,
#    falling back to deterministic offline summaries for every LLM step.
echo "LLM_API_KEY=gsk_..." > .env

# 3. Launch the dashboard
streamlit run app/ui/streamlit_app.py

# — or run the pipeline headless on a bundled mock dataset —
python run.py
```

### Running the tests

```bash
python -m unittest discover tests
```

---

## Dashboard tabs

Once a run completes, results are organized into 9 tabs: **Overview** (dataset
summary, data-quality warnings, meta-evaluation badge), **EDA**, **Preprocessing**,
**Model** (comparison table + statistical verdict), **Evaluation**, **Explainability**
(SHAP + permutation), **LLM Arbitration Log**, **Decisions Log** (the full
`AgentDecision` trail), and **Report** (PDF + trained-model download).

---

## Project status

Research & development — built as part of a final-year engineering project (PFA).
Deliberately scoped: no agent registry, no parallel agent execution, no persistent
memory database, no cloud deployment — all explicitly deferred to a future version
rather than bolted on prematurely.

---

## Author

Idriss Aaidoun
