# vrp-calendar-spread

Predicting the profitability of **pre-earnings ATM call calendar spreads** with tabular ML, a Two-Stream GRU on per-ticker earnings history, and a zero-shot LLM benchmark. 9,119 trades · 476 tickers · 2018–2025.

**Canvas Project Number:** 36

---

## Motivation

Retail option trading has surged since the elimination of brokerage commissions in 2019, with roughly 48 million options traded per day in 2024 (up from 3.6 million in 2008). This has inflated demand for short-dated calls and pushed Implied Volatility (IV) systematically above Subsequent Realized Volatility (SRV) for many names — the *Volatility Risk Premium* (VRP). The project investigates whether neural networks can learn patterns in volatility term structure, liquidity, and order flow that reliably identify profitable VRP-capture trades.

The strategy: sell the near-term option (capture IV crush after earnings) and buy the far-term option (retain vega exposure). Enter the day before earnings, exit the day after.

The project is novel in combining:
- Earnings event data
- Millisecond-level option microstructure data
- Volatility term structure metrics (near vs. far IV)
- A neural network trained specifically for volatility mispricing classification

---

## Repository Contents

| File | Description |
|------|-------------|
| `short_vol_calendar_spread_final.ipynb` | Main end-to-end notebook (Milestones 2 → 4) |
| `README.md` | This file |

---

## Notebook Structure

### Part I — Data, Target Construction, and EDA
Loads the calendar-spread CSV, handles missing values, reconstructs the canonical **quarter-spread P&L target** (a midpoint-execution assumption representing a patient retail trader), performs EDA, and exports the model-ready dataset.

### Part II — Modeling Pipeline
Chronological train (≤2022) / validation (2023) / test (2024–25) splits. Five model families:
- L2-regularized **Logistic Regression** (baseline)
- **LightGBM** (best single model by PR-AUC)
- **MLP**
- **MLP + Ticker Embeddings**
- **Two-Stream GRU** (see below)
- **Rank-average ensembles** (`LightGBM + GRU` and `LightGBM + GRU + MLPE`)

### Part III — Interpretability and EV-Aware Pipeline
- LightGBM gain-based feature importance
- **Magnitude-weighted classifier** — each row weighted by `|percent_profit_loss|`, so an −80% loss influences the loss function 8× more than a −10% loss (cost-sensitive learning, not relabeling)
- **Loss-avoider model** — second binary classifier predicting `is_safe = pnl ≥ −0.30`, combined with the main ranker as a soft veto: `score = P(profitable) × P(safe)`
- **7-ranker × 3-scenario sensitivity grid** with bootstrap CIs and validation-tuned cutoffs (worst-case fills, quarter-spread, quarter-spread + −50% stop)
- **Zero-shot LLM benchmark** (see below)

---

## Two-Stream GRU

For each earnings event, the model builds a short causal sequence of that ticker's **prior** earnings events (up to `MAX_LOOKBACK = 4`) and feeds it through a GRU. The GRU's hidden state is concatenated with the current event's features and a learned ticker embedding for the final prediction. This lets the model condition on patterns like *"AAPL's IV crush has been shrinking the last four quarters"* or *"this ticker's `vrp_z_score` is regime-shifting."*

**Architecture:**
- Sequence input → `Masking(0.0)` → `GRU(32 units, dropout=0.55, recurrent_dropout=0.35)` with L2 on kernel and recurrent weights
- Current-event features → `Dense(32, relu)` + dropout
- Ticker → `Embedding(n_tickers, 8)` with L2
- Concatenate all three → dense head → sigmoid

**Leakage guard.** The history source is strictly causal:
- Training rows look back at **earlier training rows only**
- Validation rows look back at **training rows only**
- Test rows look back at **training + validation rows only**

No future information is ever embedded in the history.

**Why this design.** Calendar spreads have a per-name component (IV crush severity, post-earnings drift) that doesn't fit cleanly into hand-engineered features. The GRU captures temporal patterns within a ticker; the embedding captures static per-ticker bias; the dense stream handles the current event's microstructure. On the held-out 2024–2025 test set the GRU reaches PR-AUC = 0.3805, marginally below LightGBM (0.3875) but contributes complementary ranking information — the `LightGBM + GRU` ensemble has the best top-decile Sharpe.

---

## Zero-Shot LLM Benchmark

Part III includes a controlled comparison against off-the-shelf LLMs. The question: **does numeric / math-instruction tuning help a generalist LLM rank tabular options trades?**

**Setup.** On a 50% sample of the 2024–2025 test set (982 trades, seeded), each trade's 22 numeric features are rendered as a `key: value` block and passed to two 7B-parameter Qwen 2.5 instruction-tuned models. Holding model size and base architecture constant lets the comparison isolate the effect of post-training objective.

| Model | Role |
|-------|------|
| `Qwen/Qwen2.5-7B-Instruct` | Generalist zero-shot baseline |
| `Qwen/Qwen2.5-Math-7B-Instruct` | Same architecture, post-trained for numeric / step-by-step math reasoning |

**Prompting trick — assistant prefill.** Both models receive a strict system prompt requesting a single decimal in `[0, 1]`. The chat template's assistant turn is then **pre-filled with `" 0."`** so greedy decoding has to *complete a probability* rather than start a chain-of-thought derivation. Without this, Qwen-Math reframes the prompt as a formal math problem and emits a derivation. With the prefill, an 8-token generation budget is sufficient.

**Results on the 982-trade sample (positive rate = 0.293):**

| Model | PR-AUC | ROC-AUC | Prec@10% |
|-------|--------|---------|----------|
| LightGBM (trained) | **0.398** | **0.636** | **0.388** |
| Qwen2.5-Math-7B (zero-shot) | 0.316 | 0.523 | 0.367 |
| Qwen2.5-7B (zero-shot) | 0.292 | 0.494 | 0.265 |

**Takeaway.** The math-tuned LLM produces a real but modest ranking lift over the 0.293 random baseline; the generalist is statistically indistinguishable from random. The score-distribution summaries explain why directly: the math model emits **dispersed, decisive probabilities** (mean 0.488, std 0.219), while the generalist clusters tightly around 0.762 with std 0.062 — it almost always says "probably profitable" and barely discriminates. Math-style instruction tuning is rewarded for committing to numeric answers, and that habit transfers — weakly — to a numeric ranking task. Neither LLM produces a positive-EV cell in the 7-ranker × 3-scenario grid under quarter-spread execution.

**Conclusion:** tabular options prediction with engineered numeric features is not a task that off-the-shelf zero-shot LLMs solve, even with finance-adjacent or numeric-reasoning post-training. A trained tabular model wins, and the project documents *why* it wins.

---

## Data

| Source | Use |
|--------|-----|
| Finnhub | Quarterly earnings dates and times |
| VolVue | Historical volatility estimators (IV, RV, ...) |
| Theta Data | Live and historical option prices, greeks, IV at millisecond intervals |
| Alpha Vantage | Historical daily OHLCV for underlyings |

**Coverage:** 476 companies · ~4 earnings events per company per year · 2018–2025 · **9,119 trade observations**.

**Target:** `percent_profit_loss` — actual P&L from closing the calendar spread the day after earnings, computed under the quarter-spread execution assumption (approximately mid-of-spread fills on both legs).

---

## Reproducibility

- **Environment:** Designed for Google Colab GPU runtime. *Restart kernel + Run All* should succeed top-to-bottom.
- **Dependencies:** Installed in the first code cell — `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `tensorflow`, `transformers ≥ 4.45`, `accelerate ≥ 0.34`, `seaborn`, `matplotlib`, `scipy`.
- **Random seed:** `SEED = 42`, fixed for NumPy, Python `random`, TensorFlow, and the LLM sampling RNG.
- **GPU note:** The LLM benchmark loads two 7B-parameter models in `bf16`. A T4 (16 GB) is the practical floor; an A100 is comfortable. Each model is loaded, used, and freed before the next is loaded.
- **Data path:** Update the Google Drive path in Step 1 if you are not running on Colab.

---

## How to Run

1. Open `short_vol_calendar_spread_final.ipynb` in Google Colab (GPU runtime — A100 preferred for the LLM cells).
2. Mount your Google Drive when prompted in Step 1.
3. Update the CSV path to point to your copy of `calendar_spread_model_ready_raw.csv`.
4. *Runtime → Run all*. The LLM benchmark cells are the slowest; skip them by setting `RUN_LLM = False` if you only want the tabular pipeline.

---

## Methodological Discipline

- All hyperparameters, callbacks, cutoffs, and gating thresholds were tuned on the 2023 validation period or fixed ex ante with stated rationale (loss-avoider target −0.30, stop level −0.50).
- The 2024–25 test set was evaluated **once**, after the pipeline was frozen.
- Bootstrap 95% CIs are reported for every test EV.
- The worst-case stress column is always shown alongside more favorable execution scenarios, so the sensitivity floor is visible.
