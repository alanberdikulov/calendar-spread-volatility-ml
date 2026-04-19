# Options calendar spread

This README documents **[exploratory analysis](./milestone2.ipynb)** for pre-earnings calendar-spread trades in this folder.

## Status

**Work in progress.** The notebook covers exploratory data analysis (EDA), data quality, preprocessing, class-balance exploration (including SMOTE), and collinearity diagnostics. **Model fitting is not finished here.**

Planned next step: **fit sequence or time-aware models** - e.g. **RNNs** (LSTM/GRU) or **Temporal Fusion Transformers (TFT)** - on appropriately constructed temporal features and **time-based train/validation/test splits** (aligned with the chronological / regime checks in the notebook).

## Contents (summary)

- Data load and schema (calendar spread trades, volatility and microstructure–style features)
- Missing-value handling and leakage-aware feature definitions
- Target construction and imbalance analysis
- Visual EDA, correlation / VIF / PCA-style redundancy checks
- Optional quant-style diagnostics (concentration, monthly regime, calendar clustering, monthly ACF)

## Running it

1. Use a Python environment with: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, and **`imbalanced-learn`** (for `imblearn` / SMOTE):

   ```bash
   python -m pip install imbalanced-learn
   ```

   In Jupyter, you can install into the **active kernel** with:

   ```python
   %pip install imbalanced-learn
   ```

2. Point the notebook at your `calendar_spread_trades_*.csv` (paths are set for Colab Drive vs local Windows in the code cells).

3. Run cells in order from the top (especially after changing the CSV path or environment).

## Outputs

Figures may be saved next to the notebook (e.g. `quant_monthly_regime.png`, `quant_monthly_acf.png`, and other plots produced in later sections), depending on which cells you execute.

---

*Modeling (RNN / TFT) is planned as a follow-on step.*
