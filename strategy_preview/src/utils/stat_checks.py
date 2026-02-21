"""Statistical quality checks for input data.

Provides non-blocking diagnostic checks:
  - Stationarity (Augmented Dickey-Fuller)
  - Multicollinearity (Variance Inflation Factor)
  - Serial Correlation (Ljung-Box)
  - Heteroskedasticity (Breusch-Pagan)

Each function returns a summary dict and prints a human-readable verdict.
"""
import warnings
import numpy as np
import pandas as pd


def check_stationarity(series: pd.Series, significance: float = 0.05) -> dict:
    """Augmented Dickey-Fuller test for stationarity.

    Returns dict with 'adf_stat', 'p_value', 'is_stationary'.
    """
    from statsmodels.tsa.stattools import adfuller
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = adfuller(series.dropna(), autolag="AIC")
    adf_stat, p_value = result[0], result[1]
    is_stationary = p_value < significance
    tag = "PASS" if is_stationary else "WARN"
    print(f"  [{tag}] ADF test on '{series.name}': stat={adf_stat:.4f}, "
          f"p={p_value:.4f} ({'stationary' if is_stationary else 'non-stationary'})")
    return {"adf_stat": adf_stat, "p_value": p_value, "is_stationary": is_stationary}


def check_multicollinearity(df: pd.DataFrame, threshold: float = 10.0) -> pd.Series:
    """Variance Inflation Factor for each column.

    VIF > threshold suggests problematic multicollinearity.
    Returns pd.Series of VIF values indexed by column name.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = df.select_dtypes(include=[np.number]).dropna()
    if X.shape[1] < 2:
        print("  [SKIP] VIF: fewer than 2 numeric columns.")
        return pd.Series(dtype=float)
    # Standardize to avoid numerical issues
    X_std = (X - X.mean()) / (X.std() + 1e-9)
    vifs = {}
    for i, col in enumerate(X_std.columns):
        try:
            vifs[col] = variance_inflation_factor(X_std.values, i)
        except Exception:
            vifs[col] = float('nan')
    s = pd.Series(vifs, name="VIF").sort_values(ascending=False)
    high = s[s > threshold]
    if len(high) > 0:
        print(f"  [WARN] {len(high)} features with VIF > {threshold}: "
              f"{list(high.index[:5])}{'...' if len(high) > 5 else ''}")
    else:
        print(f"  [PASS] All VIF values <= {threshold}.")
    return s


def check_serial_correlation(series: pd.Series, lags: int = 10,
                              significance: float = 0.05) -> dict:
    """Ljung-Box test for serial (auto-) correlation in residuals.

    Returns dict with 'lb_stat', 'lb_pvalue', 'has_autocorrelation'.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    clean = series.dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = acorr_ljungbox(clean, lags=[lags], return_df=True)
    lb_stat = float(result["lb_stat"].iloc[0])
    lb_pvalue = float(result["lb_pvalue"].iloc[0])
    has_ac = lb_pvalue < significance
    tag = "WARN" if has_ac else "PASS"
    print(f"  [{tag}] Ljung-Box (lag={lags}) on '{series.name}': "
          f"stat={lb_stat:.2f}, p={lb_pvalue:.4f} "
          f"({'autocorrelation detected' if has_ac else 'no significant autocorrelation'})")
    return {"lb_stat": lb_stat, "lb_pvalue": lb_pvalue, "has_autocorrelation": has_ac}


def check_heteroskedasticity(series: pd.Series, significance: float = 0.05) -> dict:
    """Breusch-Pagan test for heteroskedasticity.

    Fits a simple trend-based OLS and checks residual variance.
    Returns dict with 'bp_stat', 'bp_pvalue', 'has_heteroskedasticity'.
    """
    from statsmodels.stats.diagnostic import het_breuschpagan
    import statsmodels.api as sm
    clean = series.dropna()
    n = len(clean)
    X = sm.add_constant(np.arange(n).reshape(-1, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.OLS(clean.values, X).fit()
        bp_stat, bp_pvalue, _, _ = het_breuschpagan(model.resid, X)
    has_het = bp_pvalue < significance
    tag = "WARN" if has_het else "PASS"
    print(f"  [{tag}] Breusch-Pagan on '{series.name}': "
          f"stat={bp_stat:.2f}, p={bp_pvalue:.4f} "
          f"({'heteroskedastic' if has_het else 'homoskedastic'})")
    return {"bp_stat": bp_stat, "bp_pvalue": bp_pvalue,
            "has_heteroskedasticity": has_het}


def run_all_checks(panel: pd.DataFrame, ticker: str) -> dict:
    """Run all statistical checks on the panel and target series.

    Returns a dict of all results. Prints diagnostics to stdout.
    """
    print("--- Statistical Quality Checks ---")
    results = {}
    s = panel[ticker]
    results["stationarity_price"] = check_stationarity(s)
    delta = s.diff().dropna()
    delta.name = f"{ticker}_delta"
    results["stationarity_delta"] = check_stationarity(delta)
    results["serial_corr"] = check_serial_correlation(delta)
    results["heteroskedasticity"] = check_heteroskedasticity(s)
    # VIF on a subset of columns to keep it fast
    numeric_cols = panel.select_dtypes(include=[np.number]).columns[:30]
    results["vif"] = check_multicollinearity(panel[numeric_cols])
    print("--- End of Checks ---\n")
    return results
