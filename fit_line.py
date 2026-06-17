import numpy as np
from scipy import stats
import pandas as pd
from tabulate import tabulate
from sklearn.model_selection import train_test_split
import arviz as az

def nash_sutcliffe_efficiency(y_true, y_pred):
    """Nash-Sutcliffe Model Efficiency."""
    numerator = np.sum((y_true - y_pred) ** 2.0)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2.0)
    return 1.0 - numerator / denominator

def linear_model(x, slope, intercept):
    """Basic linear model."""
    return slope * x + intercept

def main():
    """Main."""
    # Random numbers
    rng = np.random.default_rng(seed=2026)

    # Get data
    data_points = 1000
    true_slope = 0.7
    true_intercept = 2.3
    x = rng.uniform(0.0, 10.0, data_points)
    y = linear_model(x, true_slope, true_intercept) + rng.gumbel(0.0, 0.25, data_points)

    # Categorize forcing
    quantiles = np.quantile(x, [0.25, 0.5, 0.75])
    groups = np.digitize(x, bins=quantiles)

    # Partition data with stratification
    x_cal, x_val, y_cal, y_val = train_test_split(
        x, y, test_size=0.4, random_state=2026, stratify=groups
    )

    # Generate sample space
    lower_bounds = [-1.0, -5.0]
    upper_bounds = [1.0, 5.0]
    sampler = stats.qmc.LatinHypercube(d=2, rng=rng)
    samples = sampler.random(n=5000)
    scaled_samples = stats.qmc.scale(
        samples,
        lower_bounds,
        upper_bounds
        )

    # Score samples
    results = []
    for slope, intercept in scaled_samples:
        y_pred = linear_model(x_cal, slope, intercept)
        nse = nash_sutcliffe_efficiency(y_cal, y_pred)
        results.append((slope, intercept, nse))

    # Pick best run
    df = pd.DataFrame.from_records(results, columns=["slope", "intercept", "nse"])
    best_run = df.iloc[df["nse"].idxmax()]

    # Compute credible intervals on "behavioral" parameter sets
    df_behavioral = df[df["nse"] > 0.0].copy()
    slope_samples = df_behavioral["slope"].to_numpy()
    intercept_samples = df_behavioral["intercept"].to_numpy()
    weights = df_behavioral["nse"].to_numpy()
    weights = weights / np.sum(weights)

    # Resample parameters to generate posteriors
    resampled_idx = rng.choice(
        len(df_behavioral), size=10_000, replace=True, p=weights
    )
    resampled_slopes = slope_samples[resampled_idx]
    resampled_intercepts = intercept_samples[resampled_idx]

    # Compute HDI
    hdi_slope = az.hdi(resampled_slopes, prob=0.89)
    hdi_intercept = az.hdi(resampled_intercepts, prob=0.89)

    # Validation
    y_pred = linear_model(x_val, best_run.slope, best_run.intercept)
    nse = nash_sutcliffe_efficiency(y_val, y_pred)

    # Print results
    print(" Results ".center(31, "-"))
    df_print = pd.DataFrame({
        "True": [true_slope, true_intercept],
        "Best": [best_run.slope, best_run.intercept],
        "HDI Lower (89%)": [hdi_slope[0], hdi_intercept[0]],
        "HDI Upper (89%)": [hdi_slope[1], hdi_intercept[1]]
    }, index=["slope", "intercept"])
    print(tabulate(df_print, headers="keys", tablefmt="psql", floatfmt=".3f"))
    print(f"Calibration NSE: {best_run.nse:.4f}")
    print(f"Validation NSE: {nse:.4f}")

if __name__ == "__main__":
    main()
