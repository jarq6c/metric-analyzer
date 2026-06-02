"""Use Latin Hypercube Sampling (LHS) to explore the sensitivity of evaluation
metrics to streamflow prediction errors."""
from pathlib import Path
import pandas as pd
import numpy as np
import numpy.typing as npt
from scipy import stats
import plotly.graph_objects as go

RANDOM_NUMBER_GENERATOR: np.random.Generator = np.random.default_rng(seed=2026)
"""Module-wide random number generator."""

def load_data(ifile: Path) -> pd.DataFrame:
    """Load USGS WaterData CSV into a pandas.DataFrame."""
    return pd.read_csv(
        ifile,
        usecols=["time", "value"],
        parse_dates=["time"],
        dtype={
            "value": float
        }
    ).drop_duplicates("time").set_index("time").rename(
        columns={"value": "observation"})

def transform_time_series(
        time_series: pd.Series,
        gain: float = 1.0,
        shift: int = 0,
        noise: float = 0.0,
        window: int = 1
) -> pd.Series:
    """Transform a time series."""

    errors = 1.0 + noise * RANDOM_NUMBER_GENERATOR.normal(size=len(time_series))
    prediction = gain * (time_series * errors).shift(shift)
    prediction[prediction < 0.0] = 0.0
    return prediction.rolling(window=window).mean()

def nash_sutcliffe_efficiency(
        y_true: npt.NDArray[np.float64],
        y_pred: npt.NDArray[np.float64]
) -> float:
    """Compute the Nash-Sutcliffe Model Efficiency."""
    num = np.sum((y_true - y_pred) ** 2.0)
    den = np.sum((y_true - np.mean(y_true)) ** 2.0)
    return 1.0 - num / den

def kling_gupta_efficiency(
        y_true: npt.NDArray[np.float64],
        y_pred: npt.NDArray[np.float64]
) -> float:
    """Compute the Kling-Gupta Model Efficiency."""
    corr = stats.pearsonr(y_true, y_pred).statistic
    vari = np.std(y_pred) / np.std(y_true)
    bias = np.mean(y_pred) / np.mean(y_true)
    return 1.0 - np.sqrt((corr - 1) ** 2.0 + (vari - 1) ** 2.0 + (bias - 1) ** 2.0)

def fdc_variability(
        y_true: npt.NDArray[np.float64],
        y_pred: npt.NDArray[np.float64]
) -> float:
    """Compute the FDC-normalized flow variability."""
    size = len(y_true)
    y_true_sorted = np.sort(y_true) / (size * np.mean(y_true))
    y_pred_sorted = np.sort(y_pred) / (size * np.mean(y_pred))
    return 1.0 - 0.5 * np.sum(np.abs(y_true_sorted - y_pred_sorted))

def non_parametric_kge(
        y_true: npt.NDArray[np.float64],
        y_pred: npt.NDArray[np.float64]
) -> float:
    """Compute a (mostly) non-parametric KGE."""
    corr = stats.spearmanr(y_true, y_pred).statistic
    vari = fdc_variability(y_true=y_true, y_pred=y_pred)
    bias = np.mean(y_pred) / np.mean(y_true)
    return 1.0 - np.sqrt((corr - 1) ** 2.0 + (vari - 1) ** 2.0 + (bias - 1) ** 2.0)

def main(
        data_source: Path,
        plot: bool = True
) -> None:
    """Load data, simulate predictions, and score."""
    # Load observations
    data = load_data(data_source)

    # Resample to hourly
    data = data.resample("1h").mean()

    # Make erroneous predictions
    data["prediction"] = transform_time_series(
        time_series=data["observation"],
        gain=2.0,
        shift=1,
        noise=0.1,
        window=3
    )

    # Drop NA
    data = data.dropna()

    # Compute metrics
    nse = nash_sutcliffe_efficiency(
        y_true=data["observation"].to_numpy(),
        y_pred=data["prediction"].to_numpy()
    )
    kge = kling_gupta_efficiency(
        y_true=data["observation"].to_numpy(),
        y_pred=data["prediction"].to_numpy()
    )
    kge_np = non_parametric_kge(
        y_true=data["observation"].to_numpy(),
        y_pred=data["prediction"].to_numpy()
    )
    print(nse)
    print(kge)
    print(kge_np)

    # Plot
    if plot:
        fig = go.Figure([
            go.Scatter(x=data.index, y=data.observation, name="Observation", mode="lines"),
            go.Scatter(x=data.index, y=data.prediction, name="Prediction", mode="lines")
        ])
        fig.show()

if __name__ == "__main__":
    main(
        data_source=Path("data/USGS_02146470.csv")
    )
