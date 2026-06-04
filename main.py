"""Use Latin Hypercube Sampling (LHS) to explore the sensitivity of evaluation
metrics to streamflow prediction errors."""
from pathlib import Path
import pandas as pd
import numpy as np
import numpy.typing as npt
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import colorcet as cc

RANDOM_NUMBER_GENERATOR: np.random.Generator = np.random.default_rng(seed=2026)
"""Module-wide random number generator."""

COLORS: list[str] = cc.b_glasbey_hv
"""Default categorical colors."""

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
    errors = noise * RANDOM_NUMBER_GENERATOR.normal(size=len(time_series))
    prediction = gain * (time_series + errors).shift(shift)
    prediction[prediction <= 0.0] = 0.01
    return prediction.rolling(window=window).mean()

def nash_sutcliffe_efficiency(
        y_true: npt.NDArray[np.float64],
        y_pred: npt.NDArray[np.float64]
) -> float:
    """Compute the Nash-Sutcliffe Model Efficiency."""
    num = np.sum((y_true - y_pred) ** 2.0)
    den = np.sum((y_true - np.mean(y_true)) ** 2.0)
    return 1.0 - num / den

def normalized_nnse(
        y_true: npt.NDArray[np.float64],
        y_pred: npt.NDArray[np.float64]
) -> float:
    """Compute the Normalized Nash-Sutcliffe Model Efficiency."""
    return 1.0 / (2.0 - nash_sutcliffe_efficiency(y_true, y_pred))

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

def show_timeseries(
        data_source: Path
) -> None:
    """Show example time series."""
    # Load observations
    data = load_data(data_source)

    # Resample to hourly
    data = data.resample("1h").mean()

    # Make erroneous predictions
    data["prediction"] = transform_time_series(
        time_series=data["observation"],
        gain=1.0,
        shift=0,
        noise=1.0,
        window=1
    )

    # Drop NA
    data = data.dropna()

    # Plot
    fig = go.Figure([
        go.Scatter(x=data.index, y=data.observation, mode="lines", name="Observations"),
        go.Scatter(x=data.index, y=data.prediction, mode="lines", name="Predictions")
    ])
    # fig.update_yaxes(type="log")
    fig.show()

def main(
        data_source: Path,
        plot: bool = False,
        sample_size: int = 5
) -> None:
    """Load data, simulate predictions, and score."""
    # Load observations
    data = load_data(data_source)

    # Resample to hourly
    data = data.resample("1h").mean()

    # Generate sample space
    lower_bounds = [0.001, -24.0, 0.0, 1]
    upper_bounds = [3.0, 24.0, 1.0, 24]
    sampler = stats.qmc.LatinHypercube(d=4, rng=RANDOM_NUMBER_GENERATOR)
    samples = sampler.random(n=sample_size)
    scaled_samples = stats.qmc.scale(
        samples,
        lower_bounds,
        upper_bounds
        )

    # Round shift and window to whole numbers
    scaled_samples[:, 1] = np.round(scaled_samples[:, 1])
    scaled_samples[:, 3] = np.round(scaled_samples[:, 3])

    # Process sample sets
    nse_scores = []
    nnse_scores = []
    kge_scores = []
    kge_np_scores = []
    for gain, shift, noise, window in scaled_samples:
        # Make erroneous predictions
        df = pd.DataFrame({
            "observation": data["observation"],
            "prediction": transform_time_series(
                time_series=data["observation"],
                gain=gain,
                shift=int(shift),
                noise=noise,
                window=int(window)
            )
        })

        # Drop NA
        df = df.dropna()

        # Compute metrics
        nse = nash_sutcliffe_efficiency(
            y_true=df["observation"].to_numpy(),
            y_pred=df["prediction"].to_numpy()
        )
        nse_scores.append(nse)
        nnse_scores.append(1.0 / (2.0 - nse))
        kge_scores.append(kling_gupta_efficiency(
            y_true=df["observation"].to_numpy(),
            y_pred=df["prediction"].to_numpy()
        ))
        kge_np_scores.append(non_parametric_kge(
            y_true=df["observation"].to_numpy(),
            y_pred=df["prediction"].to_numpy()
        ))

    # Plot
    if plot:
        rows: int = 2
        columns: int = 2
        fig = make_subplots(rows=rows, cols=columns)
        showlegend = True
        xlabels = ["Bias", "Shift (h)", "Scale of noise (CFS)", "Smoothing window (h)"]

        for n in range(4):
            # Rows and columns
            r, c = (n // columns) + 1, (n % columns) + 1

            # Add traces
            fig.add_trace(go.Scatter(x=scaled_samples[:, n], y=nse_scores, name="NSE", mode="markers", marker={"color": COLORS[0]}, showlegend=showlegend, legendgroup="NSE"), row=r, col=c)
            fig.add_trace(go.Scatter(x=scaled_samples[:, n], y=nnse_scores, name="NNSE", mode="markers", marker={"color": COLORS[5]}, showlegend=showlegend, legendgroup="NNSE"), row=r, col=c)
            fig.add_trace(go.Scatter(x=scaled_samples[:, n], y=kge_scores, name="KGE", mode="markers", marker={"color": COLORS[1]}, showlegend=showlegend, legendgroup="KGE"), row=r, col=c)
            fig.add_trace(go.Scatter(x=scaled_samples[:, n], y=kge_np_scores, name="KGE_NP", mode="markers", marker={"color": COLORS[2]}, showlegend=showlegend, legendgroup="KGE_NP"), row=r, col=c)

            # Update axis titles
            fig.update_xaxes(title_text=xlabels[n], row=r, col=c)
            fig.update_yaxes(title_text="Score", range=[-1.0, 1.0], row=r, col=c)
            showlegend = False

        fig.show()

if __name__ == "__main__":
    # Recent year
    # main(data_source=Path("data/USGS_02146470_recent.csv.gz"))
    # show_timeseries(data_source=Path("data/USGS_02146470_recent.csv.gz"))

    # Dry year
    # main(data_source=Path("data/USGS_02146470_WY2001.csv.gz"))
    # show_timeseries(data_source=Path("data/USGS_02146470_WY2001.csv.gz"))

    # Wet year
    # main(data_source=Path("data/USGS_02146470_WY2020.csv.gz"))
    show_timeseries(data_source=Path("data/USGS_02146470_WY2020.csv.gz"))

    # Median year
    # main(data_source=Path("data/USGS_02146470_WY2010.csv.gz"))
    # show_timeseries(data_source=Path("data/USGS_02146470_WY2010.csv.gz"))
