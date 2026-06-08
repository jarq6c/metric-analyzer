"""Calibrate Hymod."""
import numpy as np
import plotly.graph_objects as go
import panel as pn
from superflexpy.implementation.models.hymod import model as hymod

def main() -> None:
    """Run Hymod."""
    # Generate precipitation
    rng = np.random.default_rng(seed=2026)
    precipitation = np.zeros(100)
    precipitation[:10] = rng.integers(10, size=10)
    precipitation[25:30] = rng.integers(20, size=5)
    precipitation[40:60] = rng.integers(5, size=20)
    precipitation[80:83] = rng.integers(low=30, high=50, size=3)

    # Assume constant potential evapotranspiration
    potential_evapotranspiration = np.ones_like(precipitation) * 2.0

    # Assign input
    hymod.set_input([precipitation, potential_evapotranspiration])

    # Set timestep
    hymod.set_timestep(1.0)

    # Run the model
    hymod.reset_states()
    output = hymod.get_output()

    # Plot
    time = np.arange(precipitation.size)
    uz_e = hymod.call_internal(id='uz', method='get_AET')[0]
    upper_figure = go.Figure([
        go.Bar(x=time, y=precipitation, name="Precipitation"),
        go.Scatter(x=time, y=potential_evapotranspiration, name="PET", mode="lines")
    ],
    layout=go.Layout(
        xaxis={
            "title": {
                "text": "Time (days)"
            }
        },
        yaxis={
            "title": {
                "text": "Inputs (mm/day)"
            }
        }
    ))
    lower_figure = go.Figure([
        go.Scatter(x=time, y=output[0], name="Total outflow", mode="lines"),
        go.Scatter(x=time, y=uz_e, name="AET", mode="lines")
    ],
    layout=go.Layout(
        xaxis={
            "title": {
                "text": "Time (days)"
            }
        },
        yaxis={
            "title": {
                "text": "Outputs (mm/day)"
            }
        }
    ))

    # Render
    display = pn.Column(
        pn.pane.Plotly(upper_figure),
        pn.pane.Plotly(lower_figure)
    )
    pn.serve(display)

if __name__ == "__main__":
    main()
