# Define frames
import plotly.graph_objects as go
import numpy as np
import cupy as cp


def frame_args(duration):
    return {
        "frame": {"duration": duration},
        "mode": "immediate",
        "fromcurrent": True,
        "transition": {"duration": duration, "easing": "linear"},
    }


def slider_plot(grid, vol):
    ##No idea why this transposition works but it does if you use all the following notation
    # volume = vol.transpose(1,0,2)
    volume = vol

    r, c = volume[:, :, 0].shape

    x = np.linspace(0, grid.l[0].get(), grid.grid_spec[0])
    y = np.linspace(0, grid.l[1].get(), grid.grid_spec[1])

    nb_frames = grid.grid_spec[2]

    fig1 = go.Figure(
        frames=[
            go.Frame(
                data=go.Surface(
                    z=(grid.l[2].get() - grid.dl[2].get() * k)
                    * np.ones((volume.shape[1], volume.shape[0])),
                    x=x,
                    y=y,
                    surfacecolor=(volume[:, :, nb_frames - 1 - k].T),
                    colorscale="Gray",
                    cmin=np.amin(vol),
                    cmax=np.amax(vol),
                ),
                name=str(
                    k
                ),  # you need to name the frame for the animation to behave properly
            )
            for k in range(nb_frames)
        ]
    )

    fig1.add_trace(
        go.Surface(
            z=grid.l[2].get() * np.ones((volume.shape[1], volume.shape[0])),
            x=x,
            y=y,
            surfacecolor=(volume[:, :, 0].T),
            cmin=np.amin(vol),
            cmax=np.amax(vol),
            colorscale="Gray",
            colorbar=dict(thickness=20, ticklen=4),
        )
    )

    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(fig1.frames)
            ],
        }
    ]

    # Layout
    fig1.update_layout(
        title="Slices in volumetric data",
        width=600,
        height=600,
        scene=dict(
            #                        xaxis=dict(range=[0, 64], autorange=False),
            #                        yaxis=dict(range=[0, 64], autorange=False),
            xaxis=dict(range=[-grid.dl[0].get(), grid.l[0].get()], autorange=False),
            yaxis=dict(range=[-grid.dl[1].get(), grid.l[1].get()], autorange=False),
            zaxis=dict(range=[-grid.dl[2].get(), grid.l[2].get()], autorange=False),
            aspectratio=dict(
                x=grid.l[0].get() / cp.amax(grid.l).get(),
                y=grid.l[1].get() / cp.amax(grid.l).get(),
                z=grid.l[2].get() / cp.amax(grid.l).get(),
            ),
        ),
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, frame_args(50)],
                        "label": "&#9654;",  # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;",  # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
        sliders=sliders,
    )

    fig1.show()


def plot_3D(grid, dens):
    X = grid.grid[0].get()
    Y = grid.grid[1].get()
    Z = grid.grid[2].get()

    fig2 = go.Figure(
        data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=dens.flatten(),
            isomin=np.amin(dens),
            isomax=np.amax(dens),
            colorscale="Gray",
            opacity=0.1,  # needs to be small to see through all surfaces
            surface_count=17,  # needs to be a large number for good volume rendering
        )
    )
    fig2.show()
