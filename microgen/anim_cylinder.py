from __future__ import annotations

from functools import partial
from itertools import product
from typing import Callable

import numpy as np
import numpy.typing as npt
import pyvista as pv
from tqdm import tqdm

from microgen import CylindricalTpms
from microgen.shape.surface_functions import gyroid, schwarz_d

MAX_TIME = 1.0
N_FRAMES_PER_FUNC = 40


def morph(
    phi: list[Callable],
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    time: float,
) -> npt.NDArray[np.float64]:
    result = np.zeros_like(x)
    for index, surface_function in enumerate(phi):
        weight_func = time if index == 1 else MAX_TIME - time
        result += weight_func * surface_function(x, y, z)
    return result


def grading(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    _: npt.NDArray[np.float64],
    min_offset: float | npt.NDArray[np.float64] = 0.0,
    max_offset: float | npt.NDArray[np.float64] = 3.0,
) -> npt.NDArray[np.float64]:
    rad = np.sqrt(x**2 + y**2)
    max_rad = np.max(rad)
    min_rad = np.min(rad)

    return min_offset + (max_offset - min_offset) * (rad - min_rad) / (
        max_rad - min_rad
    )


def swapped_gyroid(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute swapped gyroid function."""
    return gyroid(x=z, y=y, z=x)


pl = pv.Plotter(off_screen=True)

pl.open_gif("morph.gif")
# pl.open_video("morph.mp4")

funcs = [
    gyroid,
    swapped_gyroid,
    schwarz_d,
    gyroid,
]

radius = 4.0
factor = 2.0
for i, time in tqdm(
    list(
        product(
            range(len(funcs) - 1),
            np.linspace(0, MAX_TIME, N_FRAMES_PER_FUNC),
        ),
    ),
):
    phi = funcs[i : i + 2]
    full_density_offset = CylindricalTpms.offset_from_density(
        surface_function=lambda x, y, z: morph(phi, x, y, z, time),
        part_type="sheet",
        density=1.0,
    )

    current_frame = i * N_FRAMES_PER_FUNC + time
    radius = 4 + factor * (1 + np.cos(2 * np.pi * current_frame / 3.0 + np.pi))

    geometry = CylindricalTpms(
        radius=radius,
        surface_function=lambda x, y, z: morph(phi, x, y, z, time),
        offset=partial(
            grading,
            min_offset=0.0,
            max_offset=0.9 * full_density_offset,
        ),
        repeat_cell=(5, 0, 1),
        resolution=20,
    )
    shape = geometry.grid_sheet

    dists = np.linalg.norm(shape.points, axis=1)

    shape["distances"] = dists
    actor = pl.add_mesh(
        shape,
        color="w",
        scalars="distances",
        clim=[0.0, 2.0 * radius],
        show_scalar_bar=False,
    )

    pl.write_frame()
    pl.remove_actor(actor)

pl.close()
