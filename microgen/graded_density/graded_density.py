import numpy as np
import pyvista as pv

from microgen import Tpms
from microgen.shape.surface_functions import gyroid, schwarzP, neovius


def linear_graded_density(x, y, z):
    min_offset = 0.3
    max_offset = 3.0
    length = 5.0
    return (max_offset - min_offset) * z / length + 0.5 * (min_offset + max_offset)


def tanh_graded_density(x, y, z, min_offset, max_offset, transition_width=0.3):

    # Calculate the range of x values
    x_range = np.max(x) - np.min(x)

    # Calculate the transition limits symmetrically around zero
    x_center = (np.max(x) + np.min(x)) / 2.0
    x_scaled = (x - x_center) / (x_range / 2.0)  # Scale x symmetrically around zero

    # Applying tanh function to create a non-linear gradient
    tanh_gradient = np.tanh(x_scaled * np.exp(1) / transition_width)

    # Calculate the density values
    density_values = (max_offset - min_offset) / 2.0 * tanh_gradient + 0.5 * (
        max_offset + min_offset
    )

    return density_values


def double_tanh_graded_density(x, y, z, min_offset, max_offset, transition_width=0.3):

    # Calculate the range of x values
    x_range = np.max(x) - np.min(x)

    # Calculate the transition limits symmetrically around zero
    x_center_1 = (np.max(x) - np.min(x)) / 6.0 + np.min(x)
    x_center_2 = 5.0 * (np.max(x) - np.min(x)) / 6.0 + np.min(x)
    x_scaled_1 = (x - x_center_1) / (x_range / 2.0)  # Scale x symmetrically around zero
    x_scaled_2 = (x - x_center_2) / (x_range / 2.0)  # Scale x symmetrically around zero

    # Applying tanh function to create a non-linear gradient
    tanh_gradient_1 = 1.0 - np.tanh(x_scaled_1 * np.exp(1) / transition_width)
    tanh_gradient_2 = np.tanh(x_scaled_2 * np.exp(1) / transition_width)

    # Calculate the density values
    density_values = (
        (max_offset - min_offset) / 4.0 * tanh_gradient_1
        + (max_offset - min_offset) / 4.0 * tanh_gradient_2
        + 0.5 * (max_offset + min_offset)
    )

    return density_values


def circular_graded_density(x: float, y: float, z: float) -> float:
    min_offset = 0.3
    max_offset = 2.0
    radius = 2.0
    return (max_offset - min_offset) * (x**2 + y**2) / radius**2 + min_offset


def sinc_graded_density(x, y, z):
    min_offset = 0.3
    max_offset = 1.5
    length = 1.0

    # Apply sinc function to create a non-linear gradient
    sinc_gradient = np.sinc(x / length)

    return (max_offset - min_offset) * sinc_gradient + 0.5 * (min_offset + max_offset)


def spiral_density(x, y, z, frequency=1.0, amplitude=1.0):
    radius = np.sqrt(x**2 + y**2)
    angle = np.arctan2(y, x)
    spiral = amplitude * np.sin(frequency * angle) / (1 + radius)
    return spiral + 0.5  # Shift to make sure the minimum density is positive


geometry = Tpms(
    surface_function=gyroid,
    offset=lambda x, y, z: double_tanh_graded_density(x, y, z, 0.5, 4.0, 0.2),
    repeat_cell=(3, 1, 1),
    cell_size=(1, 1, 1),
    resolution=30,
)


shape = geometry.generateVtk(type_part="sheet")
shape.save("gyroid.vtk")
pl = pv.Plotter()
_ = pl.add_mesh(shape, color="lightblue", lighting=True, show_edges=True)
pl.show()
