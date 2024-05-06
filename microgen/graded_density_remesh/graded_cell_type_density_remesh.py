import numpy as np
from microgen import Tpms
import pyvista as pv
from microgen.remesh import remesh_keeping_periodicity_for_fem
from microgen.shape.surface_functions import honeycomb, gyroid, neovius, schoenIWP
from microgen import BoxMesh

repeat = 4


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


def double_tanh_graded_density(
    x, y, z, repeat_cell, min_offset, max_offset, transition_width=0.3
):

    # Calculate the range of x values
    x_range = np.linspace(np.min(x), np.max(x), num=repeat_cell[0] * 2)
    cell_size = 1.0

    # Calculate the transition limits symmetrically around zero
    x_center_1 = x_range[1]
    x_center_2 = x_range[-2]
    x_scaled_1 = (x - x_center_1) / (
        cell_size / 2.0
    )  # Scale x symmetrically around zero
    x_scaled_2 = (x - x_center_2) / (
        cell_size / 2.0
    )  # Scale x symmetrically around zero

    # Applying tanh function to create a non-linear gradient
    tanh_gradient_1 = -0.5 * np.tanh(x_scaled_1 * np.exp(1) / transition_width) + 0.5
    tanh_gradient_2 = 0.5 * np.tanh(x_scaled_2 * np.exp(1) / transition_width) + 0.5

    # Calculate the density values
    density_values = (max_offset - min_offset) * (
        tanh_gradient_1 + tanh_gradient_2
    ) + min_offset

    return density_values


def tanh_graded_cell(x, y, z, transition_width=2.0):

    # Calculate the range of x values
    x_range = 1.0

    # Calculate the transition limits symmetrically around zero
    x_center = (np.max(x) + np.min(x)) / 2.0
    x_scaled = (x - x_center) / (x_range / 2.0)  # Scale x symmetrically around zero

    # Applying tanh function to create a non-linear gradient
    tanh_gradient = 0.5 * np.tanh(x_scaled * np.exp(1) / transition_width) + 0.5

    return tanh_gradient


def weight(x, y, z, bounds, index, transition_width):
    denom = 0.0
    #    for value in bounds:
    #        denom += tanh_graded_density(x, y, z, transition_width)
    #    value = bounds[index]
    #    return tanh_graded_density(x, y, z)
    if index == 0:
        return tanh_graded_cell(x, y, z, transition_width)
    else:
        return 1.0 - tanh_graded_cell(x, y, z, transition_width)


def multi_morph(phi, x, y, z, bounds, transition_width):
    result = 0.0
    wtot = 0.0
    for index, surface_function in enumerate(phi):
        weight_func = weight(x, y, z, bounds, index, transition_width)
        wtot += weight_func
        result += weight_func * surface_function(x, y, z)
    return result


def digraded(x, y, z, bounds):
    transition_width = 3
    return multi_morph(
        phi=[schoenIWP, gyroid],
        x=x,
        y=y,
        z=z,
        bounds=bounds,
        transition_width=transition_width,
    )


# Generate sample x, y, z points
# x = np.linspace(0, 10, 1000)
# y = np.linspace(0, 10, 100)
# z = np.linspace(0, 10, 100)

# Calculate bounds
# x_min = np.min(x)
# x_max = np.max(x)
bounds = [-2, 2]
repeat_cell = (3, 1, 1)

initial_graded = pv.UnstructuredGrid(
    Tpms(
        surface_function=gyroid,
        offset=lambda x, y, z: double_tanh_graded_density(
            x, y, z, repeat_cell, 0.5, 3.0, 0.9
        ),
        cell_size=1.0,
        repeat_cell=repeat_cell,
        resolution=30,
    ).generateVtk(type_part="sheet")
)

input_box_mesh = BoxMesh.from_pyvista(initial_graded)
print(input_box_mesh.rve.dim)
initial_graded.save("graded_cell_type_initial.vtk")

max_element_edge_length = 0.4
remeshed_graded = remesh_keeping_periodicity_for_fem(
    initial_graded, hmax=max_element_edge_length, tol=1.0e3
)
# remeshed_graded.plot(color="white", show_edges=True, screenshot="remeshed_gyroid.png")

pl = pv.Plotter()
_ = pl.add_mesh(remeshed_graded, color="lightblue", show_edges=True)
# _ = pl.add_mesh(boundelm['face_Xp'][0], line_width=0.1,  show_edges=True, color='red')
# _ = pl.add_mesh(transXp.points, render_points_as_spheres=True, point_size=10, color='gray')

# _ = pl.add_mesh(boundelm['face_Yp'][0], line_width=0.1,  show_edges=True, color='blue')
# _ = pl.add_mesh(transYp.points, render_points_as_spheres=True, point_size=10, color='gray')

# _ = pl.add_mesh(boundelm['face_Zp'][0], line_width=0.1,  show_edges=True, color='green')
# _ = pl.add_mesh(transZp.points, render_points_as_spheres=True, point_size=10, color='gray')

pl.view_xy()
pl.show_axes()
pl.show()

remeshed_graded.save("remeshed_graded.vtk")
