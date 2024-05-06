from microgen import Rve
import pyvista as pv
import os
import time
import fedoo as fd
import numpy as np

cell_size = 1.0

rve = Rve(dim=cell_size, center=(0.0, 0.0, 0.0))
volume = cell_size * cell_size * cell_size

fd.ModelingSpace("3D")

filename = "tension_test_small_strain"
res_dir = "results/"
res_dir_vtk = "results_vtk/"

mesh_file = "remeshed_graded.vtk"
mesh = fd.Mesh.read(mesh_file)

material = fd.constitutivelaw.ElasticIsotrop(1e3, 0.3)
wf = fd.weakform.StressEquilibrium(material)

crd = mesh.nodes
center = [np.linalg.norm(crd, axis=1).argmin()]

# Assembly
assemb = fd.Assembly.create(wf, mesh, mesh.elm_type, name="Assembly")

xmin = np.min(crd[0, :])
xmax = np.max(crd[0, :])

# node set for boundary conditions
left = mesh.find_nodes("X", -1.5)
right = mesh.find_nodes("X", 1.5)

# add CD nodes
# ref_node = mesh.add_nodes(2) #reference node for rigid body motion
ref_node = mesh.add_virtual_nodes(2)  # reference node for rigid body motion
bounds = mesh.bounding_box
print(bounds)
mesh.nodes[ref_node[0], :] = [0.0, 0.0, 0.0]
mesh.nodes[ref_node[1], :] = [0.0, 0.0, 0.0]
node_cd = [ref_node[0] for i in range(3)] + [ref_node[1] for i in range(3)]
var_cd = ["DispX", "DispY", "DispZ", "DispX", "DispY", "DispZ"]

pb = fd.problem.Linear(assemb)

pb.bc.add(
    "Dirichlet", left, ["DispX", "DispY", "DispZ"], [0.0, 0.0, 0.0]
)  # Displacement of the left end
pb.bc.add("Dirichlet", right, ["DispX", "DispY", "DispZ"], [0.05, 0.0, 0.0], name="ux")
pb.apply_boundary_conditions()


# create a 'result' folder and set the desired outputs
if not (os.path.isdir("results")):
    os.mkdir("results")
if not (os.path.isdir("results_vtk")):
    os.mkdir("results_vtk")

results_vtk = pb.add_output(
    res_dir_vtk + filename,
    "Assembly",
    ["Disp", "Stress", "Strain", "Fext"],
    output_type="Node",
    file_format="vtk",
)

t0 = time.time()
print("Solving...")
pb.solve()
print("Done in " + str(time.time() - t0) + " seconds")

pb.save_results()
